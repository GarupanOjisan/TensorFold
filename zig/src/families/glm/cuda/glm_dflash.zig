//! DFlash2 drafts for GLM-5.3-Flash on two CUDA ranks (v1.8 dflash2.py's Drafter); notes/glm_dflash.md#overview
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const kern = @import("glm_kernels.zig");
const fwd = @import("glm_forward.zig");
const tri = @import("glm_triton.zig");

const dchain = @import("glm_dflash_chain.zig");
const batch = @import("glm_dflash_batch.zig");
const Engine = fwd.Engine;
pub const bf = 2;
pub const f4 = 4;

/// Rows of a context update that replay one CUDA graph (engine.MAX_ROWS; larger updates run in pieces of this).
pub const nmax = 16;
/// The widest block (config block_size), and the widest candidate list and selector the buffers are sized for.
pub const max_block = 16;

pub fn int(x: usize) i32 {
    return @intCast(x);
}

pub const Sampling = dchain.Sampling;
pub const Policy = dchain.Policy;

/// A block pass timed by piece (`profile`): the wall time of each stretch between two syncs, by label.
pub const Prof = struct {
    io: std.Io,
    last: i96 = 0,
    ms: [8]f64 = @splat(0),
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    dr: *const wt.Draft,
    w: *const wt.Weights,
    capacity: usize,
    ring: usize, // 0: a flat buffer, logical row p at p
    cap: usize, // rows a cache holds
    mem: cuda.DeviceBuffer,
    kc: [8]u64 = @splat(0),
    vc: [8]u64 = @splat(0),
    pos_dev: u64,
    cache: u64, // the K and V caches, layer by layer (the start of `mem`, or the drafter's slot of a Batch's pool)
    ids: u64,
    context_end: usize = 0,
    // scratch (nmax rows unless noted)
    tap_in: u64,
    fc_xs: u64,
    fc_out: u64,
    ctx: u64,
    ctx_xs: u64,
    kvo: u64,
    kk: u64,
    vv: u64,
    cos: u64,
    sin: u64,
    x: u64,
    normed: u64,
    xs: u64,
    dyn: u64,
    cv: u64,
    cv_xs: u64,
    qkv: u64,
    qq: u64,
    att: u64,
    att_xs: u64,
    part: u64,
    gath: u64,
    rowb: u64,
    gu: u64,
    act: u64,
    act_xs: u64,
    hh: u64, // block - 1 rows
    hh_xs: u64,
    logits: u64,
    cand_loc: u64,
    cand_all: u64,
    proj: u64,
    cand_host: []u32,
    proj_host: []f32,
    cand_tokens: []u32,
    cand_values: []f64,
    cand_proj: []f64,
    snap_bytes: usize,
    prof: ?*Prof = null, // TF_GLM_DF_PROF: eager block passes timed by piece

    /// dflash2.Drafter.__init__: this rank's buffers and caches (a ring when shorter); notes/glm_dflash.md#drafter-init
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wt.Weights, capacity: usize) !Drafter {
        if (w.draft == null) return error.NoDraftModel;
        const dr = &w.draft.?;
        if (dr.block > max_block or dr.layers.len > 8 or dr.top_k > 32 or dr.sel_rank > 512) return error.DraftTooWide;
        const ring_rows = ((dr.window + 1 + dr.block + 63 + 63) / 64) * 64; // geometry.draft_ring_rows(window, block); window is sliding_window - 1
        const use_ring = if (std.c.getenv("TF_GLM_DRAFT_RING")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        const ring: usize = if (use_ring and ring_rows < capacity + dr.block) ring_rows else 0;
        const cap = if (ring != 0) ring else capacity + dr.block;
        const D = dr.d;
        const hd = dr.hd;
        const nb = dr.block - 1;
        const vl = w.cfg.vocab / wt.world;
        const kvr = 2 * dr.kvh * hd;
        const cand_n = nb * 2 * dr.top_k;
        var dft: Drafter = undefined;
        dft.gpa = gpa;
        dft.dr = dr;
        dft.w = w;
        dft.capacity = capacity;
        dft.ring = ring;
        dft.cap = cap;
        dft.context_end = 0;
        dft.prof = null; // `dft` starts undefined: an unset prof is garbage, and tick() would synchronize inside a captured block pass
        dft.kc = @splat(0);
        dft.vc = @splat(0);
        // one allocation: the K and V caches, then the scratch
        const cache_bytes = 2 * dr.layers.len * dr.kvh * cap * hd * bf;
        var off: usize = 0;
        const sizes = [_]usize{
            nmax * 5 * D * bf, // tap_in
            nmax * (5 * D / 64) * f4, // fc_xs
            nmax * D * bf, // fc_out
            nmax * D * bf, // ctx
            nmax * (D / 64) * f4, // ctx_xs
            nmax * kvr * bf, // kvo
            dr.kvh * nmax * hd * bf, // kk
            dr.kvh * nmax * hd * bf, // vv
            nmax * (hd / 2) * f4, // cos
            nmax * (hd / 2) * f4, // sin
            nmax * D * bf, // x
            nmax * D * bf, // normed
            nmax * (D / 64) * f4, // xs
            nmax * (4 * D / dr.gs) * bf, // dyn
            nmax * D * bf, // cv
            nmax * (D / 64) * f4, // cv_xs
            nmax * (dr.heads + 2 * dr.kvh) * hd * bf, // qkv
            dr.heads * nmax * hd * bf, // qq
            nmax * dr.heads * hd * bf, // att
            nmax * (dr.heads * hd / 64) * f4, // att_xs
            nmax * D * f4, // part
            wt.world * nmax * D * f4, // gath
            nmax * D * bf, // rowb
            nmax * 2 * dr.inter * bf, // gu
            nmax * dr.inter * bf, // act
            nmax * (dr.inter / 64) * f4, // act_xs
            nb * D * bf, // hh
            nb * (D / 64) * f4, // hh_xs
            nb * vl * bf, // logits
            cand_n * 4, // cand_loc
            wt.world * cand_n * 4, // cand_all
            nb * dr.sel_rank * f4, // proj
            4 * max_block, // ids
            8, // pos_dev
        };
        var at: [sizes.len + 1]usize = undefined;
        off = cache_bytes;
        for (sizes, 0..) |s, i| {
            off = std.mem.alignForward(usize, off, 256);
            at[i] = off;
            off += s;
        }
        dft.mem = try cuda.DeviceBuffer.alloc(d, off + 256);
        errdefer dft.mem.free();
        try dft.mem.fill8(0, null);
        const base = dft.mem.ptr;
        var c: usize = 0;
        for (0..dr.layers.len) |i| {
            dft.kc[i] = base + c;
            c += dr.kvh * cap * hd * bf;
            dft.vc[i] = base + c;
            c += dr.kvh * cap * hd * bf;
        }
        inline for (.{ "tap_in", "fc_xs", "fc_out", "ctx", "ctx_xs", "kvo", "kk", "vv", "cos", "sin", "x", "normed", "xs", "dyn", "cv", "cv_xs", "qkv", "qq", "att", "att_xs", "part", "gath", "rowb", "gu", "act", "act_xs", "hh", "hh_xs", "logits", "cand_loc", "cand_all", "proj" }, 0..) |name, i| {
            @field(dft, name) = base + at[i];
        }
        dft.ids = base + at[sizes.len - 2];
        dft.pos_dev = base + at[sizes.len - 1];
        dft.cache = base;
        // the mask ids stay: only row 0 changes
        const ids: [max_block]i32 = @splat(@intCast(dr.mask_id));
        try upload(d, dft.ids, std.mem.sliceAsBytes(ids[0..dr.block]));
        dft.cand_host = try gpa.alloc(u32, wt.world * cand_n);
        errdefer gpa.free(dft.cand_host);
        dft.proj_host = try gpa.alloc(f32, nb * dr.sel_rank);
        errdefer gpa.free(dft.proj_host);
        dft.cand_tokens = try gpa.alloc(u32, nb * dr.top_k);
        errdefer gpa.free(dft.cand_tokens);
        dft.cand_values = try gpa.alloc(f64, nb * dr.top_k);
        errdefer gpa.free(dft.cand_values);
        dft.cand_proj = try gpa.alloc(f64, nb * dr.sel_rank);
        errdefer gpa.free(dft.cand_proj);
        dft.snap_bytes = if (ring != 0) cache_bytes + 8 else 0;
        return dft;
    }

    pub fn deinit(self: *Drafter) void {
        self.gpa.free(self.cand_host);
        self.gpa.free(self.proj_host);
        self.gpa.free(self.cand_tokens);
        self.gpa.free(self.cand_values);
        self.gpa.free(self.cand_proj);
        self.mem.free();
    }

    fn upload(d: *const cuda.Driver, dst: u64, bytes: []const u8) !void {
        try d.check(d.api.cuMemcpyHtoD_v2(dst, bytes.ptr, bytes.len), "cuMemcpyHtoD");
    }

    /// dflash2.Drafter.reset: the context empty.
    pub fn reset(self: *Drafter, e: *Engine) !void {
        self.context_end = 0;
        try e.zero(self.pos_dev, 8);
    }

    /// Bytes a kept prompt state holds of the drafter: its caches (a ring; a flat one keeps no state) and position.
    pub fn stateBytes(self: *const Drafter) usize {
        return self.snap_bytes;
    }

    pub fn save(self: *const Drafter, e: *Engine, dst: u64) !void {
        if (self.snap_bytes == 0) return;
        try e.copyDev(dst, self.cache, self.snap_bytes - 8);
        try e.copyDev(dst + self.snap_bytes - 8, self.pos_dev, 8);
    }

    pub fn restore(self: *Drafter, e: *Engine, src: u64, at: usize) !void {
        if (self.snap_bytes == 0) return error.NoDrafterState;
        try e.copyDev(self.cache, src, self.snap_bytes - 8);
        try e.copyDev(self.pos_dev, src + self.snap_bytes - 8, 8);
        self.context_end = at;
    }

    /// Moves the caches and their contents to `cache`; the position stays put; notes/glm_dflash.md#adopt
    pub fn adopt(self: *Drafter, e: *Engine, cache: u64) !void {
        try e.copyDev(cache, self.cache, self.cacheBytes());
        const half = self.cacheBytes() / (2 * self.dr.layers.len);
        for (0..self.dr.layers.len) |i| {
            self.kc[i] = cache + 2 * i * half;
            self.vc[i] = self.kc[i] + half;
        }
        self.cache = cache;
    }

    pub fn cacheBytes(self: *const Drafter) usize {
        return 2 * self.dr.layers.len * self.dr.kvh * self.cap * self.dr.hd * bf;
    }

    // -- pieces -------------------------------------------------------------------------------------------------
    fn tick(self: *const Drafter, e: *Engine, which: usize) !void {
        const p = self.prof orelse return;
        try e.s.synchronize();
        const t = std.Io.Clock.awake.now(p.io).toNanoseconds();
        p.ms[which] += @as(f64, @floatFromInt(t - p.last)) / 1e6;
        p.last = t;
    }

    /// Eager block passes timed by piece and printed (both ranks run it alike); notes/glm_dflash.md#profile
    pub fn profile(self: *Drafter, e: *Engine, io: std.Io) !void {
        var p: Prof = .{ .io = io };
        self.prof = &p;
        defer self.prof = null;
        const reps = 6;
        for (0..reps + 1) |r| {
            if (r == 1) p.ms = @splat(0);
            try e.s.synchronize();
            p.last = std.Io.Clock.awake.now(io).toNanoseconds();
            try self.blockBody(e);
            try self.tick(e, 6);
        }
        if (e.w.rank == 0) std.debug.print("[tf-glm] DFlash2 block pass by piece (ms, eager): embed+rope {d:.2}, layers: input {d:.2}, prep+attention {d:.2}, o {d:.2}, mlp up {d:.2}, down {d:.2}; head+topk+gather+proj {d:.2}\n", .{ p.ms[0] / reps, p.ms[1] / reps, p.ms[2] / reps, p.ms[3] / reps, p.ms[4] / reps, p.ms[5] / reps, p.ms[6] / reps });
    }
    fn mm(self: *const Drafter, e: *Engine, x: u64, xs: u64, q: kern.Q4, out: u64, rows: usize, fp32: bool) !void {
        _ = self;
        try kern.qmmDecode(e.k, e.s, x, q.k, xs, q, out, rows, fp32);
    }

    fn norm(self: *const Drafter, e: *Engine, x: u64, weight: u64, out: u64, xs: u64, rows: usize) !void {
        try e.t.rmsnorm(x, self.dr.d, weight, out, self.dr.d, xs, rows, self.dr.d, self.dr.eps);
    }

    fn rotary(self: *const Drafter, e: *Engine, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(self.pos_dev);
        a.add(self.dr.inv_freq);
        a.add(int(rows));
        a.add(int(self.dr.hd / 2));
        a.add(self.cos);
        a.add(self.sin);
        try cuda.launch.launch(e.k.df_rope, .{ .grid = .{ .x = @intCast((rows * self.dr.hd / 2 + 127) / 128) }, .block = .{ .x = 128 } }, e.s, &a);
    }

    pub fn scatter(self: *const Drafter, e: *Engine, src: u64, cache: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(cache);
        a.add(self.pos_dev);
        a.add(int(rows));
        a.add(int(self.cap));
        a.add(int(self.ring));
        a.add(int(self.dr.hd));
        try cuda.launch.launch(e.k.df_scatter, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(self.dr.kvh) }, .block = .{ .x = 128 } }, e.s, &a);
    }

    /// `_row`: a row-parallel projection's fp32 partials added in rank order, then bf16, into rowb.
    fn row(self: *const Drafter, e: *Engine, x: u64, xs: u64, q: kern.Q4, rows: usize) !void {
        try self.mm(e, x, xs, q, self.part, rows, true);
        const n = rows * self.dr.d;
        try e.comm.allGather(e.s, self.part, self.gath, n);
        var a: cuda.Args = .{};
        a.add(self.gath);
        a.add(int(wt.world));
        a.add(int(n));
        a.add(self.rowb);
        try cuda.launch.launch(e.k.df_rowsum, .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, e.s, &a);
    }

    /// `_layer`: the block's (or a context update's) rows through layer i, in x.
    fn layer(self: *const Drafter, e: *Engine, i: usize, rows: usize) !void {
        const dr = self.dr;
        const L = dr.layers[i];
        const D = dr.d;
        try self.norm(e, self.x, L.in_norm, self.normed, self.xs, rows);
        try self.mm(e, self.normed, self.xs, L.a_kp, self.dyn, rows, false);
        try e.t.dconv(self.normed, self.dyn, L.a_base, null, self.cv, rows, D, dr.gs, 0);
        try e.t.groupSums(self.cv, D, self.cv_xs, rows, D);
        try self.mm(e, self.cv, self.cv_xs, L.qkv, self.qkv, rows, false);
        try self.tick(e, 1);
        const kbuf = self.kk;
        try e.t.draftPrep(self.qkv, L.q_norm, L.k_norm, self.cos, self.sin, self.qq, kbuf, self.vv, rows, (dr.heads + 2 * dr.kvh) * dr.hd, dr.eps, dr.heads, dr.kvh, dr.hd);
        try self.scatter(e, kbuf, self.kc[i], rows);
        try self.scatter(e, self.vv, self.vc[i], rows);
        try e.t.draftAttention(self.qq, self.kc[i], self.vc[i], self.att, self.pos_dev, dr.window, rows, dr.heads, dr.kvh, dr.hd, self.cap, dr.causal, self.ring != 0);
        try self.tick(e, 2);
        try e.t.groupSums(self.att, dr.heads * dr.hd, self.att_xs, rows, dr.heads * dr.hd);
        try self.row(e, self.att, self.att_xs, L.o, rows);
        try e.t.dconv(self.rowb, self.dyn, L.a_base, self.x, self.x, rows, D, dr.gs, 1);
        try self.tick(e, 3);
        try self.norm(e, self.x, L.post_norm, self.normed, self.xs, rows);
        try self.mm(e, self.normed, self.xs, L.m_kp, self.dyn, rows, false);
        try e.t.dconv(self.normed, self.dyn, L.m_base, null, self.cv, rows, D, dr.gs, 0);
        try e.t.groupSums(self.cv, D, self.cv_xs, rows, D);
        try self.mm(e, self.cv, self.cv_xs, L.gu, self.gu, rows, false);
        try e.t.swiglu(self.gu, self.act, self.act_xs, rows, dr.inter, 1e30);
        try self.tick(e, 4);
        try self.row(e, self.act, self.act_xs, L.down, rows);
        try e.t.dconv(self.rowb, self.dyn, L.m_base, self.x, self.x, rows, D, dr.gs, 1);
        try self.tick(e, 5);
    }

    // -- the context --------------------------------------------------------------------------------------------
    fn tapsBody(self: *Drafter, e: *Engine, n: usize) !void {
        const dr = self.dr;
        const D = dr.d;
        try e.t.groupSums(self.tap_in, 5 * D, self.fc_xs, n, 5 * D);
        try self.mm(e, self.tap_in, self.fc_xs, dr.fc, self.fc_out, n, false);
        try self.norm(e, self.fc_out, dr.hidden_norm, self.ctx, self.ctx_xs, n);
        try self.rotary(e, n);
        try e.t.groupSums(self.ctx, D, self.ctx_xs, n, D);
        for (dr.layers, 0..) |L, i| {
            try self.mm(e, self.ctx, self.ctx_xs, L.kv, self.kvo, n, false);
            try e.t.draftPrep(self.kvo, L.q_norm, L.k_norm, self.cos, self.sin, self.kvo, self.kk, self.vv, n, 2 * dr.kvh * dr.hd, dr.eps, 0, dr.kvh, dr.hd);
            try self.scatter(e, self.kk, self.kc[i], n);
            try self.scatter(e, self.vv, self.vc[i], n);
        }
        var a: cuda.Args = .{};
        a.add(self.pos_dev);
        a.add(int(n));
        try cuda.launch.launch(e.k.df_pos_add, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, e.s, &a);
    }

    fn run(self: *Drafter, e: *Engine, head: fwd.GraphKey.Head, rows: usize, comptime body: anytype, args: anytype) !void {
        _ = self;
        if (e.use_graphs) return e.replay(.{ .lane = e.lane, .head = head, .rows = @intCast(rows), .cur = 0, .np = 0 }, body, args); // the bound lane's drafter
        return @call(.auto, body, args);
    }

    /// dflash2.Drafter.add_taps for the first `n` rows of `b`'s taps (the committed rows' stream means at five layers).
    pub fn addTaps(self: *Drafter, e: *Engine, b: *const fwd.Buffers, n: usize) !void {
        const D = self.dr.d;
        var start: usize = 0;
        while (start < n) {
            const m = @min(nmax, n - start);
            if (self.context_end + m > self.capacity) return error.DrafterPastCapacity;
            var a: cuda.Args = .{};
            for (b.taps) |t| a.add(t + start * D * bf);
            a.add(int(D));
            a.add(self.tap_in);
            try cuda.launch.launch(e.k.df_cat, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 256 } }, e.s, &a);
            try self.run(e, .dtaps, m, tapsBody, .{ self, e, m });
            self.context_end += m;
            start += m;
        }
    }

    // -- drafts -------------------------------------------------------------------------------------------------
    fn blockBody(self: *Drafter, e: *Engine) !void {
        const dr = self.dr;
        const n = dr.block;
        const D = dr.d;
        try e.t.embed(self.ids, self.w.embed, self.x, n, D, 1);
        try self.rotary(e, n);
        try self.tick(e, 0);
        for (0..dr.layers.len) |i| try self.layer(e, i, n);
        try self.norm(e, self.x + D * bf, dr.norm, self.hh, self.hh_xs, n - 1);
        try kern.q4Matmul(e.k, e.s, self.hh, self.hh_xs, self.w.draft_head, self.logits, n - 1);
        const vl = self.w.head.n;
        var a: cuda.Args = .{};
        a.add(self.logits);
        a.add(int(vl));
        a.add(int(vl));
        a.add(int(dr.top_k));
        a.add(int(e.w.rank * vl));
        a.add(self.cand_loc);
        try cuda.launch.launch(e.k.topk, .{ .grid = .{ .x = @intCast(n - 1) }, .block = .{ .x = 1024 } }, e.s, &a);
        try e.comm.allGather(e.s, self.cand_loc, self.cand_all, (n - 1) * 2 * dr.top_k);
        var p: cuda.Args = .{};
        p.add(self.hh);
        p.add(dr.hproj);
        p.add(self.proj);
        p.add(int(dr.sel_rank));
        p.add(int(D));
        try cuda.launch.launch(e.k.df_proj, .{ .grid = .{ .x = @intCast((dr.sel_rank + 7) / 8), .y = @intCast(n - 1) }, .block = .{ .x = 256 } }, e.s, &p);
    }

    /// dflash2.Drafter.candidates: the block pass for `pending`, `read`: also to the host; notes/glm_dflash.md#block
    pub fn block(self: *Drafter, e: *Engine, pending: u32, read: bool) !void {
        const id: i32 = @intCast(pending);
        try e.upload(self.ids, std.mem.asBytes(&id));
        try self.run(e, .dblock, self.dr.block, blockBody, .{ self, e });
        if (!read) return;
        const dr = self.dr;
        const nb = dr.block - 1;
        // v1.10 patch 0092: a bad id is a copy read early; copy again once idle; notes/glm_dflash.md#patch-0092
        for (0..2) |attempt| {
            try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(self.cand_host.ptr, self.cand_all, wt.world * nb * 2 * dr.top_k * 4, e.s.handle), "cuMemcpyDtoHAsync");
            try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(self.proj_host.ptr, self.proj, nb * dr.sel_rank * f4, e.s.handle), "cuMemcpyDtoHAsync");
            try e.s.synchronize();
            if (self.merge(nb)) return;
            if (attempt == 1) return error.BadDrafterCandidates;
            std.log.warn("drafter: a candidate id outside the vocabulary; copying the rows again once the device is idle", .{});
            try e.d.check(e.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
        }
    }

    /// merge_candidates: each row's rank lists (value and id pairs) merged by value then id, the top k.
    pub fn merge(self: *Drafter, nb: usize) bool {
        const dr = self.dr;
        const k = dr.top_k;
        const Cand = struct { v: f32, id: u32 };
        var all: [wt.world * 32]Cand = undefined;
        for (0..nb) |r| {
            var m: usize = 0;
            for (0..wt.world) |rank| {
                const base = (rank * nb + r) * 2 * k;
                for (0..k) |j| {
                    all[m] = .{ .v = @bitCast(self.cand_host[base + 2 * j]), .id = self.cand_host[base + 2 * j + 1] };
                    if (all[m].id >= self.w.cfg.vocab) return false;
                    m += 1;
                }
            }
            std.mem.sort(Cand, all[0..m], {}, struct {
                fn less(_: void, a: Cand, b: Cand) bool {
                    return a.v > b.v or (a.v == b.v and a.id < b.id);
                }
            }.less);
            for (0..k) |j| {
                self.cand_tokens[r * k + j] = all[j].id;
                self.cand_values[r * k + j] = all[j].v;
            }
            for (0..dr.sel_rank) |j| self.cand_proj[r * dr.sel_rank + j] = self.proj_host[r * dr.sel_rank + j];
        }
        return true;
    }

    pub const chain = dchain.chain;

    /// dflash2.Drafter.propose after `block`: up to `depth` drafts for the positions after `pending`.
    pub fn propose(self: *Drafter, e: *Engine, pending: u32, depth: usize, sampling: ?Sampling, policy: Policy, out: []u32) !usize {
        const d = @min(depth, self.dr.block - 1);
        if (d < 1 or self.context_end == 0) return 0;
        try self.block(e, pending, true);
        return self.chain(d, pending, self.context_end + 1, sampling, policy, out);
    }
};

pub const Batch = batch.Batch;
