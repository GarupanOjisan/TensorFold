//! Block passes of several lanes' DFlash2 drafters in one launch sequence (glm_dflash.zig's Batch).
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const kern = @import("glm_kernels.zig");
const fwd = @import("glm_forward.zig");
const df = @import("glm_dflash.zig");

const Engine = fwd.Engine;
const Drafter = df.Drafter;
const bf = df.bf;
const f4 = df.f4;
const max_block = df.max_block;
const int = df.int;

/// Block passes of several lanes' drafters in one launch sequence (v1.10 MultiDrafter); notes/glm_dflash.md#batch
pub const Batch = struct {
    gpa: std.mem.Allocator,
    dr: *const wt.Draft,
    w: *const wt.Weights,
    lanes: usize, // blocks a pass holds at most
    mem: cuda.DeviceBuffer,
    ids: u64,
    x: u64,
    normed: u64,
    xs: u64,
    dyn: u64,
    cv: u64,
    cv_xs: u64,
    qkv: u64,
    qq: u64,
    kk: u64,
    vv: u64,
    cos: u64,
    sin: u64,
    att: u64,
    att_xs: u64,
    part: u64,
    gath: u64,
    rowb: u64,
    gu: u64,
    act: u64,
    act_xs: u64,
    hh: u64,
    hh_xs: u64,
    logits: u64,
    cand_loc: u64,
    cand_all: u64,
    proj: u64,
    cand_host: []u32,
    proj_host: []f32,
    // with a ring (adopt): every lane's caches in one pool, so a layer's cache writes and attention are one launch each
    pool: ?cuda.DeviceBuffer,
    lane_bytes: usize, // a drafter's caches (Drafter.cacheBytes)
    cap0: usize, // their rows (Drafter.cap)
    meta: u64, // the blocks' positions, i64 x lanes
    slots: u64, // the blocks' slots (rows of cap), i64 x lanes
    tap_in: u64, // context updates of several lanes (taps): their rows' five taps side by side
    fc_xs: u64,
    tpos: u64, // each stacked row's position, then (tmax rows on) its first cache row (dest of tf_glm_df_scatter_rows)

    /// The 4-bit projections' widest piece (kern.qmmDecode, kern.q4Matmul).
    const piece = 16;
    /// The most stacked rows a context update of several lanes takes (a shared window's rows).
    const tmax = fwd.Engine.shared_rows;

    /// MultiDrafter's context update of several lanes in one pass; false: not done; notes/glm_dflash.md#batch-taps
    pub fn taps(b: *Batch, e: *Engine, ds: []const *Drafter, kept: []const usize, row0s: []const usize, buf: *const fwd.Buffers) !bool {
        const pl = b.pool orelse return false;
        const D = b.dr.d;
        var T: usize = 0;
        for (kept) |k| T += k;
        if (T == 0 or T > tmax) return false;
        var tab: [2 * tmax]i64 = undefined;
        const per = ds[0].cap * b.dr.hd * bf;
        var o: usize = 0;
        for (ds, kept, row0s) |dft, k, r0| {
            if (dft.context_end + k > dft.capacity) return error.DrafterPastCapacity;
            const slot = (dft.cache - pl.ptr) / per;
            for (0..k) |r| {
                const p = dft.context_end + r;
                tab[o + r] = @intCast(p);
                tab[tmax + o + r] = @intCast(slot * dft.cap + (if (dft.ring != 0) p % dft.ring else p));
            }
            if (k > 0) {
                var a: cuda.Args = .{};
                for (buf.taps) |t| a.add(t + r0 * D * bf);
                a.add(int(D));
                a.add(b.tap_in + o * 5 * D * bf);
                try cuda.launch.launch(e.k.df_cat, .{ .grid = .{ .x = @intCast(k) }, .block = .{ .x = 256 } }, e.s, &a);
            }
            o += k;
        }
        try e.upload(b.tpos, std.mem.sliceAsBytes(tab[0..]));
        if (e.use_graphs) try e.replay(.{ .lane = 0, .head = .dtbatch, .rows = @intCast(T), .cur = 0, .np = 0 }, tapsBody, .{ b, e, T }) else try b.tapsBody(e, T);
        for (ds, kept) |dft, k| {
            dft.context_end += k;
            const end: i64 = @intCast(dft.context_end);
            try e.upload(dft.pos_dev, std.mem.asBytes(&end));
        }
        return true;
    }

    fn tapsBody(b: *Batch, e: *Engine, T: usize) !void {
        const dr = b.dr;
        const D = dr.d;
        const pl = b.pool.?;
        const half = b.lane_bytes / (2 * dr.layers.len);
        try e.t.groupSums(b.tap_in, 5 * D, b.fc_xs, T, 5 * D);
        try mm(e, b.tap_in, b.fc_xs, dr.fc, b.x, T, false);
        try b.norm(e, b.x, dr.hidden_norm, b.normed, b.xs, T);
        var a: cuda.Args = .{};
        a.add(b.tpos);
        a.add(dr.inv_freq);
        a.add(int(T));
        a.add(int(dr.hd / 2));
        a.add(b.cos);
        a.add(b.sin);
        try cuda.launch.launch(e.k.df_rope_rows, .{ .grid = .{ .x = @intCast((T * dr.hd / 2 + 127) / 128) }, .block = .{ .x = 128 } }, e.s, &a);
        try e.t.groupSums(b.normed, D, b.xs, T, D);
        for (dr.layers, 0..) |L, i| {
            const kc = pl.ptr + 2 * i * half;
            try mm(e, b.normed, b.xs, L.kv, b.qkv, T, false);
            try e.t.draftPrep(b.qkv, L.q_norm, L.k_norm, b.cos, b.sin, b.qkv, b.kk, b.vv, T, 2 * dr.kvh * dr.hd, dr.eps, 0, dr.kvh, dr.hd);
            for ([_][2]u64{ .{ b.kk, kc }, .{ b.vv, kc + half } }) |sc| {
                var c: cuda.Args = .{};
                c.add(sc[0]);
                c.add(sc[1]);
                c.add(b.tpos + tmax * 8);
                c.add(int(T));
                c.add(int(b.cap0));
                c.add(int(dr.hd));
                try cuda.launch.launch(e.k.df_scatter_rows, .{ .grid = .{ .x = @intCast(T), .y = @intCast(dr.kvh) }, .block = .{ .x = 128 } }, e.s, &c);
            }
        }
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wt.Weights, lanes: usize) !Batch {
        const dr = &(w.draft orelse return error.NoDraftModel);
        const R = lanes * dr.block;
        const nb = dr.block - 1;
        const M = lanes * nb;
        const D = dr.d;
        const hd = dr.hd;
        const vl = w.cfg.vocab / wt.world;
        const cand_n = M * 2 * dr.top_k;
        var b: Batch = undefined;
        b.gpa = gpa;
        b.dr = dr;
        b.w = w;
        b.lanes = lanes;
        b.pool = null;
        const Rb = @max(R, tmax); // the buffers a context update of up to tmax stacked rows shares with the block pass
        const sizes = [_]usize{
            R * 4, // ids
            Rb * D * bf, // x
            Rb * D * bf, // normed
            Rb * (D / 64) * f4, // xs
            R * (4 * D / dr.gs) * bf, // dyn
            R * D * bf, // cv
            R * (D / 64) * f4, // cv_xs
            Rb * (dr.heads + 2 * dr.kvh) * hd * bf, // qkv
            dr.heads * R * hd * bf, // qq
            dr.kvh * Rb * hd * bf, // kk
            dr.kvh * Rb * hd * bf, // vv
            Rb * (hd / 2) * f4, // cos
            Rb * (hd / 2) * f4, // sin
            R * dr.heads * hd * bf, // att
            R * (dr.heads * hd / 64) * f4, // att_xs
            R * D * f4, // part
            wt.world * R * D * f4, // gath
            R * D * bf, // rowb
            R * 2 * dr.inter * bf, // gu
            R * dr.inter * bf, // act
            R * (dr.inter / 64) * f4, // act_xs
            M * D * bf, // hh
            M * (D / 64) * f4, // hh_xs
            M * vl * bf, // logits
            cand_n * 4, // cand_loc
            wt.world * cand_n * 4, // cand_all
            M * dr.sel_rank * f4, // proj
            tmax * 5 * D * bf, // tap_in
            tmax * (5 * D / 64) * f4, // fc_xs
            2 * tmax * 8, // tpos
        };
        var at: [sizes.len]usize = undefined;
        var off: usize = 0;
        for (sizes, 0..) |s, i| {
            off = std.mem.alignForward(usize, off, 256);
            at[i] = off;
            off += s;
        }
        b.mem = try cuda.DeviceBuffer.alloc(d, off + 256);
        errdefer b.mem.free();
        try b.mem.fill8(0, null);
        inline for (.{ "ids", "x", "normed", "xs", "dyn", "cv", "cv_xs", "qkv", "qq", "kk", "vv", "cos", "sin", "att", "att_xs", "part", "gath", "rowb", "gu", "act", "act_xs", "hh", "hh_xs", "logits", "cand_loc", "cand_all", "proj", "tap_in", "fc_xs", "tpos" }, 0..) |name, i| {
            @field(b, name) = b.mem.ptr + at[i];
        }
        b.cand_host = try gpa.alloc(u32, wt.world * cand_n);
        errdefer gpa.free(b.cand_host);
        b.proj_host = try gpa.alloc(f32, M * dr.sel_rank);
        return b;
    }

    pub fn deinit(b: *Batch) void {
        b.gpa.free(b.cand_host);
        b.gpa.free(b.proj_host);
        if (b.pool) |*pl| pl.free();
        b.mem.free();
    }

    /// v1.10 MultiDrafter's pool: every lane's ring caches in one buffer; notes/glm_dflash.md#batch-pool
    pub fn adopt(b: *Batch, e: *Engine, ds: []const *Drafter) !void {
        if (ds.len == 0 or ds.len > b.lanes or ds[0].ring == 0) return;
        b.lane_bytes = ds[0].cacheBytes();
        b.cap0 = ds[0].cap;
        var pl = try cuda.DeviceBuffer.alloc(e.d, ds.len * b.lane_bytes + 2 * 256 + 256);
        errdefer pl.free();
        b.meta = std.mem.alignForward(u64, pl.ptr + ds.len * b.lane_bytes, 256);
        b.slots = b.meta + 256;
        for (ds, 0..) |dft, s| try dft.adopt(e, pl.ptr + s * b.lane_bytes);
        try e.s.synchronize();
        b.pool = pl;
    }

    /// A 4-bit projection of `rows` rows: 17 to 32 on the 32-row tile, else pieces of 16; notes/glm_dflash.md#batch-mm
    fn mm(e: *Engine, x: u64, xs: u64, q: kern.Q4, out: u64, rows: usize, fp32: bool) !void {
        if (rows > piece and rows <= 2 * piece) return kern.qmmShared(e.k, e.s, x, q.k, xs, q, out, rows, fp32);
        var r: usize = 0;
        while (r < rows) : (r += piece) {
            const n = @min(piece, rows - r);
            try kern.qmmDecode(e.k, e.s, x + r * q.k * bf, q.k, xs + r * (q.k / 64) * f4, q, out + r * q.n * @as(usize, if (fp32) f4 else bf), n, fp32);
        }
    }

    fn norm(b: *const Batch, e: *Engine, x: u64, weight: u64, out: u64, xs: u64, rows: usize) !void {
        try e.t.rmsnorm(x, b.dr.d, weight, out, b.dr.d, xs, rows, b.dr.d, b.dr.eps);
    }

    /// Drafter.row over every row: the fp32 partials gathered once and added in rank order into rowb.
    fn row(b: *const Batch, e: *Engine, x: u64, xs: u64, q: kern.Q4, rows: usize) !void {
        try mm(e, x, xs, q, b.part, rows, true);
        const n = rows * b.dr.d;
        try e.comm.allGather(e.s, b.part, b.gath, n);
        var a: cuda.Args = .{};
        a.add(b.gath);
        a.add(int(wt.world));
        a.add(int(n));
        a.add(b.rowb);
        try cuda.launch.launch(e.k.df_rowsum, .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, e.s, &a);
    }

    /// `_dconv` a block at a time: block s's rows of x, dyn, the residual and out.
    fn dconv(b: *const Batch, e: *Engine, x: u64, base: u64, res: ?u64, out: u64, S: usize, branch: usize) !void {
        const dr = b.dr;
        const n = dr.block;
        const D = dr.d;
        for (0..S) |s| {
            const o = s * n * D * bf;
            try e.t.dconv(x + o, b.dyn + s * n * (4 * D / dr.gs) * bf, base, if (res) |r| r + o else null, out + o, n, D, dr.gs, branch);
        }
    }

    fn layer(b: *const Batch, e: *Engine, ds: []const *Drafter, i: usize) !void {
        const dr = b.dr;
        const L = dr.layers[i];
        const D = dr.d;
        const n = dr.block;
        const hd = dr.hd;
        const S = ds.len;
        const R = S * n;
        const qkv_w = (dr.heads + 2 * dr.kvh) * hd;
        try b.norm(e, b.x, L.in_norm, b.normed, b.xs, R);
        try mm(e, b.normed, b.xs, L.a_kp, b.dyn, R, false);
        try b.dconv(e, b.normed, L.a_base, null, b.cv, S, 0);
        try e.t.groupSums(b.cv, D, b.cv_xs, R, D);
        try mm(e, b.cv, b.cv_xs, L.qkv, b.qkv, R, false);
        if (b.pool) |pl| {
            // every block at once: q, k, v [heads, R, hd]; block s's rows into its slot at its position, then attention
            const half = b.lane_bytes / (2 * dr.layers.len);
            const kc = pl.ptr + 2 * i * half;
            const d0 = ds[0];
            try e.t.draftPrep(b.qkv, L.q_norm, L.k_norm, b.cos, b.sin, b.qq, b.kk, b.vv, R, qkv_w, dr.eps, dr.heads, dr.kvh, hd);
            for ([_][2]u64{ .{ b.kk, kc }, .{ b.vv, kc + half } }) |sc| {
                var a: cuda.Args = .{};
                a.add(sc[0]);
                a.add(sc[1]);
                a.add(b.meta);
                a.add(b.slots);
                a.add(int(R));
                a.add(int(n));
                a.add(int(d0.cap));
                a.add(int(d0.ring));
                a.add(int(hd));
                try cuda.launch.launch(e.k.df_scatter_seg, .{ .grid = .{ .x = @intCast(R), .y = @intCast(dr.kvh) }, .block = .{ .x = 128 } }, e.s, &a);
            }
            try e.t.draftAttentionSeg(b.qq, kc, kc + half, b.att, b.meta, b.slots, dr.window, R, n, d0.cap, dr.heads, dr.kvh, hd, d0.cap, dr.causal, true);
        } else for (ds, 0..) |dft, s| {
            const qq = b.qq + s * dr.heads * n * hd * bf;
            const kk = b.kk + s * dr.kvh * n * hd * bf;
            const vv = b.vv + s * dr.kvh * n * hd * bf;
            const rope = s * n * (hd / 2) * f4;
            try e.t.draftPrep(b.qkv + s * n * qkv_w * bf, L.q_norm, L.k_norm, b.cos + rope, b.sin + rope, qq, kk, vv, n, qkv_w, dr.eps, dr.heads, dr.kvh, hd);
            try dft.scatter(e, kk, dft.kc[i], n);
            try dft.scatter(e, vv, dft.vc[i], n);
            try e.t.draftAttention(qq, dft.kc[i], dft.vc[i], b.att + s * n * dr.heads * hd * bf, dft.pos_dev, dr.window, n, dr.heads, dr.kvh, hd, dft.cap, dr.causal, dft.ring != 0);
        }
        try e.t.groupSums(b.att, dr.heads * hd, b.att_xs, R, dr.heads * hd);
        try b.row(e, b.att, b.att_xs, L.o, R);
        try b.dconv(e, b.rowb, L.a_base, b.x, b.x, S, 1);
        try b.norm(e, b.x, L.post_norm, b.normed, b.xs, R);
        try mm(e, b.normed, b.xs, L.m_kp, b.dyn, R, false);
        try b.dconv(e, b.normed, L.m_base, null, b.cv, S, 0);
        try e.t.groupSums(b.cv, D, b.cv_xs, R, D);
        try mm(e, b.cv, b.cv_xs, L.gu, b.gu, R, false);
        try e.t.swiglu(b.gu, b.act, b.act_xs, R, dr.inter, 1e30);
        try b.row(e, b.act, b.act_xs, L.down, R);
        try b.dconv(e, b.rowb, L.m_base, b.x, b.x, S, 1);
    }

    /// Drafter.block for every drafter of `ds` (`pend`: pending tokens) in one pass; notes/glm_dflash.md#batch-blocks
    pub fn blocks(b: *Batch, e: *Engine, ds: []const *Drafter, pend: []const u32, read: bool) !void {
        const dr = b.dr;
        const S = ds.len;
        if (S == 0 or S > b.lanes or pend.len != S) return error.BadDraftBatch;
        const n = dr.block;
        const R = S * n;
        var ids: [8 * max_block]i32 = undefined;
        for (0..S) |s| {
            @memset(ids[s * n ..][0..n], @intCast(dr.mask_id));
            ids[s * n] = @intCast(pend[s]);
        }
        try e.upload(b.ids, std.mem.sliceAsBytes(ids[0..R]));
        if (b.pool) |pl| {
            var slot: [8]i64 = undefined;
            const per = ds[0].cap * dr.hd * bf;
            for (ds, 0..) |dft, s| {
                slot[s] = @intCast((dft.cache - pl.ptr) / per);
                try e.copyDev(b.meta + 8 * s, dft.pos_dev, 8);
            }
            try e.upload(b.slots, std.mem.sliceAsBytes(slot[0..S]));
        }
        // with the pool every address below is fixed for a lane count (the positions come from `meta`): one graph each
        if (b.pool != null and e.use_graphs) {
            try e.replay(.{ .lane = 0, .head = .dbatch, .rows = @intCast(S), .cur = 0, .np = 0 }, pass, .{ b, e, ds });
        } else try b.pass(e, ds);
        if (!read) return;
        try b.readBack(e, ds);
    }

    /// Batch.blocks' device work, from the embedded block rows to the gathered candidates and the selector rows.
    fn pass(b: *Batch, e: *Engine, ds: []const *Drafter) !void {
        const dr = b.dr;
        const S = ds.len;
        const n = dr.block;
        const nb = n - 1;
        const D = dr.d;
        const k = dr.top_k;
        const R = S * n;
        const M = S * nb;
        try e.t.embed(b.ids, b.w.embed, b.x, R, D, 1);
        for (ds, 0..) |dft, s| {
            var a: cuda.Args = .{};
            a.add(if (b.pool != null) b.meta + 8 * s else dft.pos_dev);
            a.add(dr.inv_freq);
            a.add(int(n));
            a.add(int(dr.hd / 2));
            a.add(b.cos + s * n * (dr.hd / 2) * f4);
            a.add(b.sin + s * n * (dr.hd / 2) * f4);
            try cuda.launch.launch(e.k.df_rope, .{ .grid = .{ .x = @intCast((n * dr.hd / 2 + 127) / 128) }, .block = .{ .x = 128 } }, e.s, &a);
        }
        for (0..dr.layers.len) |i| try b.layer(e, ds, i);
        for (0..S) |s| try b.norm(e, b.x + (s * n + 1) * D * bf, dr.norm, b.hh + s * nb * D * bf, b.hh_xs + s * nb * (D / 64) * f4, nb);
        const vl = b.w.head.n;
        if (M > piece and M <= 2 * piece) {
            // one launch on the 32-row tile reads the head once (v1.10's qmm bucket); notes/glm_dflash.md#batch-mm
            try kern.qmmShared(e.k, e.s, b.hh, D, b.hh_xs, b.w.draft_head, b.logits, M, false);
        } else {
            var r: usize = 0;
            while (r < M) : (r += piece) {
                const m = @min(piece, M - r);
                try kern.q4Matmul(e.k, e.s, b.hh + r * D * bf, b.hh_xs + r * (D / 64) * f4, b.w.draft_head, b.logits + r * vl * bf, m);
            }
        }
        var a: cuda.Args = .{};
        a.add(b.logits);
        a.add(int(vl));
        a.add(int(vl));
        a.add(int(k));
        a.add(int(e.w.rank * vl));
        a.add(b.cand_loc);
        try cuda.launch.launch(e.k.topk, .{ .grid = .{ .x = @intCast(M) }, .block = .{ .x = 1024 } }, e.s, &a);
        try e.comm.allGather(e.s, b.cand_loc, b.cand_all, M * 2 * k);
        var p: cuda.Args = .{};
        p.add(b.hh);
        p.add(dr.hproj);
        p.add(b.proj);
        p.add(int(dr.sel_rank));
        p.add(int(D));
        try cuda.launch.launch(e.k.df_proj, .{ .grid = .{ .x = @intCast((dr.sel_rank + 7) / 8), .y = @intCast(M) }, .block = .{ .x = 256 } }, e.s, &p);
    }

    /// Rank 0: each drafter's merged candidates and selector rows from the last pass.
    fn readBack(b: *Batch, e: *Engine, ds: []const *Drafter) !void {
        const dr = b.dr;
        const nb = dr.block - 1;
        const k = dr.top_k;
        const M = ds.len * nb;
        // v1.10 patch 0092 as Drafter.block: a bad id is a copy read early; copy again once the device is idle
        for (0..2) |attempt| {
            try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(b.cand_host.ptr, b.cand_all, wt.world * M * 2 * k * 4, e.s.handle), "cuMemcpyDtoHAsync");
            try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(b.proj_host.ptr, b.proj, M * dr.sel_rank * f4, e.s.handle), "cuMemcpyDtoHAsync");
            try e.s.synchronize();
            var good = true;
            for (ds, 0..) |dft, s| {
                // drafter s's rows in its own block's layout: [rank][row][2k], then its selector rows
                for (0..wt.world) |rank| for (0..nb) |row_i| {
                    @memcpy(dft.cand_host[(rank * nb + row_i) * 2 * k ..][0 .. 2 * k], b.cand_host[(rank * M + s * nb + row_i) * 2 * k ..][0 .. 2 * k]);
                };
                @memcpy(dft.proj_host[0 .. nb * dr.sel_rank], b.proj_host[s * nb * dr.sel_rank ..][0 .. nb * dr.sel_rank]);
                if (!dft.merge(nb)) good = false;
            }
            if (good) return;
            if (attempt == 1) return error.BadDrafterCandidates;
            std.log.warn("drafter: a candidate id outside the vocabulary ({d} lanes); copying the rows again once the device is idle", .{ds.len});
            try e.d.check(e.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
        }
    }
};
