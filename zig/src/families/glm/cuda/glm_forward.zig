//! GLM-5.3-Flash's tensor-parallel forward, MTP step and commit on CUDA, launch for launch as the Python engine runs
//! them (v0.6.6 families/glm5_next/cuda/forward.py and mtp.py, TF_GLM_LATENT=1, the EXL3 checkpoint), eagerly:
//! every buffer is static, positions live on the device, and the host knows them too (Python's host_pos path).
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const tri = @import("glm_triton.zig");
const kern = @import("glm_kernels.zig");

const Weights = wt.Weights;
const B16 = wt.B16;

const bf = 2;
const f4 = 4;

/// Two ranks' all-gather of fp32 partials (comm.py over NCCL): rank r's `count` floats land at recv + r * count.
pub const Comm = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,

    pub fn allGather(c: Comm, s: cuda.Stream, send: u64, recv: u64, count: usize) !void {
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, .f32, c.comm, s.handle), "ncclAllGather");
    }
};

/// A bump allocator over one device allocation, 256-byte aligned pieces.
const Arena = struct {
    base: u64,
    used: usize = 0,
    size: usize,

    fn take(a: *Arena, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + bytes;
        std.debug.assert(a.used <= a.size or a.size == 0);
        return a.base + at;
    }
};

/// forward.Buffers: scratch for windows of up to `rows` rows; `prefill` for prompt chunks.
pub const Buffers = struct {
    rows: usize,
    prefill: bool,
    zero_first: bool = false,
    ids: u64,
    hin: u64,
    x: u64,
    normed: u64,
    xs: u64,
    post: u64,
    comb: u64,
    hcpart: u64,
    ka: u64,
    kg: u64,
    xs_fa: u64,
    xs_ga: u64,
    kxs: u64,
    kproj: u64, // prefill: [rows, width]
    kscr: KdaScratch, // prefill: one layer's chain outputs
    q_tmp: u64, // the wide chain's scratch
    y_tmp: u64,
    dp: u64,
    qr: u64,
    xs_qr: u64,
    lat: u64,
    xs_lat: u64,
    q: u64,
    vn: u64,
    xs_ao: u64,
    ikr: u64,
    igr: u64,
    qi: u64,
    gu: u64,
    act: u64,
    xs_act: u64,
    mlog: u64,
    rpart: u64, // the router's K-slice partials
    pick: u64,
    wts: u64,
    exl3: kern.Exl3Scratch,
    sgu: u64,
    sact: u64,
    sxs: u64,
    sy: u64,
    part: u64,
    gath: u64,
    sk: u64,
    hidden: u64,
    fnormed: u64,
    fxs: u64,
    logits: u64,
    me: u64,
    mcat: u64,
    mxs: u64,
    mx: u64,
    lat_s: tri.Tri.Scratch,
    qa: u64,
    ol: u64,
    tokens: u64,
    counts: u64,
    scores: u64,
    pools: u64,
    spo: u64,
    spm: u64,
    spl: u64,
    pos_at: u64, // block positions, 16 bytes apart (TORCH-GLUE `pos_dev + r0`)
    greedy: u64, // [rows, 2] u32: this rank's (value, token)
    greedy_all: u64, // [2, rows, 2] u32 gathered
    cand: u64, // [rows, 2 * max_k] u32: this rank's sampled candidates
    cand_all: u64, // [2, rows, 2 * max_k] gathered

    pub const pos_slots = 16;
};

pub const KdaScratch = struct { out: u64, k: u64, v: u64, g: u64, b: u64 };

/// A captured graph's key: which head, window rows, the KDA state half it reads (main only), pool bucket (0: dense).
const GraphKey = struct {
    head: Head,
    rows: u8,
    cur: u8,
    np: u32,

    const Head = enum(u8) { main, mtp };
};
const GraphEntry = struct { key: GraphKey, exec: cuda.graph.Exec };

/// Sizes every buffer for `rows`, then carves them out of `a` (a.size 0: count only).
fn carve(a: *Arena, w: *const Weights, rows: usize, capacity: usize, prefill: bool, np_max: usize) Buffers {
    const c = w.cfg;
    const d: usize = c.hidden;
    const s: usize = c.hc;
    const hl: usize = c.mla_heads / wt.world;
    const ll: usize = c.kda_heads / wt.world;
    const qk: usize = c.nope;
    const lw: usize = c.kv_lora;
    const slots: usize = c.topk + 1;
    const e_n: usize = c.experts;
    const ml: usize = c.moe_inter / wt.world;
    const sl = ml;
    const dl: usize = c.dense_inter / wt.world;
    const width = 3 * ll * 128 + 2 * 128 + ll;
    const head_rows: usize = if (prefill) 1 else rows;
    const part_rows = if (prefill) @min(rows, tri.prompt_att_rows) else rows;
    const nch = tri.chunksFor(@min(capacity, 2560) + rows);
    const snch = tri.chunksFor(tri.select_width);
    const srows = @min(rows, tri.select_rows);
    const bs = rows * slots;
    const maxu = @min(bs, e_n);
    var b: Buffers = undefined;
    b.rows = rows;
    b.prefill = prefill;
    b.zero_first = false;
    b.ids = a.take(rows * 4);
    b.hin = a.take(rows * d * bf);
    b.x = a.take(rows * s * d * bf);
    b.normed = a.take(rows * d * bf);
    b.xs = a.take(rows * d / 64 * f4);
    b.post = a.take(rows * s * f4);
    b.comb = a.take(rows * s * s * f4);
    b.hcpart = a.take(rows * tri.hc_blocks * 32 * f4);
    b.ka = a.take(rows * ll * 128 * bf);
    b.kg = a.take(rows * ll * 128 * bf);
    b.xs_fa = a.take(rows * 2 * f4);
    b.xs_ga = a.take(rows * 2 * f4);
    b.kxs = a.take(rows * ll * 128 / 64 * f4);
    b.kproj = if (prefill) a.take(rows * width * bf) else 0;
    b.kscr = if (prefill) .{ .out = a.take(rows * ll * 128 * bf), .k = a.take(rows * ll * 128 * f4), .v = a.take(rows * ll * 128 * bf), .g = a.take(rows * ll * 128 * f4), .b = a.take(rows * ll * f4) } else .{ .out = 0, .k = 0, .v = 0, .g = 0, .b = 0 };
    const wide = rows >= kern.wide_rows;
    b.q_tmp = if (wide) a.take(rows * ll * 128 * f4) else 0;
    b.y_tmp = if (wide) a.take(rows * ll * 128 * bf) else 0;
    b.dp = a.take(rows * (c.q_lora + c.kv_lora) * bf);
    b.qr = a.take(rows * c.q_lora * bf);
    b.xs_qr = a.take(rows * c.q_lora / 64 * f4);
    b.lat = a.take(rows * lw * bf);
    b.xs_lat = a.take(rows * lw / 64 * f4);
    b.q = a.take(rows * hl * qk * bf);
    b.vn = a.take(rows * hl * c.v_dim * bf);
    b.xs_ao = a.take(rows * hl * c.v_dim / 64 * f4);
    b.ikr = a.take(rows * (c.i_dim + c.i_heads) * bf);
    b.igr = a.take(rows * c.i_dim * f4);
    b.qi = a.take(rows * c.i_heads * c.i_dim * bf);
    b.gu = a.take(rows * 2 * dl * bf);
    b.act = a.take(rows * dl * bf);
    b.xs_act = a.take(rows * dl / 64 * f4);
    b.mlog = a.take(rows * e_n * f4);
    b.rpart = a.take(tri.router_ks * rows * e_n * f4);
    b.pick = a.take(rows * slots * 4);
    b.wts = a.take(rows * slots * f4);
    b.exl3 = .{
        .rows = rows,
        .slots = slots,
        .xg = a.take(bs * d * 2),
        .xu = a.take(bs * d * 2),
        .xd = a.take(bs * ml * 2),
        .z = a.take(@max(2 * kern.Exl3Scratch.gu_sk * ml, kern.Exl3Scratch.d_sk * d) * bs * f4),
        .y = a.take(bs * d * f4),
        .ids = a.take(maxu * 4),
        .count = a.take(4),
        .members = a.take(maxu * rows * 4),
    };
    b.sgu = a.take(rows * 2 * sl * bf);
    b.sact = a.take(rows * sl * bf);
    b.sxs = a.take(rows * sl / 64 * f4);
    b.sy = a.take(rows * d * f4);
    b.part = a.take(rows * d * f4);
    b.gath = a.take(wt.world * rows * d * f4);
    b.sk = a.take(8 * rows * 16384 * f4);
    b.hidden = a.take(rows * d * bf);
    b.fnormed = a.take(rows * d * bf);
    b.fxs = a.take(rows * d / 64 * f4);
    b.logits = a.take(head_rows * w.head.n * bf);
    b.me = a.take(rows * d * bf);
    b.mcat = a.take(rows * 2 * d * bf);
    b.mxs = a.take(rows * 2 * d / 64 * f4);
    b.mx = a.take(rows * d * bf);
    b.lat_s = .{ .po = a.take(nch * part_rows * hl * lw * f4), .pm = a.take(nch * part_rows * hl * f4), .pl = a.take(nch * part_rows * hl * f4), .dummy = a.take(4), .part_rows = part_rows };
    b.qa = a.take(rows * hl * lw * bf);
    b.ol = a.take(rows * hl * lw * bf);
    b.tokens = a.take(rows * tri.select_width * 4);
    b.counts = a.take(rows * 4);
    b.scores = a.take(srows * np_max * f4);
    b.pools = a.take(rows * tri.topk_pools * 8);
    b.spo = a.take(snch * rows * hl * lw * f4);
    b.spm = a.take(snch * rows * hl * f4);
    b.spl = a.take(snch * rows * hl * f4);
    b.pos_at = a.take(Buffers.pos_slots * 16);
    b.greedy = a.take(rows * 8);
    b.greedy_all = a.take(wt.world * rows * 8);
    const crows: usize = @min(rows, Engine.max_rows);
    b.cand = a.take(crows * 2 * Engine.max_k * 4);
    b.cand_all = a.take(wt.world * crows * 2 * Engine.max_k * 4);
    return b;
}

/// forward.State: one sequence's committed caches, and the MTP head's attention layer's.
pub const State = struct {
    capacity: usize,
    pos: usize = 0,
    mtp_len: usize = 0,
    mtp_drafted: usize = 0,
    pos_dev: u64,
    mtp_pos_dev: u64,
    kda_layers: usize,
    dsa_layers: usize,
    kda_index: [64]u8 = @splat(0),
    dsa_index: [64]u8 = @splat(0),
    conv: u64, // [n, 3, 3 LL 128] bf16
    rec: u64, // [2, n, LL, 128, 128] f32
    cur: usize = 0, // which half of rec is committed (every layer flips together)
    proj: u64, // [n, rows, width] bf16
    scr: KdaScratch, // [n, rows, ...]: each layer's window outputs for a replay
    scr_rows: usize,
    kc: [16]u64 = @splat(0), // [capacity, kv_lora] bf16 a DSA layer
    mtp_kc: u64,
    index: [16][3]u64 = undefined, // ik, ig [capacity, 128], pk [capacity/4 + 2, 128] bf16; the MTP layer's last
    long_context: bool,

    pub fn recBytes(st: *const State, ll: usize) usize {
        return st.kda_layers * ll * 128 * 128 * f4;
    }
};

pub const Engine = struct {
    w: *const Weights,
    s: cuda.Stream,
    d: *const cuda.Driver,
    t: tri.Tri,
    k: *const kern.Kernels,
    comm: Comm,
    buf: Buffers,
    pbuf: Buffers,
    mbuf: Buffers,
    st: State,
    mem: cuda.DeviceBuffer,
    width: usize,
    np_max: usize,
    last_logits: u64, // a prompt's last row's logits, kept past the MTP absorb that reuses pbuf
    last_hidden: u64, // and its final-normed hidden row, the first draft's input
    draft_head: kern.Q4, // weights.draft_head: the head quantized to 4 bits, read by draft steps only
    draft_mem: cuda.DeviceBuffer,
    /// Both ranks' logits rows for a draw over the whole vocabulary: [rank][row][head.n] bf16, on the device and here.
    full_dev: cuda.DeviceBuffer,
    full_host: []u16,
    /// The last `candidates` call's gathered pairs: [rank][row][k] of (fp32 value bits, token).
    cand_host: [wt.world * max_rows * 2 * max_k]u32 = undefined,
    /// Captured decode windows and MTP steps (graphs.py), found by key; TF_GLM_GRAPHS=0 runs everything eagerly.
    graphs: [max_graphs]GraphEntry = undefined,
    n_graphs: usize = 0,
    use_graphs: bool = true,
    capturing: bool = false,

    pub const max_rows = 8; // engine.MAX_ROWS
    /// Windows of 1..8 rows x 2 KDA parities and MTP steps of 1..8 rows, each dense or at one of ~10 pool buckets.
    pub const max_graphs = 320;
    /// The most candidates a rank gives a sampled row (top_k + exact_sampling.MARGIN).
    pub const max_k = 1024;
    pub const prefill_rows = 2048; // geometry.PREFILL_ROWS

    /// Buffers for windows of max_rows rows, prompt chunks of prefill_rows and the MTP head, and one sequence's state.
    pub fn init(w: *const Weights, d: *const cuda.Driver, s: cuda.Stream, t: tri.Tri, k: *const kern.Kernels, comm: Comm, capacity: usize, long_context: bool) !Engine {
        const c = w.cfg;
        const ll: usize = c.kda_heads / wt.world;
        const width = 3 * ll * 128 + 2 * 128 + ll;
        const np_max = capacity / 4;
        var n_kda: usize = 0;
        var n_dsa: usize = 0;
        var e: Engine = undefined;
        e.st = .{ .capacity = capacity, .pos_dev = 0, .mtp_pos_dev = 0, .kda_layers = 0, .dsa_layers = 0, .conv = 0, .rec = 0, .proj = 0, .scr = undefined, .scr_rows = max_rows, .mtp_kc = 0, .long_context = long_context };
        for (w.layers) |l| switch (l.kind) {
            .kda => {
                e.st.kda_index[l.index] = @intCast(n_kda);
                n_kda += 1;
            },
            .mla => {
                e.st.dsa_index[l.index] = @intCast(n_dsa);
                n_dsa += 1;
            },
        };
        e.st.kda_layers = n_kda;
        e.st.dsa_layers = n_dsa;
        // count, then allocate once and carve
        var count: Arena = .{ .base = 0, .size = 0 };
        _ = carve(&count, w, max_rows, capacity, false, np_max);
        _ = carve(&count, w, prefill_rows, capacity, true, np_max);
        _ = carve(&count, w, max_rows, capacity, false, np_max);
        const state_bytes = stateBytes(w, capacity, n_kda, n_dsa, width, long_context);
        const total = count.used + state_bytes + 4096 + (w.head.n + c.hidden) * bf + 1024;
        e.mem = try cuda.DeviceBuffer.alloc(d, total);
        errdefer e.mem.free();
        try e.mem.fill8(0, null);
        var a: Arena = .{ .base = e.mem.ptr, .size = total };
        e.buf = carve(&a, w, max_rows, capacity, false, np_max);
        e.pbuf = carve(&a, w, prefill_rows, capacity, true, np_max);
        e.mbuf = carve(&a, w, max_rows, capacity, false, np_max);
        const st = &e.st;
        st.pos_dev = a.take(16);
        st.mtp_pos_dev = a.take(16);
        st.conv = a.take(n_kda * 3 * 3 * ll * 128 * bf);
        st.rec = a.take(2 * n_kda * ll * 128 * 128 * f4);
        st.proj = a.take(n_kda * max_rows * width * bf);
        st.scr = .{ .out = a.take(n_kda * max_rows * ll * 128 * bf), .k = a.take(n_kda * max_rows * ll * 128 * f4), .v = a.take(n_kda * max_rows * ll * 128 * bf), .g = a.take(n_kda * max_rows * ll * 128 * f4), .b = a.take(n_kda * max_rows * ll * f4) };
        for (0..n_dsa) |i| st.kc[i] = a.take(capacity * c.kv_lora * bf);
        st.mtp_kc = a.take(capacity * c.kv_lora * bf);
        e.last_logits = a.take(w.head.n * bf);
        e.last_hidden = a.take(c.hidden * bf);
        if (long_context) for (0..n_dsa + 1) |i| {
            st.index[i] = .{ a.take(capacity * c.i_dim * bf), a.take(capacity * c.i_dim * bf), a.take((capacity / 4 + 2) * c.i_dim * bf) };
        };
        e.w = w;
        e.s = s;
        e.d = d;
        e.t = t;
        e.k = k;
        e.comm = comm;
        e.width = width;
        e.np_max = np_max;
        e.n_graphs = 0;
        e.capturing = false;
        e.draft_mem = try cuda.DeviceBuffer.alloc(d, kern.q4Bytes(w.head.n, w.head.k));
        errdefer e.draft_mem.free();
        {
            var tmp = try cuda.DeviceBuffer.alloc(d, w.head.n * w.head.k / 2 + 4 * w.head.n * w.head.k / 64);
            defer tmp.free();
            e.draft_head = try kern.quantize4(k, s, w.head.ptr, w.head.n, w.head.k, e.draft_mem.ptr, tmp.ptr);
            try s.synchronize();
        }
        e.full_dev = try cuda.DeviceBuffer.alloc(d, wt.world * max_rows * w.head.n * bf);
        errdefer e.full_dev.free();
        e.full_host = try std.heap.page_allocator.alloc(u16, wt.world * max_rows * w.head.n);
        e.use_graphs = if (std.c.getenv("TF_GLM_GRAPHS")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        return e;
    }

    fn stateBytes(w: *const Weights, capacity: usize, n_kda: usize, n_dsa: usize, width: usize, long_context: bool) usize {
        const c = w.cfg;
        const ll: usize = c.kda_heads / wt.world;
        var n: usize = 64 + n_kda * 3 * 3 * ll * 128 * bf + 2 * n_kda * ll * 128 * 128 * f4 + n_kda * max_rows * width * bf;
        n += n_kda * max_rows * ll * 128 * (bf + f4 + bf + f4) + n_kda * max_rows * ll * f4;
        n += (n_dsa + 1) * capacity * c.kv_lora * bf;
        if (long_context) n += (n_dsa + 1) * (2 * capacity * c.i_dim * bf + (capacity / 4 + 2) * c.i_dim * bf);
        return n + 64 * 256; // alignment slack
    }

    pub fn deinit(e: *Engine) void {
        for (e.graphs[0..e.n_graphs]) |*g| g.exec.deinit();
        e.draft_mem.free();
        e.full_dev.free();
        std.heap.page_allocator.free(e.full_host);
        e.mem.free();
    }

    // -- CUDA graphs (graphs.py) --------------------------------------------------------------------------------
    /// A window wholly below the dense limit, or wholly past it at one pool bucket, replays one graph per key; one
    /// across the limit runs eagerly (decode.py's same rule). Positions live on the device, so a graph serves every
    /// position of its key.
    fn graphKey(e: *const Engine, head: GraphKey.Head, rows: usize, pos: usize) ?GraphKey {
        if (!e.use_graphs) return null;
        const dl = denseLimit(e.w);
        var np: usize = 0;
        if (e.st.long_context and pos + rows - 1 >= dl) {
            if (pos < dl) return null;
            np = tri.poolBucket(pos, rows, e.np_max);
        }
        return .{ .head = head, .rows = @intCast(rows), .cur = if (head == .main) @intCast(e.st.cur) else 0, .np = @intCast(np) };
    }

    /// Launches key's graph, capturing `body(args)` into it the first time.
    fn replay(e: *Engine, key: GraphKey, comptime body: anytype, args: anytype) !void {
        for (e.graphs[0..e.n_graphs]) |g| if (std.meta.eql(g.key, key)) return g.exec.launchOn(e.s);
        if (e.n_graphs == max_graphs) return @call(.auto, body, args); // ponytail: past the table, eager
        try cuda.graph.beginCapture(e.s, .thread_local);
        e.capturing = true;
        @call(.auto, body, args) catch |err| {
            e.capturing = false;
            if (cuda.graph.endCapture(e.s)) |g| {
                var x = g;
                x.deinit();
            } else |_| {}
            return err;
        };
        e.capturing = false;
        var g = try cuda.graph.endCapture(e.s);
        defer g.deinit();
        const exec = try g.instantiate();
        e.graphs[e.n_graphs] = .{ .key = key, .exec = exec };
        e.n_graphs += 1;
        try exec.launchOn(e.s);
    }

    fn upload(e: *Engine, dst: u64, bytes: []const u8) !void {
        try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, e.s.handle), "cuMemcpyHtoDAsync");
    }

    pub fn copyDev(e: *Engine, dst: u64, src: u64, bytes: usize) !void {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.s.handle), "cuMemcpyDtoDAsync");
    }

    fn zero(e: *Engine, dst: u64, bytes: usize) !void {
        try e.d.check(e.d.api.cuMemsetD8Async(dst, 0, bytes, e.s.handle), "cuMemsetD8Async");
    }

    fn setInt(e: *Engine, dst: u64, v: usize) !void {
        const x: i32 = @intCast(v);
        try e.upload(dst, std.mem.asBytes(&x));
    }

    // -- State --------------------------------------------------------------------------------------------------
    pub fn reset(e: *Engine) !void {
        const c = e.w.cfg;
        const ll: usize = c.kda_heads / wt.world;
        try e.zero(e.st.conv, e.st.kda_layers * 3 * 3 * ll * 128 * bf);
        try e.zero(e.st.rec, 2 * e.st.kda_layers * ll * 128 * 128 * f4);
        e.st.cur = 0;
        try e.setPos(0);
        try e.setMtpLen(0);
        e.st.mtp_drafted = 0;
    }

    pub fn setPos(e: *Engine, pos: usize) !void {
        e.st.pos = pos;
        try e.setInt(e.st.pos_dev, pos);
    }

    pub fn setMtpLen(e: *Engine, n: usize) !void {
        e.st.mtp_len = n;
        try e.setInt(e.st.mtp_pos_dev, n);
    }

    fn checkRoom(e: *Engine, rows: usize, pos: usize) !void {
        if (pos + rows > denseLimit(e.w) and !e.st.long_context) return error.PastDenseLimit;
        if (pos + rows > e.st.capacity) return error.PastCacheCapacity;
    }

    /// forward.stage: the token ids into the static buffer.
    pub fn stage(e: *Engine, b: *Buffers, tokens: []const u32) !usize {
        if (tokens.len > b.rows) return error.WindowTooWide;
        try e.checkRoom(tokens.len, e.st.pos);
        try e.upload(b.ids, std.mem.sliceAsBytes(tokens));
        return tokens.len;
    }

    // -- blocks -------------------------------------------------------------------------------------------------
    fn gather(e: *Engine, b: *const Buffers, rows: usize) !u64 {
        try e.comm.allGather(e.s, b.part, b.gath, rows * e.w.cfg.hidden);
        return b.gath;
    }

    /// forward.mm for BF16 weights: x rows (stride in elements) times q.T into out.
    fn mm(e: *Engine, b: *const Buffers, x: u64, x_stride: usize, q: B16, out: u64, rows: usize, wide: bool) !void {
        try e.t.matmulB16(x, x_stride, q.ptr, out, b.sk, rows, q.n, q.k, wide);
    }

    fn outProj(e: *Engine, b: *const Buffers, x: u64, q: B16, rows: usize) !u64 {
        try e.mm(b, x, q.k, q, b.part, rows, true);
        return e.gather(b, rows);
    }

    fn kdaBlock(e: *Engine, l: *const wt.Layer, b: *const Buffers, rows: usize) !u64 {
        const c = e.w.cfg;
        const kw = l.kda.?;
        const li = e.st.kda_index[l.index];
        const ll: usize = c.kda_heads / wt.world;
        const st = &e.st;
        const p = if (b.prefill) b.kproj else st.proj + li * max_rows * e.width * bf;
        try e.mm(b, b.normed, c.hidden, kw.proj, p, rows, false);
        const fa = p + kw.fa_off * bf;
        const ga = p + kw.ga_off * bf;
        // B16 weights ignore the group sums (qmm.matmul), but the Python decode path still launches _group_sums
        if (!b.prefill) try e.t.groupSums(fa, e.width, b.xs_fa, rows, 128);
        try e.mm(b, fa, e.width, kw.fb, b.ka, rows, false);
        if (!b.prefill) try e.t.groupSums(ga, e.width, b.xs_ga, rows, 128);
        try e.mm(b, ga, e.width, kw.gb, b.kg, rows, false);
        const rec_layer = ll * 128 * 128 * f4;
        const rec_half = st.kda_layers * rec_layer;
        const scr: KdaScratch = if (b.prefill) b.kscr else .{ .out = st.scr.out + li * max_rows * ll * 128 * bf, .k = st.scr.k + li * max_rows * ll * 128 * f4, .v = st.scr.v + li * max_rows * ll * 128 * bf, .g = st.scr.g + li * max_rows * ll * 128 * f4, .b = st.scr.b + li * max_rows * ll * f4 };
        const conv_layer = 3 * 3 * ll * 128 * bf;
        try kern.chain(e.k, e.s, .{
            .heads = ll,
            .p = p,
            .p_stride = e.width,
            .b_off = kw.b_off,
            .a = b.ka,
            .a_stride = ll * 128,
            .g = b.kg,
            .g_stride = ll * 128,
            .conv_state = st.conv + li * conv_layer,
            .conv_w = kw.conv,
            .state_in = st.rec + st.cur * rec_half + li * rec_layer,
            .a_log = kw.a_log,
            .dt_bias = kw.dt_bias,
            .norm_w = kw.norm,
            .eps = c.eps,
            .lower = c.lower_bound,
            .rows = rows,
            .out = scr.out,
            .state_out = st.rec + (1 - st.cur) * rec_half + li * rec_layer,
            .k_save = scr.k,
            .v_save = scr.v,
            .g_save = scr.g,
            .b_save = scr.b,
            .q_tmp = b.q_tmp,
            .y_tmp = b.y_tmp,
        });
        if (b.prefill) {
            // a prompt chunk keeps every row: the layer's conv window moves now (its state half flips in prefillChunk)
            try e.t.convShift(st.conv + li * conv_layer, b.kproj, rows, 1, 3 * ll * 128, 3 * 3 * ll * 128, b.rows * e.width, e.width);
        }
        if (!b.prefill) try e.t.groupSums(scr.out, ll * 128, b.kxs, rows, ll * 128);
        return e.outProj(b, scr.out, kw.o, rows);
    }

    fn dsaBlock(e: *Engine, l: *const wt.Layer, b: *Buffers, kc: u64, pos_dev: u64, host_pos: usize, index: ?[3]u64, rows: usize) !u64 {
        const c = e.w.cfg;
        const a = l.dsa.?;
        const hl: usize = c.mla_heads / wt.world;
        const qk: usize = c.nope;
        const lw: usize = c.kv_lora;
        const dense_limit = denseLimit(e.w);
        try e.mm(b, b.normed, c.hidden, a.proj, b.dp, rows, false);
        const qkv = c.q_lora + c.kv_lora;
        try e.t.rmsnorm(b.dp, qkv, a.q_norm, b.qr, c.q_lora, b.xs_qr, rows, c.q_lora, c.eps);
        try e.t.rmsnorm(b.dp + c.q_lora * bf, qkv, a.kv_norm, b.lat, c.kv_lora, b.xs_lat, rows, c.kv_lora, c.eps);
        try e.mm(b, b.qr, c.q_lora, a.q_b, b.q, rows, false);
        // _dsa_latent
        try e.t.latentWrite(b.lat, c.kv_lora, kc, pos_dev, rows, lw);
        const all_sparse = host_pos >= dense_limit;
        const sparse_rows = index != null and (all_sparse or host_pos + rows - 1 >= dense_limit);
        if (index) |ix3| {
            const ix = a.index;
            try e.mm(b, b.normed, c.hidden, ix.kw, b.ikr, rows, false);
            try e.t.router(b.normed, c.hidden, ix.gate, b.rpart, b.igr, rows, c.hidden, c.i_dim);
            try e.t.indexUpdate(b.ikr, c.i_dim + c.i_heads, b.igr, ix.ln_w, ix.ln_b, ix.ape, ix3[0], ix3[1], ix3[2], pos_dev, rows);
        }
        try e.t.absorbQ(b.q, a.wk, b.qa, rows, hl, qk, lw);
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(qk), -0.5));
        if (!all_sparse) {
            const s = b.lat_s;
            const lat_nch = tri.chunksFor(@min(e.st.capacity, 2560) + b.rows);
            // a captured window serves every dense position: all chunks, as Python's graphs (pos unknown) run
            const nch = if (e.capturing) lat_nch else @min(tri.chunksFor(host_pos + rows), lat_nch);
            const blocks = (rows + s.part_rows - 1) / s.part_rows;
            var at: [Buffers.pos_slots]u64 = undefined;
            try e.blockPositions(b, pos_dev, host_pos, s.part_rows, blocks, &at);
            try e.t.denseAttention(b.qa, kc, at[0..blocks], s, b.ol, rows, hl, lw, nch, scale);
        }
        if (sparse_rows) {
            const ix = a.index;
            try e.mm(b, b.qr, c.q_lora, ix.qb, b.qi, rows, false);
            const np = tri.poolBucket(host_pos, rows, e.np_max);
            const bsz = @min(rows, tri.select_rows);
            const blocks = (rows + bsz - 1) / bsz;
            var at: [Buffers.pos_slots]u64 = undefined;
            try e.blockPositions(b, pos_dev, host_pos, bsz, blocks, &at);
            const ix3 = index.?;
            try e.t.selectTokens(b.qi, b.ikr + c.i_dim * bf, c.i_dim + c.i_heads, ix3[2], at[0..blocks], b.scores, b.pools, rows, np, c.i_heads, c.i_dim);
            try e.writeTokens(b, pos_dev, rows);
            try e.t.sparseAttention(b.qa, kc, b.tokens, b.counts, b.ol, b.spo, b.spm, b.spl, rows, hl, lw, tri.select_width, scale);
        }
        try e.t.expandV(b.ol, a.wv, b.vn, rows, hl, c.v_dim, lw);
        try e.t.groupSums(b.vn, hl * c.v_dim, b.xs_ao, rows, hl * c.v_dim);
        return e.outProj(b, b.vn, a.o, rows);
    }

    /// TORCH-GLUE `pos_dev + r0`: block j's first position into its own 16-byte slot (the host knows pos).
    fn blockPositions(e: *Engine, b: *const Buffers, pos_dev: u64, host_pos: usize, step: usize, blocks: usize, out: *[Buffers.pos_slots]u64) !void {
        if (blocks > Buffers.pos_slots) return error.TooManyBlocks;
        out[0] = pos_dev;
        for (1..blocks) |j| {
            out[j] = b.pos_at + j * 16;
            try e.setInt(out[j], host_pos + j * step);
        }
    }

    fn writeTokens(e: *Engine, b: *const Buffers, pos_dev: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(b.pools);
        a.add(pos_dev);
        a.add(b.tokens);
        a.add(b.counts);
        a.add(@as(i32, tri.select_width));
        a.add(@as(i32, tri.topk_pools));
        try cuda.launch.launch(e.k.tokens, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, e.s, &a);
    }

    fn mlpBlock(e: *Engine, m: wt.Mlp, b: *const Buffers, rows: usize) !u64 {
        const c = e.w.cfg;
        try e.mm(b, b.normed, c.hidden, m.gu, b.gu, rows, false);
        try e.t.swiglu(b.gu, b.act, b.xs_act, rows, m.width, c.swiglu_limit);
        return e.outProj(b, b.act, m.down, rows);
    }

    fn moeBlock(e: *Engine, m: wt.Moe, b: *const Buffers, rows: usize) !u64 {
        const c = e.w.cfg;
        const d: usize = c.hidden;
        const slots: usize = c.topk + 1;
        try e.t.router(b.normed, d, m.router, b.rpart, b.mlog, rows, d, c.experts);
        try e.t.select(b.mlog, m.bias, b.pick, b.wts, rows, c.topk, c.experts, c.routed_scale, true);
        try kern.routed(e.k, e.s, b.normed, d, b.pick, m.ex, b.exl3, rows, c.swiglu_limit);
        const sh = m.shared;
        try e.mm(b, b.normed, d, sh.gu, b.sgu, rows, false);
        try e.t.swiglu(b.sgu, b.sact, b.sxs, rows, sh.width, c.swiglu_limit);
        try e.mm(b, b.sact, sh.width, sh.down, b.sy, rows, true);
        try e.copyRows(b.sy, d * f4, b.exl3.y + c.topk * d * f4, slots * d * f4, d * f4, rows);
        try e.t.combine(b.exl3.y, b.wts, b.part, rows, d, slots);
        return e.gather(b, rows);
    }

    fn copyRows(e: *Engine, src: u64, src_ld: usize, dst: u64, dst_ld: usize, bytes: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(u64, src_ld));
        a.add(dst);
        a.add(@as(u64, dst_ld));
        a.add(@as(u64, bytes));
        try cuda.launch.launch(e.k.copy_rows, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, e.s, &a);
    }

    fn layerForward(e: *Engine, l: *const wt.Layer, b: *Buffers, rows: usize, host_pos: usize) !void {
        const c = e.w.cfg;
        const d: usize = c.hidden;
        const s: usize = c.hc;
        const h = l.attn_hc.?;
        try e.t.hcPre(b.x, h.fn_, h.base, h.scale, l.in_norm, b.normed, b.xs, b.post, b.comb, b.hcpart, rows, d, s, c.eps, c.hc_eps, c.sinkhorn);
        const g = switch (l.kind) {
            .kda => try e.kdaBlock(l, b, rows),
            .mla => blk: {
                const di = e.st.dsa_index[l.index];
                break :blk try e.dsaBlock(l, b, e.st.kc[di], e.st.pos_dev, host_pos, if (e.st.long_context) e.st.index[di] else null, rows);
            },
        };
        try e.t.hcPost(b.x, b.x, g, b.post, b.comb, wt.world, rows, d, s);
        const f = l.ffn_hc.?;
        try e.t.hcPre(b.x, f.fn_, f.base, f.scale, l.post_norm, b.normed, b.xs, b.post, b.comb, b.hcpart, rows, d, s, c.eps, c.hc_eps, c.sinkhorn);
        const g2 = if (l.mlp) |m| try e.mlpBlock(m, b, rows) else try e.moeBlock(l.moe.?, b, rows);
        try e.t.hcPost(b.x, b.x, g2, b.post, b.comb, wt.world, rows, d, s);
    }

    /// forward.compute with host_pos: logits of every row (decode) or of the last row (a prompt chunk).
    pub fn compute(e: *Engine, b: *Buffers, rows: usize) !void {
        const c = e.w.cfg;
        const d: usize = c.hidden;
        try e.t.embed(b.ids, e.w.embed, b.x, rows, d, c.hc);
        for (e.w.layers) |*l| try e.layerForward(l, b, rows, e.st.pos);
        try e.t.streamMean(b.x, b.hidden, rows, d, c.hc);
        try e.t.rmsnorm(b.hidden, d, e.w.norm, b.fnormed, d, b.fxs, rows, d, c.eps);
        if (b.prefill) {
            try e.mm(b, b.fnormed + (rows - 1) * d * bf, d, e.w.head, b.logits, 1, false);
        } else try e.mm(b, b.fnormed, d, e.w.head, b.logits, rows, false);
    }

    /// A decode window's forward: logits [R, V/2] in buf.logits.
    pub fn forward(e: *Engine, tokens_in: []const u32) !void {
        const rows = try e.stage(&e.buf, tokens_in);
        if (e.graphKey(.main, rows, e.st.pos)) |key| return e.replay(key, compute, .{ e, &e.buf, rows });
        try e.compute(&e.buf, rows);
    }

    /// forward.commit: keep the last decode window's first `keep` rows.
    pub fn commit(e: *Engine, rows: usize, keep: usize) !void {
        if (keep < 1 or keep > rows) return error.BadKeep;
        const c = e.w.cfg;
        const ll: usize = c.kda_heads / wt.world;
        const n = e.st.kda_layers;
        if (n > 0) {
            const half = n * ll * 128 * 128 * f4;
            const cur = e.st.cur;
            if (keep < rows) try kern.replayLayers(e.k, e.s, n, ll, e.st.rec + cur * half, e.st.scr.k, e.st.scr.v, e.st.scr.g, e.st.scr.b, max_rows, keep, e.st.rec + (1 - cur) * half);
            e.st.cur = 1 - cur;
            try e.t.convShift(e.st.conv, e.st.proj, keep, n, 3 * ll * 128, 3 * 3 * ll * 128, max_rows * e.width, e.width);
        }
        try e.setPos(e.st.pos + keep);
    }

    /// forward.commit(R, R) after a prompt chunk on pbuf: Python flips each KDA layer's state half as the layer runs
    /// (st.cur[li]); every layer flips once a chunk, so one flip for all after the chunk reads and writes the same halves.
    pub fn commitChunk(e: *Engine, rows: usize) !void {
        if (e.st.kda_layers > 0) e.st.cur = 1 - e.st.cur;
        try e.setPos(e.st.pos + rows);
    }

    // -- kept prompt states ----------------------------------------------------------------------------------
    /// A kept prompt state's bytes: every KDA layer's recurrent state (the half in use) and conv window. The
    /// attention caches (DSA, indexer, MTP) are not copied: they are positional, and the caller only restores a
    /// state while their rows below its position still hold its tokens.
    pub fn snapshotBytes(e: *const Engine) usize {
        const ll: usize = e.w.cfg.kda_heads / wt.world;
        return e.st.kda_layers * (ll * 128 * 128 * f4 + 3 * 3 * ll * 128 * bf);
    }

    /// The live state between prompt chunks into `dst` (snapshotBytes bytes).
    pub fn saveState(e: *Engine, dst: u64) !void {
        const ll: usize = e.w.cfg.kda_heads / wt.world;
        const rec = e.st.kda_layers * ll * 128 * 128 * f4;
        try e.copyDev(dst, e.st.rec + e.st.cur * rec, rec);
        try e.copyDev(dst + rec, e.st.conv, e.st.kda_layers * 3 * 3 * ll * 128 * bf);
    }

    /// The state `src` saved after `at` prompt tokens made live: the next prompt chunk starts at `at`.
    pub fn restoreState(e: *Engine, src: u64, at: usize) !void {
        const ll: usize = e.w.cfg.kda_heads / wt.world;
        const rec = e.st.kda_layers * ll * 128 * 128 * f4;
        try e.copyDev(e.st.rec + e.st.cur * rec, src, rec);
        try e.copyDev(e.st.conv, src + rec, e.st.kda_layers * 3 * 3 * ll * 128 * bf);
        try e.setPos(at);
        try e.setMtpLen(at);
        e.st.mtp_drafted = 0;
    }

    // -- MTP ----------------------------------------------------------------------------------------------------
    /// mtp_stage + mtp_compute: the head on rows (next token, hidden), cache slots mtp_len.. onward; the last row's
    /// draft-head logits in b.logits. The caller advances mtp_len.
    pub fn mtp(e: *Engine, b: *Buffers, next: []const u32, hidden: u64) !void {
        const c = e.w.cfg;
        const d: usize = c.hidden;
        if (e.w.mtp == null) return error.NoMtpHead;
        const n = next.len;
        if (n > b.rows) return error.WindowTooWide;
        b.zero_first = e.st.mtp_len == 0;
        try e.checkRoom(n, e.st.mtp_len);
        try e.upload(b.ids, std.mem.sliceAsBytes(next));
        if (hidden != b.hin) try e.copyDev(b.hin, hidden, n * d * bf);
        try e.t.embed(b.ids, e.w.embed, b.me, n, d, 1);
        if (b.zero_first) try e.zero(b.me, d * bf);
        if (!b.prefill and !b.zero_first) {
            if (e.graphKey(.mtp, n, e.st.mtp_len)) |key| return e.replay(key, mtpBody, .{ e, b, n });
        }
        try e.mtpBody(b, n);
    }

    /// The MTP step after its inputs are staged and embedded (b.me, b.hin).
    fn mtpBody(e: *Engine, b: *Buffers, n: usize) !void {
        const c = e.w.cfg;
        const d: usize = c.hidden;
        const m = e.w.mtp.?;
        try e.t.rmsnorm(b.me, d, m.enorm, b.mcat, 2 * d, null, n, d, c.eps);
        try e.t.rmsnorm(b.hin, d, m.hnorm, b.mcat + d * bf, 2 * d, null, n, d, c.eps);
        if (!b.prefill) try e.t.groupSums(b.mcat, 2 * d, b.mxs, n, 2 * d);
        try e.mm(b, b.mcat, 2 * d, m.eh, b.mx, n, false);
        const l = &m.layer;
        try e.t.rmsnorm(b.mx, d, l.in_norm, b.normed, d, b.xs, n, d, c.eps);
        const ix: ?[3]u64 = if (e.st.long_context) e.st.index[e.st.dsa_layers] else null;
        const g = try e.dsaBlock(l, b, e.st.mtp_kc, e.st.mtp_pos_dev, e.st.mtp_len, ix, n);
        try e.t.residualAdd(b.mx, b.mx, g, wt.world, n, d);
        try e.t.rmsnorm(b.mx, d, l.post_norm, b.normed, d, b.xs, n, d, c.eps);
        const g2 = try e.moeBlock(l.moe.?, b, n);
        try e.t.residualAdd(b.mx, b.mx, g2, wt.world, n, d);
        try e.t.rmsnorm(b.mx + (n - 1) * d * bf, d, m.head_norm, b.fnormed, d, b.fxs, 1, d, c.eps);
        try e.draftHead(b, b.fnormed, 1);
    }

    /// The MTP head's logits from the 4-bit draft head (mtp.py: `w.draft_head`, forward.mm): a prompt chunk's on the
    /// prefill matmul, a decode step's on qmm_group with the rows' group sums (b.fxs, from the final norm).
    fn draftHead(e: *Engine, b: *const Buffers, x: u64, rows: usize) !void {
        if (b.prefill) return kern.q4Prefill(e.k, e.s, x, e.draft_head, b.logits, rows);
        try kern.q4Matmul(e.k, e.s, x, b.fxs, e.draft_head, b.logits, rows);
    }

    // -- sampling -----------------------------------------------------------------------------------------------
    /// sample_rows' candidates for sampled rows: each rank's top `k` of its vocabulary slice (k = top_k + MARGIN),
    /// gathered on every rank into cand_host, for the host chooser.
    pub fn candidates(e: *Engine, b: *const Buffers, logits: u64, rows: usize, k: usize) !void {
        if (k == 0 or k > max_k or k > e.w.head.n or rows > max_rows) return error.BadCandidateCount;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(i32, @intCast(e.w.head.n)));
        a.add(@as(i32, @intCast(e.w.head.n)));
        a.add(@as(i32, @intCast(k)));
        a.add(@as(i32, @intCast(e.w.rank * e.w.head.n)));
        a.add(b.cand);
        try cuda.launch.launch(e.k.topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.s, &a);
        const n = rows * 2 * k;
        try e.comm.lib.check(e.comm.lib.api.ncclAllGather(b.cand, b.cand_all, n, .u32, e.comm.comm, e.s.handle), "ncclAllGather");
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(&e.cand_host, b.cand_all, wt.world * n * 4, e.s.handle), "cuMemcpyDtoHAsync");
        try e.s.synchronize();
    }

    /// nucleus_rows' gather of whole shards: every rank's `rows` logits rows into full_host, [rank][row][head.n].
    pub fn fullRows(e: *Engine, logits: u64, rows: usize) ![]const u16 {
        if (rows > max_rows) return error.WindowTooWide;
        const n = rows * e.w.head.n;
        try e.comm.lib.check(e.comm.lib.api.ncclAllGather(logits, e.full_dev.ptr, n, .bf16, e.comm.comm, e.s.handle), "ncclAllGather");
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(e.full_host.ptr, e.full_dev.ptr, wt.world * n * bf, e.s.handle), "cuMemcpyDtoHAsync");
        try e.s.synchronize();
        return e.full_host[0 .. wt.world * n];
    }

    /// sample_rows, greedy: each rank's row maximum (first column on ties), gathered; the larger value wins, the
    /// smaller token on a tie (np.lexsort((tokens, -values))).
    pub fn greedy(e: *Engine, b: *const Buffers, logits: u64, rows: usize, vocab: usize, out: []u32) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(i32, @intCast(vocab)));
        a.add(@as(i32, @intCast(e.w.head.n)));
        a.add(@as(i32, @intCast(e.w.rank * e.w.head.n)));
        a.add(b.greedy);
        try cuda.launch.launch(e.k.greedy, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.s, &a);
        try e.comm.lib.check(e.comm.lib.api.ncclAllGather(b.greedy, b.greedy_all, rows * 2, .u32, e.comm.comm, e.s.handle), "ncclAllGather");
        var host: [wt.world * max_rows * 2]u32 = undefined;
        const n = wt.world * rows * 2;
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(&host, b.greedy_all, n * 4, e.s.handle), "cuMemcpyDtoHAsync");
        try e.s.synchronize();
        for (0..rows) |r| {
            var best_v: f32 = -std.math.inf(f32);
            var best_t: u32 = std.math.maxInt(u32);
            for (0..wt.world) |k| {
                const v: f32 = @bitCast(host[(k * rows + r) * 2]);
                const t = host[(k * rows + r) * 2 + 1];
                if (v > best_v or (v == best_v and t < best_t)) {
                    best_v = v;
                    best_t = t;
                }
            }
            out[r] = best_t;
        }
    }
};

pub fn denseLimit(w: *const Weights) usize {
    return @as(usize, w.cfg.i_topk) + w.cfg.kpool - 1;
}
