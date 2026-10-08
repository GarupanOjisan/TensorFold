//! GLM-5.3-Flash's tensor-parallel forward, MTP step and commit on CUDA, launch for launch as the Python engine runs
//! them (v0.6.6 families/glm5_next/cuda/forward.py and mtp.py, TF_GLM_LATENT=1, the EXL3 checkpoint), eagerly:
//! every buffer is static, positions live on the device, and the host knows them too (Python's host_pos path).
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const tri = @import("glm_triton.zig");
const kern = @import("glm_kernels.zig");
const l2pf = @import("glm_l2pf.zig");
const roce_mod = @import("glm_roce.zig");

const Weights = wt.Weights;
const Mat = wt.Mat;

const bf = 2;
const f4 = 4;

/// Two ranks' all-gather of fp32 partials (comm.py over NCCL): rank r's `count` floats land at recv + r * count.
pub const Comm = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    roce: ?*roce_mod.Roce = null,

    /// Small gathers (up to TF_ROCE_MAX_KB a rank, a multiple of 16 bytes) over RoCE when it is up, the rest through NCCL; the
    /// choice is by byte count alone, which both ranks' matching calls share.
    pub fn allGather(c: Comm, s: cuda.Stream, send: u64, recv: u64, count: usize) !void {
        if (c.roce) |r| if (r.fits(count * 4) and send % 16 == 0 and recv % 16 == 0) return r.gather(s, send, recv, count * 4);
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, .f32, c.comm, s.handle), "ncclAllGather");
    }

    /// comm.exchange: byte ranges sent to and received from the peer in one NCCL group, always through NCCL (hcsplit's swaps).
    pub fn exchange(c: Comm, s: cuda.Stream, sends: []const Seg, recvs: []const Seg, peer: usize) !void {
        const api = c.lib.api;
        try c.lib.check(api.ncclGroupStart(), "ncclGroupStart");
        for (sends) |x| try c.lib.check(api.ncclSend(x.ptr, x.bytes, .u8, @intCast(peer), c.comm, s.handle), "ncclSend");
        for (recvs) |x| try c.lib.check(api.ncclRecv(x.ptr, x.bytes, .u8, @intCast(peer), c.comm, s.handle), "ncclRecv");
        try c.lib.check(api.ncclGroupEnd(), "ncclGroupEnd");
    }

    pub const Seg = struct { ptr: u64, bytes: usize };
};

/// TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP (v1.10 hcsplit.SplitSettings; the recipe's default SPLIT=1 sets 1 and 2, as here):
/// a prompt chunk's hyper-connection glue split by rows between the ranks (each rank sends the other its partial of the
/// other's rows and glues its own), the swaps on a second stream in row pieces (overlap), and the next block's
/// row-independent front on own rows while the swaps run (fronts). Every glue kernel is row-independent and the partials are
/// summed rank 0 first as unsplit, so every bit is the unsplit path's. Swaps are p2p only (no TF_GLM_HC_EXCHANGE=gather).
pub const SplitSettings = struct {
    split: bool = true,
    overlap: bool = true,
    fronts: bool = true,
    pieces: usize = 2, // TF_GLM_OVERLAP_PIECES
    min_rows: usize = 64, // TF_GLM_HC_SPLIT_MIN_ROWS: shorter chunks (a prompt's tail) run unsplit

    pub fn fromEnv() !SplitSettings {
        var s: SplitSettings = .{};
        if (std.c.getenv("TF_GLM_HC_SPLIT")) |v| {
            const t = std.mem.span(v);
            if (!std.mem.eql(u8, t, "0") and !std.mem.eql(u8, t, "1")) return bad("TF_GLM_HC_SPLIT is 0 or 1", t);
            s.split = t[0] == '1';
        }
        const level: u8 = if (std.c.getenv("TF_GLM_PREFILL_OVERLAP")) |v| blk: {
            const t = std.mem.span(v);
            if (t.len != 1 or t[0] < '0' or t[0] > '2') return bad("TF_GLM_PREFILL_OVERLAP is 0, 1 or 2", t);
            break :blk t[0] - '0';
        } else if (s.split) 2 else 0;
        if (level > 0 and !s.split) return bad("TF_GLM_PREFILL_OVERLAP overlaps the row-split swaps: it needs TF_GLM_HC_SPLIT=1, not", "0");
        s.overlap = level > 0;
        s.fronts = level == 2;
        if (std.c.getenv("TF_GLM_HC_EXCHANGE")) |v| if (!std.mem.eql(u8, std.mem.span(v), "p2p")) return bad("TF_GLM_HC_EXCHANGE: only p2p is ported, not", std.mem.span(v));
        if (std.c.getenv("TF_GLM_OVERLAP_PIECES")) |v| s.pieces = try ranged("TF_GLM_OVERLAP_PIECES", v, 1, 8);
        if (std.c.getenv("TF_GLM_HC_SPLIT_MIN_ROWS")) |v| s.min_rows = try ranged("TF_GLM_HC_SPLIT_MIN_ROWS", v, 2, 16384);
        return s;
    }

    pub fn pieceCount(s: SplitSettings) usize {
        return if (s.overlap) s.pieces else 1;
    }

    /// What both ranks must agree on (hcsplit.SplitSettings.code), one word.
    pub fn code(s: SplitSettings) u32 {
        if (!s.split) return 0;
        return 1 | @as(u32, @intFromBool(s.overlap)) << 1 | @as(u32, @intFromBool(s.fronts)) << 2 | @as(u32, @intCast(s.pieceCount())) << 4 | @as(u32, @intCast(s.min_rows)) << 8;
    }

    fn bad(what: []const u8, v: []const u8) error{BadSplitSetting} {
        std.log.err("{s} {s}", .{ what, v });
        return error.BadSplitSetting;
    }

    fn ranged(name: []const u8, v: [*:0]const u8, lo: usize, hi: usize) !usize {
        const t = std.mem.span(v);
        const n = std.fmt.parseInt(usize, t, 10) catch return bad(name, t);
        if (n < lo or n > hi) return bad(name, t);
        return n;
    }
};

/// One prompt buffer's row split (hcsplit.HcSplit, two ranks, p2p): rank r owns rows [r H, r H + H), H = Rp / 2 (an odd
/// chunk gets one zero pad row, never read); the second stream and its events when overlapping.
const Split = struct {
    set: SplitSettings,
    side: ?cuda.Stream,
    ev_begin: cuda.Event,
    ev_done: cuda.Event,
    ev_fill: [8]cuda.Event,
    ev_part: [8]cuda.Event,
    ev_glue: [8]cuda.Event,
    active: bool = false,
    r: usize = 0,
    rp: usize = 0,
    h: usize = 0,
    pieces: [8][2]usize = undefined, // (offset in the rank's rows, rows)
    n: usize = 0,

    fn init(d: *const cuda.Driver, set: SplitSettings) !Split {
        var sp: Split = .{ .set = set, .side = null, .ev_begin = try cuda.Event.init(d, false), .ev_done = try cuda.Event.init(d, false), .ev_fill = undefined, .ev_part = undefined, .ev_glue = undefined };
        for (0..8) |j| {
            sp.ev_fill[j] = try cuda.Event.init(d, false);
            sp.ev_part[j] = try cuda.Event.init(d, false);
            sp.ev_glue[j] = try cuda.Event.init(d, false);
        }
        // TF_GLM_OVERLAP_PRIORITY's default: the side stream at normal priority (0022)
        if (set.overlap) sp.side = try cuda.Stream.init(d, true);
        return sp;
    }

    fn deinit(sp: *Split) void {
        if (sp.side) |*s| s.deinit();
        sp.ev_begin.deinit();
        sp.ev_done.deinit();
        for (0..8) |j| {
            sp.ev_fill[j].deinit();
            sp.ev_part[j].deinit();
            sp.ev_glue[j].deinit();
        }
    }
};

/// What a split site writes into b.part for rows lo .. hi (forward.partials' `fill`): an output projection's rows, or the
/// MoE's shared expert on the rows its front has not run on, then the combine.
const Fill = union(enum) {
    proj: struct { x: u64, xs: u64, q: Mat },
    moe: struct { m: wt.Moe, done: ?[2]usize },
};

/// The next block's row-independent start (forward.mixer_front / ffn_front) the glue runs on own rows (TF_GLM_PREFILL_OVERLAP=2).
const Front = union(enum) { none, kda: *const wt.Layer, dsa: *const wt.Layer, mlp: wt.Mlp, shared: wt.Mlp };

/// forward.undone: the row ranges of [0, rows) outside `done` (rows a front already ran on), in row order; empty ranges
/// have hi <= lo.
fn undone(rows: usize, done: ?[2]usize) [2][2]usize {
    const d = done orelse return .{ .{ 0, rows }, .{ 0, 0 } };
    return .{ .{ 0, @min(d[0], rows) }, .{ d[1], rows } };
}

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
    split: u64, // sparse.select_split's int32 scratch (the rows that take it: under 64)
    spo: u64,
    spm: u64,
    spl: u64,
    taps: [5]u64, // DFlash2's taps: the stream means after five layers, [rows, D] each (0: off)
    pos_at: u64, // block positions, 16 bytes apart (TORCH-GLUE `pos_dev + r0`)
    greedy: u64, // [rows, 2] u32: this rank's (value, token)
    greedy_all: u64, // [2, rows, 2] u32 gathered
    cand: u64, // [rows, 2 * max_k] u32: this rank's sampled candidates
    cand_all: u64, // [2, rows, 2 * max_k] gathered

    pub const pos_slots = 16;
};

pub const KdaScratch = struct { out: u64, k: u64, v: u64, g: u64, b: u64 };

/// A captured graph's key: which head, window rows, the KDA state half it reads (main only), pool bucket (0: dense).
pub const GraphKey = struct {
    head: Head,
    rows: u8,
    cur: u8,
    np: u32,

    pub const Head = enum(u8) { main, mtp, dtaps, dblock }; // DFlash2's context update and block pass (glm_dflash.zig)
};
const GraphEntry = struct { key: GraphKey, exec: cuda.graph.Exec };

/// Sizes every buffer for `rows`, then carves them out of `a` (a.size 0: count only).
fn carve(a: *Arena, w: *const Weights, rows: usize, capacity: usize, prefill: bool, np_max: usize, tap_n: usize) Buffers {
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
    for (&b.taps, 0..) |*t, i| t.* = if (i < tap_n) a.take(rows * d * bf) else 0;
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
        .plan = .{
            .members = a.take(bs * 4),
            .items = a.take(kern.maxItems(bs, e_n + 1, 16) * 12),
            .counts = a.take(8),
            .rank = a.take(if (bs > 1024) bs * 4 else 4),
            .hist = a.take(if (bs > 1024) (bs + 1023) / 1024 * (e_n + 1) * 4 else 4),
            .order = a.take((bs / 16 + 1024) * 4),
        },
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
    b.split = a.take(tri.Tri.splitInts(@min(rows, tri.split_rows - 1), np_max) * 4);
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
    /// Both ranks' logits rows for a draw over the whole vocabulary: [rank][row][head.n] bf16, on the device and here.
    full_dev: cuda.DeviceBuffer,
    full_host: []u16,
    /// The last `candidates` call's gathered pairs: [rank][row][k] of (fp32 value bits, token).
    cand_host: [wt.world * max_rows * 2 * max_k]u32 = undefined,
    /// Captured decode windows and MTP steps (graphs.py), found by key; TF_GLM_GRAPHS=0 runs everything eagerly.
    graphs: [max_graphs]GraphEntry = undefined,
    n_graphs: usize = 0,
    use_graphs: bool = true,
    /// TF_GLM_WIDE_GRAPHS (v1.8 production: 16; 0: off): the one verify-window width past 8 rows that replays a CUDA graph
    /// (copy drafts pad their windows up to it, glm_copy.zig); the others of 9..15 rows run eagerly.
    wide_graph: usize = 16,
    /// DFlash2's tap layers (the target layers whose stream means `Buffers.taps` take); none when taps are off.
    tap_layers: [5]u32 = @splat(std.math.maxInt(u32)),
    /// TF_GLM_EXL3_DEC (default on): v1.8's routed-expert kernels, plan and direct shared-expert combine; 0: the generic
    /// grouped kernel of v0.6.6 (the A/B path).
    exl3_v18: bool = true,
    /// TF_GLM_KV (default fp8, v1.8's production setting; bf16 the earlier format): the DSA latent cache and the indexer's
    /// pooled keys as e4m3 rows with a power-of-two scale each (kv8.py).
    kv_fp8: bool = true,
    /// TF_GLM_HC_DEC / TF_GLM_HC_MMA (default on): hyper-connection mixing dots of decode rows in hc.cu, of prompt rows on the
    /// tensor cores.
    hc_dec: bool = true,
    /// TF_GLM_L2PF (default on): decode windows of up to TF_GLM_L2PF_ROWS rows prefetch the next kernels' weights into L2 from
    /// a side stream (glm_l2pf.zig); `l2pf_active` is set while such a window runs.
    l2pf: ?l2pf.Prefetch = null,
    l2pf_active: bool = false,
    hc_mma: bool = true,
    /// TF_GLM_LATENT_MMA / TF_GLM_SPARSE_ONEPASS (default on): a prompt chunk's absorb and expand on the tensor cores, its
    /// sparse attention in one pass a row.
    latent_mma: bool = true,
    sparse_onepass: bool = true,
    /// The indexer's key rings hold the whole context here, so no row wraps; the ring length only picks the kernels'
    /// 16-divisibility specialization, as Python's does.
    index_ring: usize = 0,
    /// TF_GLM_KDA_CHUNKED (default on, v1.8's production setting): prompt chunks run KDA in chunked (WY) form, which
    /// needs prompt chunks to start on a grid of 32 (`kdaChunked`, the server's `promptGrid`); 0: the serial chain.
    kda_chunked: bool = true,
    /// TF_GLM_COMBINE_SY (default on, with exl3_v18): decode windows combine the shared expert's row where it is.
    combine_sy: bool = true,
    capturing: bool = false,
    /// The nucleus draw's device scratch (`nucleusGather`), allocated at its first use, and its host copy.
    nuc_dev: ?cuda.DeviceBuffer = null,
    nuc_host: []u64 = &.{},
    /// TF_GLM_HC_SPLIT (default on): prompt chunks' glue split by rows between the ranks (`SplitSettings`); null when off.
    split: ?Split = null,

    pub const max_rows = 16; // engine.MAX_ROWS: a pending token and up to 15 copied drafts
    pub const mtp_rows = 8; // engine.MTP_ROWS: rows an MTP step takes at once (a kept backlog is absorbed in steps of this)
    /// Windows of 1..8 and `wide_graph` rows x 2 KDA parities and MTP steps of 1..8 rows, each dense or at one of ~10 pool
    /// buckets (9 widths x 2 x 11 + 8 x 11 = 286).
    pub const max_graphs = 352;
    /// The most candidates a rank gives a sampled row (top_k + exact_sampling.MARGIN).
    pub const max_k = 1024;
    pub const prefill_rows = 2048; // geometry.PREFILL_ROWS

    /// Buffers for windows of max_rows rows, prompt chunks of prefill_rows and the MTP head, and one sequence's state.
    pub fn init(w: *const Weights, d: *const cuda.Driver, s: cuda.Stream, t: tri.Tri, k: *const kern.Kernels, comm: Comm, capacity: usize, long_context: bool, taps: ?[5]u32) !Engine {
        if (std.c.getenv("TF_GLM_KDA_WIDE_ROWS")) |v| kern.wide_rows = try std.fmt.parseInt(usize, std.mem.span(v), 10);
        if (std.c.getenv("TF_GLM_EXL3_DEC_ORDER")) |v| {
            kern.dec_order = std.fmt.parseInt(usize, std.mem.span(v), 10) catch 9;
            if (kern.dec_order > 2) {
                std.log.err("TF_GLM_EXL3_DEC_ORDER is 0, 1 or 2, not {s}", .{std.mem.span(v)});
                return error.BadDecOrder;
            }
        }
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
        const tap_n: usize = if (taps != null) 5 else 0;
        _ = carve(&count, w, max_rows, capacity, false, np_max, tap_n);
        _ = carve(&count, w, prefill_rows, capacity, true, np_max, tap_n);
        _ = carve(&count, w, mtp_rows, capacity, false, np_max, 0);
        const kv_fp8 = try kvFp8();
        const state_bytes = stateBytes(w, capacity, n_kda, n_dsa, width, long_context, kv_fp8);
        const total = count.used + state_bytes + 4096 + (w.head.n + c.hidden) * bf + 1024;
        e.mem = try cuda.DeviceBuffer.alloc(d, total);
        errdefer e.mem.free();
        try e.mem.fill8(0, null);
        var a: Arena = .{ .base = e.mem.ptr, .size = total };
        e.buf = carve(&a, w, max_rows, capacity, false, np_max, tap_n);
        e.pbuf = carve(&a, w, prefill_rows, capacity, true, np_max, tap_n);
        e.mbuf = carve(&a, w, mtp_rows, capacity, false, np_max, 0);
        e.tap_layers = taps orelse @splat(std.math.maxInt(u32));
        const st = &e.st;
        const lat_b: usize = if (kv_fp8) c.kv_lora + 16 else c.kv_lora * bf; // a latent cache row (kv8.row_bytes)
        const pk_b: usize = if (kv_fp8) c.i_dim + 16 else c.i_dim * bf; // and a pooled key row
        st.pos_dev = a.take(16);
        st.mtp_pos_dev = a.take(16);
        st.conv = a.take(n_kda * 3 * 3 * ll * 128 * bf);
        st.rec = a.take(2 * n_kda * ll * 128 * 128 * f4);
        st.proj = a.take(n_kda * max_rows * width * bf);
        st.scr = .{ .out = a.take(n_kda * max_rows * ll * 128 * bf), .k = a.take(n_kda * max_rows * ll * 128 * f4), .v = a.take(n_kda * max_rows * ll * 128 * bf), .g = a.take(n_kda * max_rows * ll * 128 * f4), .b = a.take(n_kda * max_rows * ll * f4) };
        for (0..n_dsa) |i| st.kc[i] = a.take(capacity * lat_b);
        st.mtp_kc = a.take(capacity * lat_b);
        e.last_logits = a.take(w.head.n * bf);
        e.last_hidden = a.take(c.hidden * bf);
        if (long_context) for (0..n_dsa + 1) |i| {
            st.index[i] = .{ a.take(capacity * c.i_dim * bf), a.take(capacity * c.i_dim * bf), a.take((capacity / 4 + 2) * pk_b) };
        };
        e.w = w;
        e.s = s;
        e.d = d;
        e.t = t;
        e.t.fp8 = kv_fp8;
        e.kv_fp8 = kv_fp8;
        e.k = k;
        e.comm = comm;
        e.width = width;
        e.np_max = np_max;
        e.n_graphs = 0;
        e.capturing = false;
        e.nuc_dev = null;
        e.nuc_host = &.{};
        e.l2pf_active = false; // `e` starts undefined, so every field with a default but no assignment here is garbage
        e.draft_head = w.draft_head;
        e.full_dev = try cuda.DeviceBuffer.alloc(d, wt.world * max_rows * w.head.n * bf);
        errdefer e.full_dev.free();
        e.full_host = try std.heap.page_allocator.alloc(u16, wt.world * max_rows * w.head.n);
        e.kda_chunked = kdaChunked();
        const pf_settings = try l2pf.Settings.fromEnv();
        e.l2pf = if (pf_settings.on()) try l2pf.Prefetch.build(std.heap.page_allocator, d, w, pf_settings) else null;
        e.hc_dec = envOn("TF_GLM_HC_DEC");
        e.hc_mma = envOn("TF_GLM_HC_MMA");
        e.latent_mma = envOn("TF_GLM_LATENT_MMA");
        e.sparse_onepass = envOn("TF_GLM_SPARSE_ONEPASS");
        e.index_ring = std.mem.alignForward(usize, capacity, 16);
        e.exl3_v18 = if (std.c.getenv("TF_GLM_EXL3_DEC")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        e.combine_sy = e.exl3_v18 and (if (std.c.getenv("TF_GLM_COMBINE_SY")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true);
        e.use_graphs = if (std.c.getenv("TF_GLM_GRAPHS")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        e.wide_graph = try wideGraph();
        const split_set = try SplitSettings.fromEnv();
        e.split = if (split_set.split) try Split.init(d, split_set) else null;
        errdefer if (e.split) |*sp| sp.deinit();
        if (e.split != null) {
            // hcsplit.warm (0064): both ranks open the send/receive connection now, not on the first prompt chunk
            try e.comm.exchange(s, &.{.{ .ptr = e.pbuf.part, .bytes = 16 }}, &.{.{ .ptr = e.pbuf.gath, .bytes = 16 }}, 1 - e.w.rank);
            try s.synchronize();
        }
        return e;
    }

    fn stateBytes(w: *const Weights, capacity: usize, n_kda: usize, n_dsa: usize, width: usize, long_context: bool, kv_fp8: bool) usize {
        const c = w.cfg;
        const ll: usize = c.kda_heads / wt.world;
        var n: usize = 64 + n_kda * 3 * 3 * ll * 128 * bf + 2 * n_kda * ll * 128 * 128 * f4 + n_kda * max_rows * width * bf;
        n += n_kda * max_rows * ll * 128 * (bf + f4 + bf + f4) + n_kda * max_rows * ll * f4;
        n += (n_dsa + 1) * capacity * (if (kv_fp8) c.kv_lora + 16 else c.kv_lora * bf);
        if (long_context) n += (n_dsa + 1) * (2 * capacity * c.i_dim * bf + (capacity / 4 + 2) * (if (kv_fp8) c.i_dim + 16 else c.i_dim * bf));
        return n + 64 * 256; // alignment slack
    }

    pub fn deinit(e: *Engine) void {
        for (e.graphs[0..e.n_graphs]) |*g| g.exec.deinit();
        if (e.l2pf) |*p| p.deinit();
        e.full_dev.free();
        if (e.split) |*sp| sp.deinit();
        if (e.nuc_dev) |*b| b.free();
        if (e.nuc_host.len > 0) std.heap.page_allocator.free(e.nuc_host);
        std.heap.page_allocator.free(e.full_host);
        e.mem.free();
    }

    // -- CUDA graphs (graphs.py) --------------------------------------------------------------------------------
    /// A window wholly below the dense limit, or wholly past it at one pool bucket, replays one graph per key; one
    /// across the limit runs eagerly (decode.py's same rule). Positions live on the device, so a graph serves every
    /// position of its key.
    fn graphKey(e: *const Engine, head: GraphKey.Head, rows: usize, pos: usize) ?GraphKey {
        if (!e.use_graphs) return null;
        if (rows > 8 and rows != e.wide_graph) return null; // engine.GRAPH_ROWS + WIDE_GRAPHS: the others run eagerly
        const dl = denseLimit(e.w);
        var np: usize = 0;
        if (e.st.long_context and pos + rows - 1 >= dl) {
            if (pos < dl) return null;
            np = tri.poolBucket(pos, rows, e.np_max);
        }
        return .{ .head = head, .rows = @intCast(rows), .cur = if (head == .main) @intCast(e.st.cur) else 0, .np = @intCast(np) };
    }

    /// Launches key's graph, capturing `body(args)` into it the first time.
    pub fn replay(e: *Engine, key: GraphKey, comptime body: anytype, args: anytype) !void {
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

    pub fn upload(e: *Engine, dst: u64, bytes: []const u8) !void {
        try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, e.s.handle), "cuMemcpyHtoDAsync");
    }

    pub fn copyDev(e: *Engine, dst: u64, src: u64, bytes: usize) !void {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.s.handle), "cuMemcpyDtoDAsync");
    }

    pub fn zero(e: *Engine, dst: u64, bytes: usize) !void {
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

    /// forward.mm: x rows (stride in elements) times q.T into out (fp32 with `wide`). BF16 and FP8 weights ignore the
    /// group sums `xs` (qmm.matmul); a 4-bit q takes the prompt matmul in a prompt chunk, else the decode matmul on `xs`.
    fn mm(e: *Engine, b: *const Buffers, x: u64, x_stride: usize, xs: u64, q: Mat, out: u64, rows: usize, wide: bool) !void {
        switch (q.kind) {
            .b16 => try e.t.matmulB16(x, x_stride, q.ptr, out, b.sk, rows, q.n, q.k, wide),
            .q4 => if (b.prefill) try kern.q4Prefill(e.k, e.s, x, x_stride, q.q4(), out, rows, wide) else try kern.qmmDecode(e.k, e.s, x, x_stride, xs, q.q4(), out, rows, wide),
            .f8 => try e.t.matmulF8(x, x_stride, q.ptr, q.s, out, b.sk, rows, q.n, q.k, wide),
        }
    }

    /// forward.out_proj: every rank's partials [world, R, D] (b.gath), or 0 while a prompt chunk runs row-split, which
    /// exchanges them itself (forward.partials).
    fn outProj(e: *Engine, b: *const Buffers, x: u64, xs: u64, q: Mat, rows: usize, site: ?struct { u32, usize }) !u64 {
        if (e.splitActive(b)) {
            try e.splitPartial(b, .{ .proj = .{ .x = x, .xs = xs, .q = q } });
            return 0;
        }
        try e.mm(b, x, q.k, xs, q, b.part, rows, true);
        if (site) |s| try e.pfSite(s[0], s[1]);
        return e.gather(b, rows);
    }

    /// forward.kda_front: a prompt chunk's KDA input projections of rows lo .. hi (row-independent).
    fn kdaFront(e: *Engine, l: *const wt.Layer, b: *const Buffers, lo: usize, hi: usize) !void {
        const c = e.w.cfg;
        const kw = l.kda.?;
        const ll: usize = c.kda_heads / wt.world;
        const n = hi - lo;
        const p = b.kproj + lo * e.width * bf;
        try e.mm(b, b.normed + lo * c.hidden * bf, c.hidden, b.xs + lo * c.hidden / 64 * f4, kw.proj, p, n, false);
        try e.mm(b, p + kw.fa_off * bf, e.width, b.xs_fa + lo * 2 * f4, kw.fb, b.ka + lo * ll * 128 * bf, n, false);
        try e.mm(b, p + kw.ga_off * bf, e.width, b.xs_ga + lo * 2 * f4, kw.gb, b.kg + lo * ll * 128 * bf, n, false);
    }

    /// `done`: rows whose projections kdaFront already wrote (a prompt chunk, TF_GLM_PREFILL_OVERLAP=2).
    fn kdaBlock(e: *Engine, l: *const wt.Layer, b: *const Buffers, rows: usize, host_pos: usize, done: ?[2]usize) !u64 {
        const c = e.w.cfg;
        const kw = l.kda.?;
        const li = e.st.kda_index[l.index];
        const ll: usize = c.kda_heads / wt.world;
        const st = &e.st;
        const p = if (b.prefill) b.kproj else st.proj + li * max_rows * e.width * bf;
        if (b.prefill) {
            for (undone(rows, done)) |r| if (r[1] > r[0]) try e.kdaFront(l, b, r[0], r[1]);
        } else {
            try e.mm(b, b.normed, c.hidden, b.xs, kw.proj, p, rows, false);
            const fa = p + kw.fa_off * bf;
            const ga = p + kw.ga_off * bf;
            // f_b and g_b read these group sums when 4-bit; BF16 weights ignore them (qmm.matmul), but Python still launches _group_sums
            try e.t.groupSums(fa, e.width, b.xs_fa, rows, 128);
            try e.mm(b, fa, e.width, b.xs_fa, kw.fb, b.ka, rows, false);
            try e.t.groupSums(ga, e.width, b.xs_ga, rows, 128);
            try e.mm(b, ga, e.width, b.xs_ga, kw.gb, b.kg, rows, false);
            try e.pfSite(l.index, 2);
        }
        const rec_layer = ll * 128 * 128 * f4;
        const rec_half = st.kda_layers * rec_layer;
        const scr: KdaScratch = if (b.prefill) b.kscr else .{ .out = st.scr.out + li * max_rows * ll * 128 * bf, .k = st.scr.k + li * max_rows * ll * 128 * f4, .v = st.scr.v + li * max_rows * ll * 128 * bf, .g = st.scr.g + li * max_rows * ll * 128 * f4, .b = st.scr.b + li * max_rows * ll * f4 };
        const conv_layer = 3 * 3 * ll * 128 * bf;
        const chain: kern.Chain = .{
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
        };
        // kda.chain: a prompt chunk takes the chunked form under TF_GLM_KDA_CHUNKED=1 (scratch's k/v/g/b are not written)
        if (b.prefill and e.kda_chunked) try kern.chainChunked(e.k, e.s, chain, host_pos) else try kern.chain(e.k, e.s, chain);
        if (b.prefill) {
            // a prompt chunk keeps every row: the layer's conv window moves now (its state half flips in prefillChunk)
            try e.t.convShift(st.conv + li * conv_layer, b.kproj, rows, 1, 3 * ll * 128, 3 * 3 * ll * 128, b.rows * e.width, e.width);
        }
        if (!b.prefill) try e.t.groupSums(scr.out, ll * 128, b.kxs, rows, ll * 128);
        return e.outProj(b, scr.out, b.kxs, kw.o, rows, .{ l.index, 0 });
    }

    /// forward.dsa_front: DSA's projection, query and latent norms and query expansion of rows lo .. hi (row-independent).
    fn dsaFront(e: *Engine, l: *const wt.Layer, b: *const Buffers, lo: usize, hi: usize) !void {
        const c = e.w.cfg;
        const a = l.dsa.?;
        const hl: usize = c.mla_heads / wt.world;
        const n = hi - lo;
        const qkv = c.q_lora + c.kv_lora;
        const dp = b.dp + lo * qkv * bf;
        const qr = b.qr + lo * c.q_lora * bf;
        const xs_qr = b.xs_qr + lo * c.q_lora / 64 * f4;
        try e.mm(b, b.normed + lo * c.hidden * bf, c.hidden, b.xs + lo * c.hidden / 64 * f4, a.proj, dp, n, false);
        try e.t.rmsnorm(dp, qkv, a.q_norm, qr, c.q_lora, xs_qr, n, c.q_lora, c.eps);
        try e.t.rmsnorm(dp + c.q_lora * bf, qkv, a.kv_norm, b.lat + lo * c.kv_lora * bf, c.kv_lora, b.xs_lat + lo * c.kv_lora / 64 * f4, n, c.kv_lora, c.eps);
        try e.mm(b, qr, c.q_lora, xs_qr, a.q_b, b.q + lo * hl * c.nope * bf, n, false);
    }

    /// `done`: rows dsaFront already ran on (a prompt chunk, TF_GLM_PREFILL_OVERLAP=2).
    fn dsaBlock(e: *Engine, l: *const wt.Layer, b: *Buffers, kc: u64, pos_dev: u64, host_pos: usize, index: ?[3]u64, rows: usize, done: ?[2]usize) !u64 {
        const c = e.w.cfg;
        const a = l.dsa.?;
        const hl: usize = c.mla_heads / wt.world;
        const qk: usize = c.nope;
        const lw: usize = c.kv_lora;
        const dense_limit = denseLimit(e.w);
        for (undone(rows, done)) |r| if (r[1] > r[0]) try e.dsaFront(l, b, r[0], r[1]);
        // _dsa_latent
        try e.t.latentWrite(b.lat, c.kv_lora, kc, pos_dev, rows, lw);
        const all_sparse = host_pos >= dense_limit;
        const sparse_rows = index != null and (all_sparse or host_pos + rows - 1 >= dense_limit);
        if (index) |ix3| {
            const ix = a.index;
            try e.mm(b, b.normed, c.hidden, b.xs, ix.kw, b.ikr, rows, false);
            try e.t.router(b.normed, c.hidden, ix.gate, b.rpart, b.igr, rows, c.hidden, c.i_dim);
            try e.t.indexUpdate(b.ikr, c.i_dim + c.i_heads, b.igr, ix.ln_w, ix.ln_b, ix.ape, ix3[0], ix3[1], ix3[2], pos_dev, rows, e.index_ring);
        }
        if (b.prefill and e.latent_mma) try e.t.absorbQMma(b.q, a.wk, b.qa, rows, hl, qk, lw) else try e.t.absorbQ(b.q, a.wk, b.qa, rows, hl, qk, lw);
        try e.pfSite(l.index, 2);
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(qk), -0.5));
        if (!all_sparse) {
            const s = b.lat_s;
            const lat_nch = tri.chunksFor(@min(e.st.capacity, 2560) + b.rows);
            // a captured window serves every dense position: all chunks, as Python's graphs (pos unknown) run
            const nch = if (e.capturing) lat_nch else @min(tri.chunksFor(host_pos + rows), lat_nch);
            // rows past the dense limit are recomputed sparsely below: the dense pass needs only the rows before it (a row's
            // dense bits do not depend on the others), with the window's head block (forward.dense_rows)
            const dense = if (sparse_rows and host_pos < dense_limit) @min(rows, dense_limit - host_pos) else rows;
            const blocks = (dense + s.part_rows - 1) / s.part_rows;
            var at: [Buffers.pos_slots]u64 = undefined;
            try e.blockPositions(b, pos_dev, host_pos, s.part_rows, blocks, &at);
            try e.t.denseAttention(b.qa, kc, at[0..blocks], s, b.ol, dense, rows, hl, lw, nch, scale);
        }
        if (sparse_rows) {
            const ix = a.index;
            try e.mm(b, b.qr, c.q_lora, b.xs_qr, ix.qb, b.qi, rows, false);
            const ix3 = index.?;
            if (b.prefill and rows >= tri.split_rows) { // select_tokens: a prompt chunk's rows in blocks (_select_prompt)
                const bsz = @min(rows, tri.select_rows);
                const blocks = (rows + bsz - 1) / bsz;
                var at: [Buffers.pos_slots]u64 = undefined;
                try e.blockPositions(b, pos_dev, host_pos, bsz, blocks, &at);
                try e.t.selectPrompt(b.qi, b.ikr + c.i_dim * bf, c.i_dim + c.i_heads, ix3[2], at[0..blocks], b.scores, b.pools, b.split, b.tokens, b.counts, rows, host_pos, e.np_max, c.i_heads, c.i_dim);
            } else {
                const np = tri.poolBucket(host_pos, rows, e.np_max);
                try e.t.selectDecode(b.qi, b.ikr + c.i_dim * bf, c.i_dim + c.i_heads, ix3[2], pos_dev, b.scores, b.pools, b.split, b.tokens, b.counts, rows, np, c.i_heads, c.i_dim);
            }
            if (b.prefill and e.sparse_onepass) { // prompt rows: one pass a row, no partials (their own bits)
                try e.t.sparseOnepass(b.qa, kc, b.tokens, b.counts, b.ol, rows, hl, lw, tri.select_width, scale);
            } else try e.t.sparseAttention(b.qa, kc, b.tokens, b.counts, b.ol, b.spo, b.spm, b.spl, rows, hl, lw, tri.select_width, scale);
        }
        if (b.prefill and e.latent_mma) try e.t.expandVMma(b.ol, a.wv, b.vn, rows, hl, c.v_dim, lw) else try e.t.expandV(b.ol, a.wv, b.vn, rows, hl, c.v_dim, lw);
        try e.t.groupSums(b.vn, hl * c.v_dim, b.xs_ao, rows, hl * c.v_dim);
        return e.outProj(b, b.vn, b.xs_ao, a.o, rows, .{ l.index, 0 });
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

    /// forward.mlp_front: the dense MLP's gate/up and SwiGLU of rows lo .. hi.
    fn mlpFront(e: *Engine, m: wt.Mlp, b: *const Buffers, lo: usize, hi: usize) !void {
        const c = e.w.cfg;
        const gu = b.gu + lo * 2 * m.width * bf;
        try e.mm(b, b.normed + lo * c.hidden * bf, c.hidden, b.xs + lo * c.hidden / 64 * f4, m.gu, gu, hi - lo, false);
        try e.t.swiglu(gu, b.act + lo * m.width * bf, b.xs_act + lo * m.width / 64 * f4, hi - lo, m.width, c.swiglu_limit);
    }

    fn mlpBlock(e: *Engine, m: wt.Mlp, b: *const Buffers, rows: usize, layer: u32, done: ?[2]usize) !u64 {
        for (undone(rows, done)) |r| if (r[1] > r[0]) try e.mlpFront(m, b, r[0], r[1]);
        return e.outProj(b, b.act, b.xs_act, m.down, rows, .{ layer, 1 });
    }

    /// `done`: rows whose shared expert its front already ran (a prompt chunk, TF_GLM_PREFILL_OVERLAP=2).
    fn moeBlock(e: *Engine, m: wt.Moe, b: *const Buffers, rows: usize, layer: u32, done: ?[2]usize) !u64 {
        const c = e.w.cfg;
        const d: usize = c.hidden;
        const slots: usize = c.topk + 1;
        try e.t.router(b.normed, d, m.router, b.rpart, b.mlog, rows, d, c.experts);
        try e.t.select(b.mlog, m.bias, b.pick, b.wts, rows, c.topk, c.experts, c.routed_scale, true);
        const sh = m.shared;
        // TF_GLM_L2PF: the shared expert before the routed experts (disjoint buffers, the same kernels: the same bits), while the
        // L2 still holds what site "a" prefetched for it
        if (e.l2pf_active) try e.sharedExpert(b, sh, rows);
        if (e.exl3_v18) try kern.routed18(e.k, e.s, b.normed, d, b.pick, m.ex, b.exl3, rows, c.swiglu_limit, b.prefill) else try kern.routed(e.k, e.s, b.normed, d, b.pick, m.ex, b.exl3, rows, c.swiglu_limit);
        if (e.splitActive(b)) { // the shared expert and combine a piece of rows at a time (forward.moe_block's fill)
            try e.splitPartial(b, .{ .moe = .{ .m = m, .done = done } });
            return 0;
        }
        if (!e.l2pf_active) try e.sharedExpert(b, sh, rows);
        // forward.py moe_block: the shared row read from sy where it is (a prompt chunk always, a decode window unless
        // TF_GLM_COMBINE_SY=0), else copied into the routed rows' last slot
        if (e.exl3_v18 and (b.prefill or e.combine_sy)) {
            try e.t.combineSy(b.exl3.y, b.sy, b.wts, b.part, rows, d, slots);
        } else {
            try e.copyRows(b.sy, d * f4, b.exl3.y + c.topk * d * f4, slots * d * f4, d * f4, rows);
            try e.t.combine(b.exl3.y, b.wts, b.part, rows, d, slots);
        }
        try e.pfSite(layer, 1);
        return e.gather(b, rows);
    }

    /// forward.shared_front: the MoE's shared expert on b.normed into b.sy.
    fn sharedExpert(e: *Engine, b: *const Buffers, sh: wt.Mlp, rows: usize) !void {
        return e.sharedRows(b, sh, 0, rows);
    }

    /// forward.shared_front: the shared expert on rows lo .. hi into b.sy.
    fn sharedRows(e: *Engine, b: *const Buffers, sh: wt.Mlp, lo: usize, hi: usize) !void {
        const c = e.w.cfg;
        const n = hi - lo;
        const sgu = b.sgu + lo * 2 * sh.width * bf;
        const sact = b.sact + lo * sh.width * bf;
        const sxs = b.sxs + lo * sh.width / 64 * f4;
        try e.mm(b, b.normed + lo * c.hidden * bf, c.hidden, b.xs + lo * c.hidden / 64 * f4, sh.gu, sgu, n, false);
        try e.t.swiglu(sgu, sact, sxs, n, sh.width, c.swiglu_limit);
        try e.mm(b, sact, sh.width, sxs, sh.down, b.sy + lo * c.hidden * f4, n, true);
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

    /// glue.hc_pre: the mixing dots of a prompt chunk on the tensor cores (TF_GLM_HC_MMA), of a decode window in hc.cu
    /// (TF_GLM_HC_DEC), else glue._hc_partial; then _hc_finish.
    fn hcPre(e: *Engine, b: *const Buffers, h: wt.Hc, norm: u64, rows: usize) !void {
        return e.hcPreRows(b, h, norm, 0, rows);
    }

    /// hcPre of rows lo .. lo + n (every glue kernel is row-independent).
    fn hcPreRows(e: *Engine, b: *const Buffers, h: wt.Hc, norm: u64, lo: usize, n: usize) !void {
        const c = e.w.cfg;
        const d: usize = c.hidden;
        const s: usize = c.hc;
        const x = b.x + lo * s * d * bf;
        const part = b.hcpart + lo * tri.hc_blocks * 32 * f4;
        if (b.prefill and e.hc_mma) try e.t.hcPartialMma(x, h.fn_, part, n, d, s) else if (e.hc_dec) try kern.hcPartial(e.k, e.s, x, h.fn_, part, n) else try e.t.hcPartial(x, h.fn_, part, n, d, s);
        try e.t.hcFinish(x, h.base, h.scale, norm, b.normed + lo * d * bf, b.xs + lo * d / 64 * f4, b.post + lo * s * f4, b.comb + lo * s * s * f4, part, n, d, s, c.eps, c.hc_eps, c.sinkhorn);
    }

    /// The layer's KDA or DSA block on b.normed (forward._mixer).
    fn mixer(e: *Engine, l: *const wt.Layer, b: *Buffers, rows: usize, host_pos: usize, done: ?[2]usize) !u64 {
        return switch (l.kind) {
            .kda => try e.kdaBlock(l, b, rows, host_pos, done),
            .mla => blk: {
                const di = e.st.dsa_index[l.index];
                break :blk try e.dsaBlock(l, b, e.st.kc[di], e.st.pos_dev, host_pos, if (e.st.long_context) e.st.index[di] else null, rows, done);
            },
        };
    }

    fn layerForward(e: *Engine, l: *const wt.Layer, b: *Buffers, rows: usize, host_pos: usize) !void {
        const c = e.w.cfg;
        const d: usize = c.hidden;
        const s: usize = c.hc;
        const h = l.attn_hc.?;
        try e.hcPre(b, h, l.in_norm, rows);
        const g = try e.mixer(l, b, rows, host_pos, null);
        try e.t.hcPost(b.x, b.x, g, b.post, b.comb, wt.world, rows, d, s);
        const f = l.ffn_hc.?;
        try e.hcPre(b, f, l.post_norm, rows);
        const g2 = if (l.mlp) |m| try e.mlpBlock(m, b, rows, l.index, null) else try e.moeBlock(l.moe.?, b, rows, l.index, null);
        try e.t.hcPost(b.x, b.x, g2, b.post, b.comb, wt.world, rows, d, s);
    }

    // -- the row split (hcsplit.HcSplit, TF_GLM_HC_SPLIT) ------------------------------------------------------------------

    fn splitActive(e: *const Engine, b: *const Buffers) bool {
        return b.prefill and if (e.split) |sp| sp.active else false;
    }

    /// Whether a prompt chunk of `rows` rows runs split (hcsplit.applies): long enough, its pad row fits, v1.8's experts.
    fn splitApplies(e: *const Engine, b: *const Buffers, rows: usize) bool {
        const sp = e.split orelse return false;
        return b.prefill and e.exl3_v18 and rows >= sp.set.min_rows and (rows + 1) / 2 * 2 <= b.rows;
    }

    fn splitBegin(e: *Engine, b: *const Buffers, rows: usize) !void {
        const c = e.w.cfg;
        const sp = &e.split.?;
        sp.r = rows;
        sp.rp = (rows + 1) / 2 * 2;
        sp.h = sp.rp / 2;
        const np = sp.set.pieceCount();
        var size = (sp.h + np - 1) / np;
        if (size > 64) size = @min(sp.h, (size + 63) / 64 * 64); // whole 64-row tiles for the glue kernels where possible
        sp.n = 0;
        var off: usize = 0;
        while (off < sp.h) : (off += size) {
            sp.pieces[sp.n] = .{ off, @min(size, sp.h - off) };
            sp.n += 1;
        }
        if (sp.rp > rows) { // the pad row: zero streams and partial, never read
            try e.zero(b.x + rows * c.hc * c.hidden * bf, c.hc * c.hidden * bf);
            try e.zero(b.part + rows * c.hidden * f4, c.hidden * f4);
        }
        if (sp.side) |side| {
            try sp.ev_begin.record(e.s);
            try side.wait(sp.ev_begin);
        }
        sp.active = true;
    }

    fn splitFinish(e: *Engine) !void {
        const sp = &e.split.?;
        if (sp.side) |side| if (sp.active) {
            try sp.ev_done.record(side);
            try e.s.wait(sp.ev_done);
        };
        sp.active = false;
    }

    fn mine(e: *const Engine) usize {
        return e.w.rank * e.split.?.h;
    }

    fn theirs(e: *const Engine) usize {
        return (1 - e.w.rank) * e.split.?.h;
    }

    /// `f`'s rows lo .. hi of b.part, clipped to the chunk (hcsplit._fill).
    fn fillRows(e: *Engine, b: *const Buffers, f: Fill, lo: usize, hi_in: usize) !void {
        const hi = @min(hi_in, e.split.?.r);
        if (hi <= lo) return;
        const c = e.w.cfg;
        const d: usize = c.hidden;
        switch (f) {
            .proj => |p| try e.mm(b, p.x + lo * p.q.k * bf, p.q.k, p.xs + lo * p.q.k / 64 * f4, p.q, b.part + lo * d * f4, hi - lo, true),
            .moe => |m| {
                for (undone(hi, m.done)) |r| {
                    const a = @max(r[0], lo);
                    if (a < r[1]) try e.sharedRows(b, m.m.shared, a, r[1]);
                }
                const slots: usize = c.topk + 1;
                try e.t.combineSy(b.exl3.y + lo * slots * d * f4, b.sy + lo * d * f4, b.wts + lo * slots * f4, b.part + lo * d * f4, hi - lo, d, slots);
            },
        }
    }

    /// Piece j: this rank's partial of the peer's rows out, the peer's partial of this rank's rows in (b.gath, unused while split).
    fn swapPartial(e: *Engine, s: cuda.Stream, b: *const Buffers, j: usize) !void {
        const d: usize = e.w.cfg.hidden;
        const off, const n = e.split.?.pieces[j];
        try e.comm.exchange(s, &.{.{ .ptr = b.part + (e.theirs() + off) * d * f4, .bytes = n * d * f4 }}, &.{.{ .ptr = b.gath + off * d * f4, .bytes = n * d * f4 }}, 1 - e.w.rank);
    }

    /// Piece j of bf16 [rows, D] buffers: this rank's rows out, the peer's in.
    fn swapRows(e: *Engine, s: cuda.Stream, j: usize, outs: []const u64) !void {
        const d: usize = e.w.cfg.hidden;
        const off, const n = e.split.?.pieces[j];
        var sends: [8]Comm.Seg = undefined;
        var recvs: [8]Comm.Seg = undefined;
        for (outs, 0..) |t, i| {
            sends[i] = .{ .ptr = t + (e.mine() + off) * d * bf, .bytes = n * d * bf };
            recvs[i] = .{ .ptr = t + (e.theirs() + off) * d * bf, .bytes = n * d * bf };
        }
        try e.comm.exchange(s, sends[0..outs.len], recvs[0..outs.len], 1 - e.w.rank);
    }

    /// hcsplit.partial: the peer's rows of this rank's partial first, piece by piece, each piece's swap starting as it is done
    /// (overlap), then this rank's rows.
    fn splitPartial(e: *Engine, b: *const Buffers, f: Fill) !void {
        const sp = &e.split.?;
        const side = sp.side orelse {
            try e.fillRows(b, f, 0, sp.r);
            for (0..sp.n) |j| try e.swapPartial(e.s, b, j);
            return;
        };
        for (sp.pieces[0..sp.n], 0..) |pc, j| {
            try e.fillRows(b, f, e.theirs() + pc[0], e.theirs() + pc[0] + pc[1]);
            try sp.ev_fill[j].record(e.s);
            try side.wait(sp.ev_fill[j]);
            try e.swapPartial(side, b, j);
            try sp.ev_part[j].record(side);
        }
        try e.fillRows(b, f, e.mine(), e.mine() + sp.h);
    }

    fn runFront(e: *Engine, b: *const Buffers, f: Front, lo: usize, hi_in: usize) !void {
        const hi = @min(hi_in, e.split.?.r);
        if (hi <= lo) return;
        switch (f) {
            .none => {},
            .kda => |l| try e.kdaFront(l, b, lo, hi),
            .dsa => |l| try e.dsaFront(l, b, lo, hi),
            .mlp => |m| try e.mlpFront(m, b, lo, hi),
            .shared => |m| try e.sharedRows(b, m, lo, hi),
        }
    }

    /// hcsplit.glue: hc_post of this rank's rows (both partials summed rank 0 first), then the layer's DFlash2 taps, the final
    /// stream mean (`final`) and the next hc_pre (`hc`), their rows swapped with the peer's; `front` on own rows of each piece
    /// while its swap is on the wire.
    fn splitGlue(e: *Engine, b: *const Buffers, hc: ?wt.Hc, norm: u64, tap_layer: ?u32, final: bool, front: Front) !void {
        const c = e.w.cfg;
        const d: usize = c.hidden;
        const s: usize = c.hc;
        const sp = &e.split.?;
        var outs: [8]u64 = undefined;
        var no: usize = 0;
        if (tap_layer) |tl| for (e.tap_layers, 0..) |t, slot| if (t == tl and b.taps[slot] != 0) {
            outs[no] = b.taps[slot];
            no += 1;
        };
        if (final) {
            outs[no] = b.hidden;
            no += 1;
        }
        if (hc != null) {
            outs[no] = b.normed;
            no += 1;
        }
        for (sp.pieces[0..sp.n], 0..) |pc, j| {
            const lo = e.mine() + pc[0];
            const n = pc[1];
            if (sp.side != null) try e.s.wait(sp.ev_part[j]);
            // glue.hc_post_pair: rank 1's partial rows read at their distance from rank 0's (may be negative)
            const own = b.part + lo * d * f4;
            const got = b.gath + pc[0] * d * f4;
            const g0, const g1 = if (e.w.rank == 0) .{ own, got } else .{ got, own };
            const rs = @divExact(@as(i64, @intCast(g1)) - @as(i64, @intCast(g0)), f4);
            const x = b.x + lo * s * d * bf;
            try e.t.hcPostPair(x, x, g0, b.post + lo * s * f4, b.comb + lo * s * s * f4, rs, n, d, s);
            if (tap_layer) |tl| for (e.tap_layers, 0..) |t, slot| if (t == tl and b.taps[slot] != 0) try e.t.streamMean(x, b.taps[slot] + lo * d * bf, n, d, s);
            if (final) try e.t.streamMean(x, b.hidden + lo * d * bf, n, d, s);
            if (hc) |h| try e.hcPreRows(b, h, norm, lo, n);
            if (sp.side) |side| {
                try sp.ev_glue[j].record(e.s);
                try side.wait(sp.ev_glue[j]);
                try e.swapRows(side, j, outs[0..no]);
            } else try e.swapRows(e.s, j, outs[0..no]);
            try e.runFront(b, front, lo, lo + n);
        }
        if (sp.side) |side| {
            try sp.ev_done.record(side);
            try e.s.wait(sp.ev_done);
        }
    }

    /// forward.split_layers: every layer of a prompt chunk with its glue split by rows; the first hc_pre runs on every row as
    /// unsplit, and b.normed, the taps and b.hidden come back whole from the peer.
    fn splitLayers(e: *Engine, b: *Buffers, rows: usize) !void {
        try e.splitBegin(b, rows);
        const fronts = e.split.?.set.fronts;
        const layers = e.w.layers;
        try e.hcPreRows(b, layers[0].attn_hc.?, layers[0].in_norm, 0, e.split.?.rp);
        var done: ?[2]usize = null;
        for (layers, 0..) |*l, i| {
            _ = try e.mixer(l, b, rows, e.st.pos, done);
            const ff: Front = if (!fronts) .none else if (l.mlp) |m| .{ .mlp = m } else .{ .shared = l.moe.?.shared };
            try e.splitGlue(b, l.ffn_hc.?, l.post_norm, null, false, ff);
            const own: [2]usize = .{ e.mine(), @min(e.mine() + e.split.?.h, rows) };
            _ = if (l.mlp) |m| try e.mlpBlock(m, b, rows, l.index, if (fronts) own else null) else try e.moeBlock(l.moe.?, b, rows, l.index, if (fronts) own else null);
            const nxt: ?*const wt.Layer = if (i + 1 < layers.len) &layers[i + 1] else null;
            const mf: Front = if (!fronts or nxt == null) .none else switch (nxt.?.kind) {
                .kda => .{ .kda = nxt.? },
                .mla => .{ .dsa = nxt.? },
            };
            try e.splitGlue(b, if (nxt) |n| n.attn_hc.? else null, if (nxt) |n| n.in_norm else 0, l.index, nxt == null, mf);
            done = if (fronts and nxt != null) own else null;
        }
        try e.splitFinish();
    }

    /// forward.compute with host_pos: logits of every row (decode) or of the last row (a prompt chunk).
    pub fn compute(e: *Engine, b: *Buffers, rows: usize) !void {
        const prefetch = !b.prefill and !e.l2pf_active and if (e.l2pf) |p| rows <= p.s.rows else false;
        if (!prefetch) return e.computeWindow(b, rows);
        e.l2pf_active = true; // TF_GLM_L2PF: this decode window's sites prefetch (joined below)
        defer e.l2pf_active = false;
        try e.computeWindow(b, rows);
        try e.l2pf.?.join(e.s);
    }

    /// The L2 prefetch at `layer`'s site `which` (0 a, 1 f, 2 o) while a decode window runs.
    fn pfSite(e: *Engine, layer: u32, which: usize) !void {
        if (e.l2pf_active) try e.l2pf.?.site(e.k, e.s, layer, which);
    }

    fn computeWindow(e: *Engine, b: *Buffers, rows: usize) !void {
        const c = e.w.cfg;
        const d: usize = c.hidden;
        try e.t.embed(b.ids, e.w.embed, b.x, rows, d, c.hc);
        if (e.splitApplies(b, rows)) {
            try e.splitLayers(b, rows); // also leaves the taps and b.hidden
        } else {
            for (e.w.layers) |*l| {
                try e.layerForward(l, b, rows, e.st.pos);
                for (e.tap_layers, 0..) |tl, slot| if (tl == l.index and b.taps[slot] != 0) try e.t.streamMean(b.x, b.taps[slot], rows, d, c.hc);
            }
            try e.t.streamMean(b.x, b.hidden, rows, d, c.hc);
        }
        try e.t.rmsnorm(b.hidden, d, e.w.norm, b.fnormed, d, b.fxs, rows, d, c.eps);
        if (b.prefill) {
            try e.mm(b, b.fnormed + (rows - 1) * d * bf, d, b.fxs, e.w.head, b.logits, 1, false);
        } else try e.mm(b, b.fnormed, d, b.fxs, e.w.head, b.logits, rows, false);
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
        try e.mm(b, b.mcat, 2 * d, b.mxs, m.eh, b.mx, n, false);
        const l = &m.layer;
        try e.t.rmsnorm(b.mx, d, l.in_norm, b.normed, d, b.xs, n, d, c.eps);
        const ix: ?[3]u64 = if (e.st.long_context) e.st.index[e.st.dsa_layers] else null;
        const g = try e.dsaBlock(l, b, e.st.mtp_kc, e.st.mtp_pos_dev, e.st.mtp_len, ix, n, null);
        try e.t.residualAdd(b.mx, b.mx, g, wt.world, n, d);
        try e.t.rmsnorm(b.mx, d, l.post_norm, b.normed, d, b.xs, n, d, c.eps);
        const g2 = try e.moeBlock(l.moe.?, b, n, l.index, null);
        try e.t.residualAdd(b.mx, b.mx, g2, wt.world, n, d);
        try e.t.rmsnorm(b.mx + (n - 1) * d * bf, d, m.head_norm, b.fnormed, d, b.fxs, 1, d, c.eps);
        try e.draftHead(b, b.fnormed, 1);
    }

    /// The MTP head's logits from the 4-bit draft head (mtp.py: `w.draft_head`, forward.mm): a prompt chunk's on the
    /// prefill matmul, a decode step's on qmm_group with the rows' group sums (b.fxs, from the final norm).
    fn draftHead(e: *Engine, b: *const Buffers, x: u64, rows: usize) !void {
        if (b.prefill) return kern.q4Prefill(e.k, e.s, x, e.draft_head.k, e.draft_head, b.logits, rows, false);
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

    /// The most candidates a rank sends for a nucleus draw (sampling.WIDER).
    pub const max_nucleus = 16384;

    /// sampling.nucleus_rows' `_shares` for `rows` rows of `logits` (this rank's half of the vocabulary, bf16): every rank's top
    /// `count` (value, id, mass) per row and its shard's mass total and width, as 64-bit words [rank][row][count float64 values,
    /// count ids, count masses, sum, width], gathered on every rank. The row maxima come from `greedy`'s gather.
    pub fn nucleusGather(e: *Engine, logits: u64, rows: usize, temperature: f64, count: usize) ![]const u64 {
        if (rows == 0 or rows > max_rows or count == 0 or count > max_nucleus) return error.BadNucleus;
        const world = wt.world;
        const vn = e.w.head.n;
        const per_row = 3 * max_nucleus + 2;
        // [sums: max_rows words][candidate pairs: max_rows * 2 * max_nucleus u32][packed: max_rows * per_row words][gathered]
        const cand_off = max_rows * 8;
        const loc_off = cand_off + max_rows * 2 * max_nucleus * 4;
        const all_off = loc_off + max_rows * per_row * 8;
        if (e.nuc_dev == null) {
            e.nuc_dev = try cuda.DeviceBuffer.alloc(e.d, all_off + world * max_rows * per_row * 8);
            e.nuc_host = try std.heap.page_allocator.alloc(u64, world * max_rows * per_row);
        }
        const base = e.nuc_dev.?.ptr;
        const inv_t: f64 = 1.0 / @max(temperature, 1e-6);
        const b = &e.buf;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(i32, @intCast(vn)));
        a.add(@as(i32, @intCast(vn)));
        a.add(@as(i32, @intCast(e.w.rank * vn)));
        a.add(b.greedy);
        try cuda.launch.launch(e.k.greedy, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.s, &a);
        try e.comm.lib.check(e.comm.lib.api.ncclAllGather(b.greedy, b.greedy_all, rows * 2, .u32, e.comm.comm, e.s.handle), "ncclAllGather");
        var st: cuda.Args = .{};
        st.add(logits);
        st.add(@as(i32, @intCast(vn)));
        st.add(@as(i32, @intCast(vn)));
        st.add(inv_t);
        st.add(b.greedy_all);
        st.add(@as(i32, world));
        st.add(@as(i32, @intCast(rows)));
        st.add(base);
        try cuda.launch.launch(e.k.nuc_stats, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.s, &st);
        var tk: cuda.Args = .{};
        tk.add(logits);
        tk.add(@as(i32, @intCast(vn)));
        tk.add(@as(i32, @intCast(vn)));
        tk.add(@as(i32, @intCast(count)));
        tk.add(@as(i32, @intCast(e.w.rank * vn)));
        tk.add(base + cand_off);
        try cuda.launch.launch(e.k.topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.s, &tk);
        const words = rows * (3 * count + 2);
        var pk: cuda.Args = .{};
        pk.add(base + cand_off);
        pk.add(@as(i32, @intCast(count)));
        pk.add(inv_t);
        pk.add(b.greedy_all);
        pk.add(@as(i32, world));
        pk.add(@as(i32, @intCast(rows)));
        pk.add(base);
        pk.add(@as(i32, @intCast(vn)));
        pk.add(base + loc_off);
        try cuda.launch.launch(e.k.nuc_pack, .{ .grid = .{ .x = @intCast((count + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, e.s, &pk);
        try e.comm.lib.check(e.comm.lib.api.ncclAllGather(base + loc_off, base + all_off, words * 2, .u32, e.comm.comm, e.s.handle), "ncclAllGather");
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(e.nuc_host.ptr, base + all_off, world * words * 8, e.s.handle), "cuMemcpyDtoHAsync");
        try e.s.synchronize();
        return e.nuc_host[0 .. world * words];
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
        try e.comm.allGather(e.s, b.greedy, b.greedy_all, rows * 2);
        var host: [wt.world * max_rows * 2]u32 = undefined;
        const n = wt.world * rows * 2;
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(&host, b.greedy_all, n * 4, e.s.handle), "cuMemcpyDtoHAsync");
        try e.s.synchronize();
        if (e.comm.roce) |r| try r.check();
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

/// TF_GLM_WIDE_GRAPHS: the verify-window width past 8 rows that replays a CUDA graph, 9 to max_rows (default 16, v1.8's
/// production setting; 0 or empty: none). Python takes a list of widths; one is all the copy drafts' padding uses.
pub fn wideGraph() !usize {
    const v = std.c.getenv("TF_GLM_WIDE_GRAPHS") orelse return 16;
    const t = std.mem.trim(u8, std.mem.span(v), " ");
    if (t.len == 0 or std.mem.eql(u8, t, "0")) return 0;
    const n = std.fmt.parseInt(usize, t, 10) catch return error.BadWideGraphs;
    if (n <= 8 or n > Engine.max_rows) {
        std.log.err("TF_GLM_WIDE_GRAPHS is 0 or a width from 9 to {d}, not {s}", .{ Engine.max_rows, t });
        return error.BadWideGraphs;
    }
    return n;
}

/// TF_GLM_KDA_CHUNKED: on unless "0".
pub fn kdaChunked() bool {
    return if (std.c.getenv("TF_GLM_KDA_CHUNKED")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
}

/// engine.prompt_grid: the rows a prompt chunk and a kept prompt state start on. Chunked KDA depends on the alignment of its
/// 32-row sub-chunks, so under TF_GLM_KDA_CHUNKED the grid is 64 (a multiple of 32 that divides the 2,048-row chunks).
pub fn promptGrid() u32 {
    return if (kdaChunked()) 64 else 0;
}

/// A switch that is on unless set to "0".
fn envOn(name: [*:0]const u8) bool {
    return if (std.c.getenv(name)) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
}

/// TF_GLM_KV: fp8 (the default) or bf16 (kv8.kv_kind).
pub fn kvFp8() !bool {
    const v = std.c.getenv("TF_GLM_KV") orelse return true;
    const k = std.mem.span(v);
    if (k.len == 0 or std.mem.eql(u8, k, "fp8")) return true;
    if (std.mem.eql(u8, k, "bf16")) return false;
    std.log.err("TF_GLM_KV is bf16 or fp8, not {s}", .{k});
    return error.BadKvKind;
}

pub fn denseLimit(w: *const Weights) usize {
    return @as(usize, w.cfg.i_topk) + w.cfg.kpool - 1;
}
