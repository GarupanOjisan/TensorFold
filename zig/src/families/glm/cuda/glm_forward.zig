//! GLM-5.3-Flash's tensor-parallel forward, MTP step and commit on CUDA; notes/glm_forward.md#engine
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const tri = @import("glm_triton.zig");
const kern = @import("glm_kernels.zig");
const segments = @import("glm_segments.zig");
const l2pf = @import("glm_l2pf.zig");
const roce_mod = @import("glm_roce.zig");
const buffers = @import("glm_forward_buffers.zig");
const lane_fns = @import("glm_forward_lanes.zig");
const block_fns = @import("glm_forward_blocks.zig");
const prompt_fns = @import("glm_forward_prompt.zig");
const shared_fns = @import("glm_forward_shared.zig");
const sample_fns = @import("glm_forward_sample.zig");

const Weights = wt.Weights;
const Mat = wt.Mat;
pub const Buffers = buffers.Buffers;
pub const KdaScratch = buffers.KdaScratch;
pub const State = buffers.State;
pub const extent = buffers.extent;
pub const Pool = buffers.Pool;

pub const bf = 2;
pub const f4 = 4;

/// Two ranks' all-gather of fp32 partials (comm.py over NCCL): rank r's `count` floats land at recv + r * count.
pub const Comm = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    roce: ?*roce_mod.Roce = null,

    /// Small gathers over RoCE when it is up, the rest through NCCL, chosen by byte count; notes/glm_forward.md#comm
    pub fn allGather(c: Comm, s: cuda.Stream, send: u64, recv: u64, count: usize) !void {
        if (c.roce) |r| if (r.fits(count * 4) and send % 16 == 0 and recv % 16 == 0) return r.gather(s, send, recv, count * 4);
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, .f32, c.comm, s.handle), "ncclAllGather");
    }

    /// comm.exchange: byte ranges to and from the peer in one NCCL group, always through NCCL (hcsplit's swaps).
    pub fn exchange(c: Comm, s: cuda.Stream, sends: []const Seg, recvs: []const Seg, peer: usize) !void {
        const api = c.lib.api;
        try c.lib.check(api.ncclGroupStart(), "ncclGroupStart");
        for (sends) |x| try c.lib.check(api.ncclSend(x.ptr, x.bytes, .u8, @intCast(peer), c.comm, s.handle), "ncclSend");
        for (recvs) |x| try c.lib.check(api.ncclRecv(x.ptr, x.bytes, .u8, @intCast(peer), c.comm, s.handle), "ncclRecv");
        try c.lib.check(api.ncclGroupEnd(), "ncclGroupEnd");
    }

    pub const Seg = struct { ptr: u64, bytes: usize };
};

/// TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP (v1.10 hcsplit.SplitSettings); notes/glm_forward.md#row-split
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

/// One prompt buffer's row split between the two ranks (hcsplit.HcSplit); notes/glm_forward.md#row-split
pub const Split = struct {
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

    pub fn init(d: *const cuda.Driver, set: SplitSettings) !Split {
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

    pub fn deinit(sp: *Split) void {
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

/// What a split site writes into b.part for rows lo .. hi (forward.partials' `fill`); notes/glm_forward.md#row-split
pub const Fill = union(enum) {
    proj: struct { x: u64, xs: u64, q: Mat },
    moe: struct { m: wt.Moe, done: ?[2]usize },
};

/// The next block's row-independent start the glue runs on own rows (forward.mixer_front / ffn_front, OVERLAP=2).
pub const Front = union(enum) { none, kda: *const wt.Layer, dsa: *const wt.Layer, mlp: wt.Mlp, shared: wt.Mlp };

/// forward.undone: the row ranges of [0, rows) outside `done`, in row order; an empty range has hi <= lo.
pub fn undone(rows: usize, done: ?[2]usize) [2][2]usize {
    const d = done orelse return .{ .{ 0, rows }, .{ 0, 0 } };
    return .{ .{ 0, @min(d[0], rows) }, .{ d[1], rows } };
}

/// A captured graph's key: which head, window rows, the KDA state half it reads (main only), pool bucket (0: dense).
pub const GraphKey = struct {
    lane: u8,
    head: Head,
    rows: u8,
    cur: u8,
    np: u32,

    pub const Head = enum(u8) { main, mtp, dtaps, dblock, seg, dbatch, dtbatch }; // DFlash2's context update, block pass and the lanes' batched block pass (glm_dflash.zig)
};
const GraphEntry = struct { key: GraphKey, exec: cuda.graph.Exec };

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
    /// Every lane's state but the bound one's (in `st`; bind swaps them); their memory is in `lane_mem`.
    lanes: [max_lanes]State = undefined,
    lane_mem: [max_lanes]cuda.DeviceBuffer = undefined, // each lane's own part (all of lane 0's until `addLanes` pools them)
    /// Every lane's KDA states and DSA caches once `addLanes` made several (null: lane 0 alone, all in lane_mem[0]).
    pool: ?Pool = null,
    /// The segmented window's host tables (uploaded into the pool's), its selection buffers, TF_GLM_SEG (default on).
    seg_host: [segments.SegRows.words]i32 = undefined,
    kseg_host: [max_lanes]kern.SegRow = undefined,
    seg_sel: ?cuda.DeviceBuffer = null,
    seg_enabled: bool = true,
    n_lanes: usize = 1,
    lane: u8 = 0,
    draft_head: kern.Q4, // weights.draft_head: the head quantized to 4 bits, read by draft steps only
    /// Both ranks' logits rows for a draw over the whole vocabulary: [rank][row][head.n] bf16, on the device and here.
    full_dev: cuda.DeviceBuffer,
    full_host: []u16,
    /// The last `candidates` call's gathered pairs: [rank][row][k] of (fp32 value bits, token).
    cand_host: [wt.world * shared_rows * 2 * max_k]u32 = undefined, // a shared window's rows at most (Node.pickRows)
    /// Captured decode windows and MTP steps (graphs.py), found by key; TF_GLM_GRAPHS=0 runs everything eagerly.
    graphs: [max_graphs]GraphEntry = undefined,
    n_graphs: usize = 0,
    use_graphs: bool = true,
    /// TF_GLM_WIDE_GRAPHS: the one window width past 8 rows that replays a graph; notes/glm_forward.md#wide-graph
    wide_graph: usize = 16,
    /// DFlash2's tap layers (the target layers whose stream means `Buffers.taps` take); none when taps are off.
    tap_layers: [5]u32 = @splat(std.math.maxInt(u32)),
    /// A shared window's KDA projections and chain outputs, every lane's rows back to back (forwardShared).
    sproj: u64 = 0,
    sout: u64 = 0,
    /// TF_GLM_EXL3_DEC (default on): v1.8's routed-expert kernels, else v0.6.6's; notes/glm_forward.md#switches
    exl3_v18: bool = true,
    /// TF_GLM_KV (default fp8): DSA latents and pooled keys as e4m3 rows with a scale; notes/glm_forward.md#switches
    kv_fp8: bool = true,
    /// TF_GLM_HC_DEC / TF_GLM_HC_MMA (default on): mixing dots of decode rows in hc.cu, of prompt rows on tensor cores.
    hc_dec: bool = true,
    /// TF_GLM_L2PF (default on): small decode windows prefetch the next weights into L2; notes/glm_forward.md#switches
    l2pf: ?l2pf.Prefetch = null,
    l2pf_active: bool = false,
    pieces: []const Piece = &.{}, // a multi-prompt chunk's pieces while it runs (prefillPieces)
    hc_mma: bool = true,
    /// TF_GLM_LATENT_MMA / TF_GLM_SPARSE_ONEPASS (default on): prompt absorb/expand on tensor cores, one-pass sparse.
    latent_mma: bool = true,
    sparse_onepass: bool = true,
    /// The indexer's ring length: the rings hold the whole context, so it only picks the 16-divisible kernels.
    index_ring: usize = 0,
    /// TF_GLM_KDA_CHUNKED (default on): prompt chunks run KDA in chunked (WY) form; notes/glm_forward.md#switches
    kda_chunked: bool = true,
    /// TF_GLM_COMBINE_SY (default on, with exl3_v18): decode windows combine the shared expert's row where it is.
    combine_sy: bool = true,
    capturing: bool = false,
    /// The nucleus draw's device scratch (`nucleusGather`), allocated at its first use, and its host copy.
    nuc_dev: ?cuda.DeviceBuffer = null,
    nuc_host: []u64 = &.{},
    /// TF_GLM_HC_SPLIT (default on): prompt chunks' glue split by rows between the ranks; null when off.
    split: ?Split = null,

    pub const max_rows = 16; // engine.MAX_ROWS: a pending token and up to 15 copied drafts
    /// A shared round's rows across lanes (v1.10's TF_GLM_MULTI_WINDOW); a lane's own part stays within max_rows.
    pub const shared_rows = segments.max_rows;
    pub const mtp_rows = 8; // engine.MTP_ROWS: rows an MTP step takes at once (a kept backlog is absorbed in steps of this)
    /// The graphs a lane may capture; notes/glm_forward.md#graph-count
    pub const max_graphs = 352 * max_lanes; // a lane's each (graphs bake in its state's addresses)
    /// Sequences whose caches are allocated at once (one a stream, idle ones keeping their rows for a later turn).
    pub const max_lanes = 8;
    /// The most candidates a rank gives a sampled row (top_k + exact_sampling.MARGIN).
    pub const max_k = 1024;
    pub const prefill_rows = 4096; // geometry.PREFILL_ROWS

    pub const init = lane_fns.init;
    pub const laneBytes = lane_fns.laneBytes;
    pub const addLanes = lane_fns.addLanes;
    pub const bind = lane_fns.bind;
    pub const laneState = lane_fns.laneState;
    pub const copyLaneRows = lane_fns.copyLaneRows;
    pub const deinit = lane_fns.deinit;
    pub const graphKey = lane_fns.graphKey;
    pub const replay = lane_fns.replay;
    pub const upload = lane_fns.upload;
    pub const copyDev = lane_fns.copyDev;
    pub const zero = lane_fns.zero;
    pub const setInt = lane_fns.setInt;
    pub const reset = lane_fns.reset;
    pub const setPos = lane_fns.setPos;
    pub const setMtpLen = lane_fns.setMtpLen;
    pub const checkRoom = lane_fns.checkRoom;
    pub const stage = lane_fns.stage;
    pub const gather = block_fns.gather;
    pub const mm = block_fns.mm;
    pub const outProj = block_fns.outProj;
    pub const kdaFront = block_fns.kdaFront;
    pub const kdaBlock = block_fns.kdaBlock;
    pub const kdaState = block_fns.kdaState;
    pub const dsaFront = block_fns.dsaFront;
    pub const dsaBlock = block_fns.dsaBlock;
    pub const dsaState = block_fns.dsaState;
    pub const blockPositions = block_fns.blockPositions;
    pub const mlpFront = block_fns.mlpFront;
    pub const mlpBlock = block_fns.mlpBlock;
    pub const moeBlock = block_fns.moeBlock;
    pub const sharedExpert = block_fns.sharedExpert;
    pub const sharedRows = block_fns.sharedRows;
    pub const copyRows = block_fns.copyRows;
    pub const hcPre = block_fns.hcPre;
    pub const hcPreRows = block_fns.hcPreRows;
    pub const mixer = block_fns.mixer;

    /// A lane's rows of a multi-prompt chunk, at row `lo` of the prompt buffers; notes/glm_forward.md#pieces
    pub const Piece = struct { lane: usize, rows: usize, lo: usize };

    pub const mixPieces = prompt_fns.mixPieces;
    pub const pieceView = prompt_fns.pieceView;
    pub const prefillPieces = prompt_fns.prefillPieces;
    pub const layerForward = prompt_fns.layerForward;

    pub const splitActive = prompt_fns.splitActive;
    pub const splitApplies = prompt_fns.splitApplies;
    pub const splitBegin = prompt_fns.splitBegin;
    pub const splitFinish = prompt_fns.splitFinish;
    pub const mine = prompt_fns.mine;
    pub const theirs = prompt_fns.theirs;
    pub const fillRows = prompt_fns.fillRows;
    pub const swapPartial = prompt_fns.swapPartial;
    pub const swapRows = prompt_fns.swapRows;
    pub const splitPartial = prompt_fns.splitPartial;
    pub const runFront = prompt_fns.runFront;
    pub const splitGlue = prompt_fns.splitGlue;
    pub const splitLayers = prompt_fns.splitLayers;
    pub const compute = prompt_fns.compute;
    pub const pfSite = prompt_fns.pfSite;
    pub const computeWindow = prompt_fns.computeWindow;
    pub const forward = prompt_fns.forward;

    /// A lane's rows of a shared window, in order.
    pub const Seg = struct { lane: usize, rows: usize };

    pub const forwardShared = shared_fns.forwardShared;
    pub const kdaShared = shared_fns.kdaShared;
    pub const dsaShared = shared_fns.dsaShared;

    pub const segSelBuffers = shared_fns.segSelBuffers;
    pub const segSel = shared_fns.segSel;
    pub const forwardSeg = shared_fns.forwardSeg;
    pub const segBody = shared_fns.segBody;
    pub const kdaSeg = shared_fns.kdaSeg;
    pub const dsaSeg = shared_fns.dsaSeg;
    pub const commitSeg = shared_fns.commitSeg;

    pub const Keep = struct { lane: usize, rows: usize, keep: usize };

    pub const commitSegs = shared_fns.commitSegs;
    pub const commit = shared_fns.commit;
    pub const commitChunk = shared_fns.commitChunk;
    pub const snapshotBytes = lane_fns.snapshotBytes;
    pub const saveState = lane_fns.saveState;
    pub const restoreState = lane_fns.restoreState;
    pub const mtp = sample_fns.mtp;
    pub const mtpBody = sample_fns.mtpBody;
    pub const draftHead = sample_fns.draftHead;
    pub const candidates = sample_fns.candidates;

    /// The most candidates a rank sends for a nucleus draw (sampling.WIDER).
    pub const max_nucleus = 16384;

    pub const nucleusGather = sample_fns.nucleusGather;
    pub const fullRows = sample_fns.fullRows;
    pub const greedy = sample_fns.greedy;
};

/// TF_GLM_WIDE_GRAPHS: 9 to max_rows (default 16), 0 or empty for none; notes/glm_forward.md#wide-graph
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

/// engine.prompt_grid: the rows prompt chunks and kept states start on; notes/glm_forward.md#prompt-grid
pub fn promptGrid() u32 {
    return if (kdaChunked()) 64 else 0;
}

/// A switch that is on unless set to "0".
pub fn envOn(name: [*:0]const u8) bool {
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
