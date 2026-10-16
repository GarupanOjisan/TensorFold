//! The forward engine's set-up, lanes, CUDA graphs, device copies, position state and kept prompt states.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const tri = @import("glm_triton.zig");
const kern = @import("glm_kernels.zig");
const l2pf = @import("glm_l2pf.zig");
const fwd = @import("glm_forward.zig");
const buffers = @import("glm_forward_buffers.zig");

const Engine = fwd.Engine;
const Weights = wt.Weights;
const bf = fwd.bf;
const f4 = fwd.f4;
const Comm = fwd.Comm;
const SplitSettings = fwd.SplitSettings;
const Split = fwd.Split;
const GraphKey = fwd.GraphKey;
const wideGraph = fwd.wideGraph;
const kdaChunked = fwd.kdaChunked;
const envOn = fwd.envOn;
const kvFp8 = fwd.kvFp8;
const denseLimit = fwd.denseLimit;
const Arena = buffers.Arena;
const Buffers = buffers.Buffers;
const carve = buffers.carve;
const State = buffers.State;
const Pool = buffers.Pool;
const max_rows = Engine.max_rows;
const shared_rows = Engine.shared_rows;
const mtp_rows = Engine.mtp_rows;
const max_graphs = Engine.max_graphs;
const max_lanes = Engine.max_lanes;
const prefill_rows = Engine.prefill_rows;

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
    var e: Engine = undefined;
    // count, then allocate once and carve
    var count: Arena = .{ .base = 0, .size = 0 };
    const tap_n: usize = if (taps != null) 5 else 0;
    _ = carve(&count, w, shared_rows, capacity, false, np_max, tap_n);
    _ = carve(&count, w, prefill_rows, capacity, true, np_max, tap_n);
    _ = carve(&count, w, mtp_rows, capacity, false, np_max, 0);
    const kv_fp8 = try kvFp8();
    const shared_bytes = shared_rows * width * bf + shared_rows * ll * 128 * bf + 512;
    const total = count.used + shared_bytes + 4096;
    e.mem = try cuda.DeviceBuffer.alloc(d, total);
    errdefer e.mem.free();
    try e.mem.fill8(0, null);
    var a: Arena = .{ .base = e.mem.ptr, .size = total };
    e.buf = carve(&a, w, shared_rows, capacity, false, np_max, tap_n);
    e.pbuf = carve(&a, w, prefill_rows, capacity, true, np_max, tap_n);
    e.mbuf = carve(&a, w, mtp_rows, capacity, false, np_max, 0);
    kern.fuse_gu = if (std.c.getenv("TF_GLM_EXL3_FUSE2")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    e.tap_layers = taps orelse @splat(std.math.maxInt(u32));
    // lane 0 apart, so `addLanes` can give its room to the pool
    const lane0 = State.bytes(w, capacity, long_context, kv_fp8, false);
    e.lane_mem[0] = try cuda.DeviceBuffer.alloc(d, lane0);
    errdefer e.lane_mem[0].free();
    try e.lane_mem[0].fill8(0, null);
    var a0: Arena = .{ .base = e.lane_mem[0].ptr, .size = lane0 };
    e.st = State.carveIn(&a0, w, capacity, long_context, kv_fp8, false);
    e.pool = null;
    e.seg_sel = null;
    e.seg_enabled = envOn("TF_GLM_SEG");
    e.sproj = a.take(shared_rows * width * bf);
    e.sout = a.take(shared_rows * ll * 128 * bf);
    e.n_lanes = 1;
    e.lane = 0;
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
    e.pieces = &.{};
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

/// One more lane's device bytes once pooled: its own part and its share of the pool; notes/glm_forward.md#lanes
pub fn laneBytes(e: *const Engine) usize {
    const st = &e.st;
    return State.bytes(e.w, st.capacity, st.long_context, st.kv_fp8, true) + Pool.bytes(e.w, st.capacity, st.long_context, st.kv_fp8, 2) - Pool.bytes(e.w, st.capacity, st.long_context, st.kv_fp8, 1);
}

/// Lanes up to `total`, all zeroed, their caches in one Pool; returns how many there are; notes/glm_forward.md#lanes
pub fn addLanes(e: *Engine, total: usize) !usize {
    const want = @min(total, max_lanes);
    if (want <= e.n_lanes) return e.n_lanes;
    if (e.n_lanes != 1 or e.pool != null) return error.LanesAlreadyPooled;
    const st0 = e.st;
    const cap = st0.capacity;
    // the captured graphs hold lane 0's old addresses
    for (e.graphs[0..e.n_graphs]) |*g| g.exec.deinit();
    e.n_graphs = 0;
    e.lane_mem[0].free();
    e.n_lanes = 0; // until the pool's lanes are made (deinit frees lane_mem[0 .. n_lanes])
    var pool: Pool = blk: {
        var n = want;
        while (n >= 1) : (n -= 1) {
            const size = Pool.bytes(e.w, cap, st0.long_context, st0.kv_fp8, n);
            var m = cuda.DeviceBuffer.alloc(e.d, size) catch continue;
            m.fill8(0, null) catch {
                m.free();
                continue;
            };
            var a: Arena = .{ .base = m.ptr, .size = size };
            var p = Pool.carvePool(&a, e.w, cap, st0.long_context, st0.kv_fp8, n);
            p.mem = m;
            break :blk p;
        }
        return error.OutOfMemory;
    };
    errdefer pool.mem.free();
    const own = State.bytes(e.w, cap, st0.long_context, st0.kv_fp8, true);
    var made: usize = 0;
    while (made < pool.slots) : (made += 1) {
        var m = cuda.DeviceBuffer.alloc(e.d, own) catch break;
        m.fill8(0, null) catch {
            m.free();
            break;
        };
        var a: Arena = .{ .base = m.ptr, .size = own };
        var st = State.carveIn(&a, e.w, cap, st0.long_context, st0.kv_fp8, true);
        pool.view(&st, made, e.w.cfg.i_dim);
        e.lane_mem[made] = m;
        e.lanes[made] = st;
    }
    if (made == 0) return error.OutOfMemory;
    e.pool = pool;
    e.n_lanes = made;
    e.st = e.lanes[0];
    e.lane = 0;
    return made;
}

/// Makes lane `i`'s state the one every call reads and writes (`st`).
pub fn bind(e: *Engine, i: usize) !void {
    if (i >= e.n_lanes) return error.NoSuchLane;
    if (i == e.lane) return;
    e.lanes[e.lane] = e.st;
    e.st = e.lanes[i];
    e.lane = @intCast(i);
}

pub fn laneState(e: *Engine, i: usize) *State {
    return if (i == e.lane) &e.st else &e.lanes[i];
}

/// Lane `src`'s attention-cache rows below `rows` into lane `dst`; notes/glm_forward.md#lanes
pub fn copyLaneRows(e: *Engine, src: usize, dst: usize, rows: usize) !void {
    if (src >= e.n_lanes or dst >= e.n_lanes or src == dst) return error.NoSuchLane;
    const c = e.w.cfg;
    const a = e.laneState(src);
    const b = e.laneState(dst);
    if (rows > a.capacity) return error.PastCacheCapacity;
    for (0..a.dsa_layers) |i| try e.copyDev(b.kc[i], a.kc[i], rows * a.latentRow(c));
    try e.copyDev(b.mtp_kc, a.mtp_kc, rows * a.latentRow(c));
    if (a.long_context) for (0..a.dsa_layers + 1) |i| {
        try e.copyDev(b.index[i][0], a.index[i][0], rows * c.i_dim * bf);
        try e.copyDev(b.index[i][1], a.index[i][1], rows * c.i_dim * bf);
        try e.copyDev(b.index[i][2], a.index[i][2], @min(rows / 4 + 2, a.capacity / 4 + 2) * a.pooledRow(c));
    };
}

pub fn deinit(e: *Engine) void {
    for (e.lane_mem[0..e.n_lanes]) |*m| m.free();
    if (e.pool) |*p| p.mem.free();
    if (e.seg_sel) |*b| b.free();
    for (e.graphs[0..e.n_graphs]) |*g| g.exec.deinit();
    if (e.l2pf) |*p| p.deinit();
    e.full_dev.free();
    if (e.split) |*sp| sp.deinit();
    if (e.nuc_dev) |*b| b.free();
    if (e.nuc_host.len > 0) std.heap.page_allocator.free(e.nuc_host);
    std.heap.page_allocator.free(e.full_host);
    e.mem.free();
}

/// The graph key of a window, or null when it runs eagerly (graphs.py); notes/glm_forward.md#graph-count
pub fn graphKey(e: *const Engine, head: GraphKey.Head, rows: usize, pos: usize) ?GraphKey {
    if (!e.use_graphs) return null;
    if (rows > 8 and rows != e.wide_graph) return null; // engine.GRAPH_ROWS + WIDE_GRAPHS: the others run eagerly
    const dl = denseLimit(e.w);
    var np: usize = 0;
    if (e.st.long_context and pos + rows - 1 >= dl) {
        if (pos < dl) return null;
        np = tri.poolBucket(pos, rows, e.np_max);
    }
    return .{ .head = head, .rows = @intCast(rows), .cur = if (head == .main) @intCast(e.st.cur) else 0, .np = @intCast(np), .lane = e.lane };
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

pub fn setInt(e: *Engine, dst: u64, v: usize) !void {
    const x: i32 = @intCast(v);
    try e.upload(dst, std.mem.asBytes(&x));
}

pub fn reset(e: *Engine) !void {
    const c = e.w.cfg;
    const ll: usize = c.kda_heads / wt.world;
    try e.zero(e.st.conv, e.st.kda_layers * 3 * 3 * ll * 128 * bf);
    try e.zero(e.st.rec, 2 * e.st.kda_layers * ll * 128 * 128 * f4);
    e.st.cur = 0;
    e.st.seg = false;
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

pub fn checkRoom(e: *Engine, rows: usize, pos: usize) !void {
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

/// A kept prompt state's bytes: every KDA layer's state in use and conv window; notes/glm_forward.md#kept-states
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
    e.st.seg = false;
    try e.setPos(at);
    try e.setMtpLen(at);
    e.st.mtp_drafted = 0;
}
