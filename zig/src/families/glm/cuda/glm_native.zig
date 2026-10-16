//! GLM-5.3-Flash's rank 0 of two CUDA ranks behind native/cuda.zig's family interface; notes/glm_native.md#overview
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const node = @import("glm_node.zig");
const fwd = @import("glm_forward.zig");
const copy = @import("glm_copy.zig");
const dflash = @import("glm_dflash.zig");
const glm_sample = @import("glm_native_sample.zig");
const glm_prompt = @import("glm_native_prompt.zig");
const glm_round = @import("glm_native_round.zig");
const glm_timing = @import("glm_native_timing.zig");

const be = lanes.backend;
const Node = node.Node;
pub const max_rows = fwd.Engine.max_rows;
/// Drafts the MTP head chains at most: the 8-row window of earlier builds (the rows past it serve copies).
const head_drafts = 7;
pub const dflash_max = 15;
/// sampling.WIDER: a rank's candidates when a row's nucleus runs past the first ones (union rule), then whole shards.
pub const nucleus_wider = fwd.Engine.max_nucleus;

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{"exl3-mcg-4bit"};
pub const default_context: i64 = 1048576;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = fwd.Engine.prefill_rows;
pub const promptGrid = fwd.promptGrid;
/// Sequences' caches allocated at load (`grow`), a stream each.
pub const max_lanes: u32 = fwd.Engine.max_lanes;
pub const default_port: u16 = 29660;
/// GLM runs on a captured Triton kernel set only (TENSORFOLD_CUDA_KERNELS), never on the built-in glue kernels.
pub const captured_kernels = true;
pub const explains_open = true;

/// `slide` stays false: the server refuses --slide for a family without a learner.
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

pub const Loaded = struct {
    backend: be.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    /// Streams that allocate nothing: the first takes the engine's own sequence (grow adds the rest).
    free_streams: u32 = 1,
    learn_bytes: usize = 0, // no learner: the streams keep all the room
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    target: *anyopaque, // the prompt cache keeps this family's states (snap_*): the same as ctx
};

/// Whether `dir` is an EXL3 checkpoint (quant_method "exl3"), the kind this port reads; notes/glm_native.md#reads
pub fn reads(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return false;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch return false;
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return false;
    if (root != .object) return false;
    const q = root.object.get("quantization_config") orelse return false;
    if (q != .object) return false;
    const m = q.object.get("quant_method") orelse return false;
    return m == .string and std.ascii.eqlIgnoreCase(m.string, "exl3");
}

/// This rank's share of the checkpoint, for the memory admission; notes/glm_native.md#weight-share
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64) u64 {
    _ = a;
    _ = io;
    _ = dir;
    return weightBytes(total);
}

fn weightBytes(total: u64) u64 {
    return total / 2 + (4 << 30) + if (node.drafterDir() != null) @as(u64, 2 << 30) else 0; // DFlash2's 4-bit layers: 1.8 GiB a rank
}

pub const ring = 64;

/// glm_copy.zig as the lane core's proposer (lanes/proposer.zig), one a lane; notes/glm_native.md#copy-rule
const CopyRule = struct {
    cfg: copy.Settings,
    prompt: usize = 0,
    buf: [copy.max_drafts]u32 = undefined,
    memo_len: usize = std.math.maxInt(usize),
    memo_room: i64 = 0,
    memo_n: usize = 0,

    const vtable: lanes.proposer.Proposer.VTable = .{ .propose = proposeFn, .last_match = lastMatchFn };

    pub fn reset(r: *CopyRule, prompt: usize) void {
        r.prompt = prompt;
        r.memo_len = std.math.maxInt(usize);
    }

    pub fn proposer(r: *CopyRule) lanes.proposer.Proposer {
        return .{ .ptr = r, .vtable = &vtable };
    }

    fn proposeFn(ptr: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32 {
        const r: *CopyRule = @ptrCast(@alignCast(ptr));
        if (max_draft <= 0) return &.{};
        if (r.memo_len != context.len or r.memo_room != max_draft) {
            r.memo_n = r.cfg.propose(context, r.prompt, @intCast(max_draft), &r.buf).len;
            r.memo_len = context.len;
            r.memo_room = max_draft;
        }
        return r.buf[0..r.memo_n];
    }

    /// Any copy matched the settings' `match` tokens, which the core's enter_match (8) must not second-guess.
    fn lastMatchFn(ptr: *anyopaque) i64 {
        const r: *CopyRule = @ptrCast(@alignCast(ptr));
        return if (r.memo_n > 0) 64 else 0;
    }
};

/// One sequence's caches (the engine's lane of the same index) and the stream using them, if any.
pub const Lane = struct {
    stream: ?*lanes.Stream = null,
    copy: CopyRule,
    dtoks: [dflash_max]u32 = undefined, // the chain DFlash2 picked for the stream's next window
    dn: usize = 0,
    held: [max_rows]u32 = undefined,
    held_n: usize = 0,
    pending_rows: usize = 0, // the last verify's rows, until a keep or a draft commits them
    window: [max_rows]u32 = undefined, // the last verify's rows, the tokens a commit makes live
    /// The tokens whose attention-cache rows are committed, from position 0; notes/glm_native.md#live-tokens
    live: std.ArrayList(u32) = .empty,
    first: u64 = 0, // the handle of the stream's first token
    used: u64 = 0, // when a stream last took the lane (least recently used goes first)
    fill_at: usize = 0, // a prompt filling in rounds (prefillManyFn): where it stands, and its next planned start
    fill_next: usize = 0,
};

pub const Glm = struct {
    gpa: std.mem.Allocator,
    n: *Node,
    drafts: bool,
    dflash: bool = false, // DFlash2 drafts (the MTP head is not loaded)
    policy: dflash.Policy = .{},
    nucleus_first: usize = 1024, // TF_GLM_NUCLEUS: the candidates a rank sends first for a top_k-off draw (sampling.NUCLEUS)
    nucleus_union: bool = true, // TENSORFOLD_NUCLEUS_UNION
    busy_rows: usize = fwd.Engine.prefill_rows / 2, // TF_GLM_BUSY_ROWS: a filling round's rows while other lanes decode
    copy_cfg: copy.Settings,
    lanes: [max_lanes]Lane = undefined,
    n_lanes: usize = 1,
    clock: u64 = 0,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    costs: [max_rows]lanes.config.Cost = undefined,
    /// Shared windows' ms by total rows (the facts point here); notes/glm_native.md#shared-costs
    shared: [fwd.Engine.shared_rows]lanes.config.Cost = undefined,
    io: std.Io = undefined,
    mtp_ms: f64 = 1.0,
    vtable: be.Backend.VTable = undefined,
    pass: []const u32 = &.{}, // the prompt a prompt pass is running, and where it stands between chunks
    pass_at: ?u32 = null,
    next_snap: u32 = 0,

    pub fn of(p: *anyopaque) *Glm {
        return @ptrCast(@alignCast(p));
    }

    pub fn take(g: *Glm, t: u32) u64 {
        g.drawn[g.next % ring] = t;
        g.next += 1;
        return g.next - 1;
    }

    pub fn value(g: *Glm, f: be.Feed) !u32 {
        return switch (f) {
            .value => |v| v,
            .handle => |h| try readFn(g, h),
        };
    }

    /// The lane stream `s` uses, bound on both ranks.
    pub fn enter(g: *Glm, s: *const lanes.Stream) !*Lane {
        for (g.lanes[0..g.n_lanes], 0..) |*ln, i| if (ln.stream == s) {
            try g.n.bind(i);
            return ln;
        };
        return error.NoLane;
    }

    /// Whether a lane other than `filling` holds a stream.
    pub fn othersDecode(g: *const Glm, filling: []const usize) bool {
        for (g.lanes[0..g.n_lanes], 0..) |*ln, li| {
            if (ln.stream != null and std.mem.indexOfScalar(usize, filling, li) == null) return true;
        }
        return false;
    }

    pub fn laneOf(g: *Glm, s: *const lanes.Stream) ?*Lane {
        for (g.lanes[0..g.n_lanes]) |*ln| if (ln.stream == s) return ln;
        return null;
    }

    /// Commit the bound lane's last verify's first `keep` rows once (keep and the next draft both name them).
    pub fn commit(g: *Glm, ln: *Lane, keep: usize) !void {
        if (ln.pending_rows == 0) return;
        try g.n.keep(ln.pending_rows, keep);
        try ln.live.appendSlice(g.gpa, ln.window[0..keep]);
        ln.pending_rows = 0;
    }

    pub const chooseWhole = glm_sample.chooseWhole;
    pub const drawNucleus = glm_sample.drawNucleus;
    pub const choose = glm_sample.choose;
    pub const chooseIn = glm_sample.chooseIn;
    pub const prefillFn = glm_prompt.prefillFn;
    pub const nextCut = glm_prompt.nextCut;
    pub const firstOf = glm_prompt.firstOf;
    pub const admitPrompt = glm_prompt.admitPrompt;
    pub const prefillManyFn = glm_prompt.prefillManyFn;
    pub const firstFn = glm_round.firstFn;
    pub const queueFn = glm_round.queueFn;
    pub const readFn = glm_round.readFn;
    pub const verifyFn = glm_round.verifyFn;
    pub const verifyShared = glm_round.verifyShared;
    pub const keepFn = glm_round.keepFn;
    pub const draftDflash = glm_round.draftDflash;
    pub const Turn = glm_round.Turn;
    pub const dflashTaps = glm_round.dflashTaps;
    pub const dflashChain = glm_round.dflashChain;
    pub const draftDflashLanes = glm_round.draftDflashLanes;
    pub const treeFn = glm_round.treeFn;
    pub const draftFn = glm_round.draftFn;
    pub const releaseFn = glm_round.releaseFn;
    pub const timing_ids = glm_timing.timing_ids;
    pub const measureShared = glm_timing.measureShared;
    pub const compareOne = glm_timing.compareOne;
    pub const flatten = glm_timing.flatten;
    pub const sharedRows = glm_timing.sharedRows;
    pub const measure = glm_timing.measure;

    fn backend(g: *Glm) be.Backend {
        g.vtable = .{ .prefill = prefillFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .tree = treeFn, .prefill_many = prefillManyFn };
        return .{ .ptr = g, .vtable = &g.vtable };
    }

    fn facts(g: *const Glm) lanes.Model {
        if (!g.drafts) return .{ .exact_width = 1, .max_streams = 1 };
        return .{
            .exact_width = max_rows,
            .first_copy_rows = if (g.copy_cfg.on) @intCast(@min(g.copy_cfg.most + 1, max_rows)) else 0,
            .mtp = true,
            .speculate = true,
            .speculate_early = false,
            .drafts = head_drafts,
            .window_costs = &g.costs,
            .shared_costs = &g.shared,
            .mtp_step_ms = if (g.dflash) 0.1 else g.mtp_ms, // DFlash2's block pass costs the same at any depth
            // several live streams share a forward (verifyShared); notes/glm_native.md#hidden-rows
            .hidden_rows = true,
            .draft_streams = true, // a shared round's drafts in one call: DFlash2's block passes share one (draftDflashLanes)
            .streams_exact = true,
            .batch_rows = @intCast(g.sharedRows()),
            .max_streams = max_lanes,
        };
    }
};

test {
    _ = @import("glm_segments.zig");
    _ = @import("glm_kernels.zig");
    _ = glm_sample;
}

/// A switch that is on unless set to "0".
fn envOff(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

fn envCount(name: [*:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return default;
    return if (n >= 1 and n <= fwd.Engine.max_nucleus) n else default;
}

/// TF_GLM_BUSY_ROWS: a multiple of 64 up to Engine.prefill_rows (that one turns the cap off), else half of it.
fn envBusyRows() usize {
    const half = fwd.Engine.prefill_rows / 2;
    const v = std.c.getenv("TF_GLM_BUSY_ROWS") orelse return half;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return half;
    return if (n >= 64 and n <= fwd.Engine.prefill_rows and n % 64 == 0) n else half;
}

fn envPort() u16 {
    const v = std.c.getenv("TF_GLM_PORT") orelse return default_port;
    return std.fmt.parseInt(u16, std.mem.span(v), 10) catch default_port;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, captured: ?[]const u8, o: Options) !Loaded {
    const kernels = captured orelse return error.NoCapturedKernels;
    std.log.info("GLM-5.3-Flash rank 0: waiting for rank 1 (tf-glm follow) on port {d}", .{envPort()});
    const n = try Node.init(gpa, io, ctx, dir, kernels, o.context, 0, "", envPort());
    errdefer n.deinit();
    const g = try gpa.create(Glm);
    errdefer gpa.destroy(g);
    g.* = .{ .nucleus_union = !envOff("TENSORFOLD_NUCLEUS_UNION"), .nucleus_first = envCount("TF_GLM_NUCLEUS", 1024), .busy_rows = envBusyRows(), .gpa = gpa, .n = n, .drafts = o.drafts and (n.w.mtp != null or n.dflash != null), .dflash = n.dflash != null, .policy = try dflash.Policy.fromEnv(), .copy_cfg = try copy.Settings.fromEnv(n.e.wide_graph) };
    for (&g.lanes) |*ln| ln.* = .{ .copy = .{ .cfg = g.copy_cfg } };
    if (g.dflash and g.drafts) std.log.info("GLM drafts: DFlash2 (block {d}, up to {d} drafts a round, stop at {d:.2} of the noisy chain, beta {d:.1}); the MTP head is not loaded", .{ n.w.draft.?.block, g.policy.most, g.policy.confidence, g.policy.beta });
    if (g.drafts and g.copy_cfg.on) std.log.info("GLM copy drafts: up to {d} a round, match {d} tokens, {d} inside the reply, windows of 9 to 15 rows padded to {d}", .{ g.copy_cfg.most, g.copy_cfg.match, g.copy_cfg.reply_match, g.copy_cfg.pad });
    g.io = io;
    try g.measure(io);
    @memcpy(g.shared[0..max_rows], &g.costs);
    for (g.shared[max_rows..]) |*c| c.* = .{ .width = 0, .ms = 0 }; // width 0: left out (lanes config); measureShared fills them
    return .{ .backend = g.backend(), .facts = g.facts(), .rows = if (g.drafts) max_rows else 1, .stream_bytes = n.laneBytes(), .ctx = g, .target = g, .deinit = release };
}

/// Allocates lanes up to `total` on both ranks and returns how many there are; notes/glm_native.md#grow
pub fn grow(p: *anyopaque, total: u32) !u32 {
    const g = Glm.of(p);
    g.n_lanes = try g.n.grow(total);
    if (g.n_lanes > 1 and g.drafts) {
        try g.measureShared();
        g.flatten();
    }
    return @intCast(g.n_lanes);
}

/// A lane's state for the pick: whether it is busy, when it was last taken, whether its rows hold the kept state.
pub const View = struct { busy: bool, used: u64, holds: bool };
/// The lane a prompt pass runs in, and the lane whose rows it copies first (null: none).
const Pick = struct { lane: usize, from: ?usize = null };

/// A free lane holding the kept state; else the free lane used longest ago, copying from a busy lane holding it.
pub fn pickLane(v: []const View) ?Pick {
    var oldest: ?usize = null;
    var holder: ?usize = null;
    for (v, 0..) |x, i| {
        if (x.holds) {
            if (!x.busy) return .{ .lane = i };
            holder = i;
        }
        if (!x.busy and (oldest == null or x.used < v[oldest.?].used)) oldest = i;
    }
    return .{ .lane = oldest orelse return null, .from = holder };
}

test "lanes: a free holder first, else the oldest free lane, copying from a busy holder" {
    const t = std.testing;
    try t.expectEqual(Pick{ .lane = 2 }, pickLane(&.{ .{ .busy = false, .used = 1, .holds = false }, .{ .busy = true, .used = 2, .holds = false }, .{ .busy = false, .used = 5, .holds = true } }).?);
    try t.expectEqual(Pick{ .lane = 2, .from = 0 }, pickLane(&.{ .{ .busy = true, .used = 1, .holds = true }, .{ .busy = false, .used = 9, .holds = false }, .{ .busy = false, .used = 3, .holds = false } }).?);
    try t.expectEqual(Pick{ .lane = 0 }, pickLane(&.{ .{ .busy = false, .used = 4, .holds = false }, .{ .busy = false, .used = 7, .holds = false } }).?);
    try t.expectEqual(@as(?Pick, null), pickLane(&.{ .{ .busy = true, .used = 1, .holds = true }, .{ .busy = true, .used = 2, .holds = false } }));
    try t.expectEqual(Pick{ .lane = 0 }, pickLane(&.{.{ .busy = false, .used = 0, .holds = false }}).?); // one lane: as before
}

/// Whether lane `ln`'s attention caches still hold kept state `sn`'s `at + 1` tokens; notes/glm_native.md#live-tokens
pub fn liveHolds(ln: *const Lane, sn: *const Snap) bool {
    return ln.live.items.len > sn.at and std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ln.live.items[0 .. sn.at + 1])) == sn.key;
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OneStreamAtATime => "this build of the GLM CUDA engine runs one request's window at a time",
        error.NoFreeLane => "every lane of the GLM CUDA engine is in use: retry when a request finishes",
        error.NoCapturedKernels => "the EXL3 GLM-5.3-Flash runs only on a captured CUDA kernel set: set TENSORFOLD_CUDA_KERNELS to a folder from aot_pack.py",
        else => null,
    };
}

// -- kept prompt states (prompt_cache.Snapshots, wired by native/cuda.zig) ------------------------------------------

/// A kept state: KDA's state after `at` prompt tokens under `id`, and its tokens' hash; notes/glm_native.md#snapshots
pub const Snap = struct { id: u32, at: u32, key: u64 };

/// prompt_cache.Rules: one lookahead token, states kept at planned chunk starts only; notes/glm_native.md#cache-rules
pub const cache_lookahead: u32 = 1;
pub const cache_planned = true;

pub fn snap_bytes(p: *anyopaque, at: u32) u64 {
    _ = at;
    return Glm.of(p).n.snapshotBytes();
}

pub fn snap_save(p: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const g = Glm.of(p);
    if (g.pass_at != at) return error.SnapshotOutOfStep;
    if (at >= g.pass.len) return error.NoLookaheadToken;
    const sn = try g.gpa.create(Snap);
    errdefer g.gpa.destroy(sn);
    sn.* = .{ .id = g.next_snap, .at = at, .key = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(g.pass[0 .. at + 1])) };
    g.next_snap = if (g.next_snap + 1 == node.no_snapshot) 0 else g.next_snap + 1;
    try g.n.save(sn.id);
    return sn;
}

/// The prompt pass restores a kept state itself (prefillFn), at its first chunk.
pub fn snap_restore(p: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = p;
    _ = owner;
    _ = saved;
    return error.RestoredByThePromptPass;
}

pub fn snap_drop(p: *anyopaque, saved: *anyopaque) void {
    const g = Glm.of(p);
    const sn: *Snap = @ptrCast(@alignCast(saved));
    g.n.drop(sn.id);
    g.gpa.destroy(sn);
}

fn release(p: *anyopaque) void {
    const g = Glm.of(p);
    for (&g.lanes) |*ln| ln.live.deinit(g.gpa);
    g.n.quit();
    g.n.deinit();
    g.gpa.destroy(g);
}
