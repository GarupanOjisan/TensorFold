//! GLM-5.3-Flash on two CUDA ranks behind the native family interface (native/cuda.zig): this process is rank 0
//! and serves; rank 1 runs `tf-glm follow` on the other Spark. One stream at a time, windows of up to 8 rows,
//! MTP drafts. TF_GLM_PORT (default 29660) is the port rank 0 listens on for rank 1.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const node = @import("glm_node.zig");
const fwd = @import("glm_forward.zig");

const be = lanes.backend;
const Node = node.Node;
const max_rows = fwd.Engine.max_rows;

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{"exl3-mcg-4bit"};
pub const default_context: i64 = 1048576;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = fwd.Engine.prefill_rows;
/// One sequence's caches, allocated at load: one stream at a time.
pub const max_lanes: u32 = 1;
pub const default_port: u16 = 29660;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

pub const Loaded = struct {
    backend: be.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
};

/// This rank's share of the checkpoint (the routed experts and most projections halve, the embedding and the
/// replicated layers do not): what the memory admission compares with the budget before loading.
pub fn weightBytes(total: u64) u64 {
    return total / 2 + (4 << 30);
}

const ring = 64;

const Glm = struct {
    gpa: std.mem.Allocator,
    n: *Node,
    drafts: bool,
    stream: ?*lanes.Stream = null,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    held: [max_rows]u32 = undefined,
    held_n: usize = 0,
    pending_rows: usize = 0, // the last verify's rows, until a keep or a draft commits them
    costs: [max_rows]lanes.config.Cost = undefined,
    mtp_ms: f64 = 1.0,
    vtable: be.Backend.VTable = undefined,

    fn of(p: *anyopaque) *Glm {
        return @ptrCast(@alignCast(p));
    }

    fn take(g: *Glm, t: u32) u64 {
        g.drawn[g.next % ring] = t;
        g.next += 1;
        return g.next - 1;
    }

    fn value(g: *Glm, f: be.Feed) !u32 {
        return switch (f) {
            .value => |v| v,
            .handle => |h| try readFn(g, h),
        };
    }

    /// Commit the last verify's first `keep` rows once (keep and the next draft both name them).
    fn commit(g: *Glm, keep: usize) !void {
        if (g.pending_rows == 0) return;
        try g.n.keep(g.pending_rows, keep);
        g.pending_rows = 0;
    }

    fn prefillFn(p: *anyopaque, s: *lanes.Stream) anyerror!void {
        const g = of(p);
        const e = &g.n.e;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + max_rows > e.st.capacity) return error.PromptTooLong;
        if (s.sampling) |sm| if (sm.temperature > 0) return error.SamplingNotPorted;
        g.stream = s;
        g.pending_rows = 0;
        g.held_n = 0;
        _ = g.take(try g.n.prefill(ids, g.drafts));
    }

    fn firstFn(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const g = of(p);
        if (position != s.prompt_len) return error.PositionMismatch;
        return g.next - 1;
    }

    fn queueFn(p: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = p;
        _ = s;
        _ = feed;
        _ = position;
        return error.NotPipelined;
    }

    fn readFn(p: *anyopaque, handle: u64) anyerror!u32 {
        const g = of(p);
        if (handle >= g.next or g.next - handle > ring) return error.NoSuchToken;
        return g.drawn[handle % ring];
    }

    /// One window: the pending token, the drafts held here, then host drafts; every row's greedy token.
    fn verifyFn(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const g = of(p);
        if (windows.len != 1) return error.OneStreamAtATime;
        const w = windows[0];
        if (w.parents != null) return error.TreesNotBuilt;
        if (g.pending_rows != 0) return error.UnkeptWindow;
        const rows = w.rows();
        if (rows > max_rows) return error.WindowTooWide;
        if (w.held > g.held_n) return error.NoHeldDrafts;
        const e = &g.n.e;
        for (w.positions, 0..) |pos, r| if (pos != e.st.pos + 1 + r) return error.PositionMismatch;
        var ids: [max_rows]u32 = undefined;
        ids[0] = w.pending;
        @memcpy(ids[1..][0..w.held], g.held[0..w.held]);
        @memcpy(ids[1 + w.held ..][0..w.tokens.len], w.tokens);
        try g.n.verify(ids[0..rows], out[0].sampled[0..rows]);
        @memcpy(out[0].drafts[0 .. rows - 1], ids[1..rows]);
        g.pending_rows = rows;
        g.held_n = 0;
    }

    fn keepFn(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const g = of(p);
        for (paths) |path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            try g.commit(path.len);
        }
        _ = windows;
    }

    /// Absorb the kept rows with their following tokens, then hold `depth` drafts for the next window.
    fn draftFn(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const g = of(p);
        for (requests) |r| {
            if (r.depth > max_rows - 1) return error.DepthPastWindow;
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                try g.commit(rows.len);
                if (r.position != g.n.e.st.pos + 1) return error.PositionMismatch;
                try g.n.draft(.verify, r.follow[0..rows.len], g.held[0..r.depth]);
            } else {
                if (r.position != g.n.e.st.pos + 1) return error.PositionMismatch;
                const first = if (r.follow.len > 0) r.follow[0] else try g.value(r.first orelse return error.NoFirstToken);
                try g.n.draft(.prompt, &.{first}, g.held[0..r.depth]);
            }
            g.held_n = r.depth;
        }
    }

    fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
        const g = of(p);
        if (g.stream == s) g.stream = null;
        g.pending_rows = 0;
        g.held_n = 0;
    }

    fn backend(g: *Glm) be.Backend {
        g.vtable = .{ .prefill = prefillFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn };
        return .{ .ptr = g, .vtable = &g.vtable };
    }

    fn facts(g: *const Glm) lanes.Model {
        if (!g.drafts) return .{ .exact_width = 1, .max_streams = 1 };
        return .{
            .exact_width = max_rows,
            .mtp = true,
            .speculate = true,
            .speculate_early = false,
            .drafts = max_rows - 1,
            .window_costs = &g.costs,
            .mtp_step_ms = g.mtp_ms,
            .max_streams = 1,
        };
    }

    /// Each window width's forward and one chained MTP step, timed on a short dummy prompt (both ranks run it).
    fn measure(g: *Glm, io: std.Io) !void {
        var prompt: [64]u32 = undefined;
        for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + i);
        _ = try g.n.prefill(&prompt, true);
        var out: [max_rows]u32 = undefined;
        var ids: [max_rows]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast(2000 + i);
        for (1..max_rows + 1) |width| {
            var best: f64 = std.math.inf(f64);
            for (0..3) |_| {
                const t0 = now(io);
                try g.n.verify(ids[0..width], out[0..width]);
                best = @min(best, ms(io, t0));
                try g.n.keep(width, 1);
            }
            g.costs[width - 1] = .{ .width = @intCast(width), .ms = best };
        }
        var one: f64 = std.math.inf(f64);
        var three: f64 = std.math.inf(f64);
        for (0..3) |_| {
            var t0 = now(io);
            try g.n.draft(.verify, out[0..1], ids[0..1]);
            one = @min(one, ms(io, t0));
            t0 = now(io);
            try g.n.draft(.verify, out[0..1], ids[0..3]);
            three = @min(three, ms(io, t0));
        }
        g.mtp_ms = @max(0.1, (three - one) / 2);
        std.log.info("GLM window ms by width: {d:.1} {d:.1} {d:.1} {d:.1} {d:.1} {d:.1} {d:.1} {d:.1}; MTP step {d:.2} ms", .{ g.costs[0].ms, g.costs[1].ms, g.costs[2].ms, g.costs[3].ms, g.costs[4].ms, g.costs[5].ms, g.costs[6].ms, g.costs[7].ms, g.mtp_ms });
    }
};

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn ms(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e6;
}

fn envPort() u16 {
    const v = std.c.getenv("TF_GLM_PORT") orelse return default_port;
    return std.fmt.parseInt(u16, std.mem.span(v), 10) catch default_port;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    std.log.info("GLM-5.3-Flash rank 0: waiting for rank 1 (tf-glm follow) on port {d}", .{envPort()});
    const n = try Node.init(gpa, io, ctx, dir, kernels, o.context, 0, "", envPort());
    errdefer n.deinit();
    const g = try gpa.create(Glm);
    errdefer gpa.destroy(g);
    g.* = .{ .gpa = gpa, .n = n, .drafts = o.drafts and n.w.mtp != null };
    try g.measure(io);
    return .{ .backend = g.backend(), .facts = g.facts(), .rows = if (g.drafts) max_rows else 1, .stream_bytes = 0, .ctx = g, .deinit = release };
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.SamplingNotPorted => "this build of the GLM CUDA engine decodes greedily only: send temperature 0",
        error.OneStreamAtATime => "this build of the GLM CUDA engine serves one request at a time",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const g = Glm.of(p);
    g.n.quit();
    g.n.deinit();
    g.gpa.destroy(g);
}
