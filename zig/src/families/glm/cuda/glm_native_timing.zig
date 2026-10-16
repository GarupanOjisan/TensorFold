//! The GLM backend's start-up timings: window widths, MTP and DFlash2 steps, and shared windows.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const native = @import("glm_native.zig");
const node = @import("glm_node.zig");
const fwd = @import("glm_forward.zig");
const dflash = @import("glm_dflash.zig");
const glm_sample = @import("glm_native_sample.zig");
const glm_prompt = @import("glm_native_prompt.zig");
const glm_round = @import("glm_native_round.zig");

const be = lanes.backend;
const Node = node.Node;
const Glm = native.Glm;
const Lane = native.Lane;
const Snap = native.Snap;
const View = native.View;
const of = Glm.of;
const max_rows = native.max_rows;
const max_lanes = native.max_lanes;
const ring = native.ring;
const dflash_max = native.dflash_max;
const nucleus_wider = native.nucleus_wider;
const pickLane = native.pickLane;
const liveHolds = native.liveHolds;
const candidateCount = glm_sample.candidateCount;

/// v1.10 verify.timing_tokens(vocab, 32): numpy default_rng(1234).choice(150000, 32, replace=False), the distinct ids its
/// startup timings run (one token repeated, or a run of neighbours, routes to a few experts and times light).
pub const timing_ids = [_]u32{ 47774, 77768, 138462, 36260, 39247, 129560, 129538, 101230, 47857, 119011, 91475, 96893, 143736, 144598, 39544, 20813, 80395, 148162, 119363, 146872, 117894, 57018, 98981, 17711, 66146, 83336, 25699, 134558, 15731, 143261, 38205, 146475 };

/// Shared windows of 2 to max_rows rows over lanes 0 and 1 (half each), timed on whatever their caches hold (both
/// ranks run them; no commit, so the lanes stay where they were). Serving overwrites the rows they wrote.
pub fn measureShared(g: *Glm) !void {
    const ids = timing_ids;
    var out: [max_rows]u32 = undefined;
    g.shared[0] = g.costs[0];
    for (2..g.sharedRows() + 1) |total| {
        const segs = [2]fwd.Engine.Seg{ .{ .lane = 0, .rows = (total + 1) / 2 }, .{ .lane = 1, .rows = total / 2 } };
        var best: f64 = std.math.inf(f64);
        for (0..3) |_| {
            const t0 = now(g.io);
            try g.n.verifyShared(&segs, ids[0..total]);
            for (segs) |sg| try g.n.pickSegment(sg, 0, out[0..sg.rows]);
            best = @min(best, ms(g.io, t0));
        }
        g.shared[total - 1] = .{ .width = @intCast(total), .ms = best };
    }
    std.log.info("GLM shared window ms by rows (two lanes): 2 {d:.1}, 4 {d:.1}, 8 {d:.1}, 16 {d:.1}, {d} {d:.1}", .{ g.shared[1].ms, g.shared[3].ms, g.shared[7].ms, g.shared[15].ms, g.sharedRows(), g.shared[g.sharedRows() - 1].ms });
    try g.compareOne(8);
}

/// v1.10 multi._compare_paths: one lane's window of `rows` rows on the one-stream graphs (verify) and on the shared ones
/// (verifyShared over a single segment), the same tokens, timed by GPU events (fastest of 5, after a warm-up), each
/// with its greedy pick; logged only.
pub fn compareOne(g: *Glm, rows: usize) !void {
    const e = &g.n.e;
    try g.n.bind(0);
    var start = try cuda.Event.init(e.d, true);
    defer start.deinit();
    var stop = try cuda.Event.init(e.d, true);
    defer stop.deinit();
    var out: [max_rows]u32 = undefined;
    var solo: f32 = std.math.inf(f32);
    var shared: f32 = std.math.inf(f32);
    const seg = [1]fwd.Engine.Seg{.{ .lane = 0, .rows = rows }};
    for (0..6) |rep| {
        try start.record(e.s);
        try g.n.verify(timing_ids[0..rows], 0, out[0..rows]);
        try stop.record(e.s);
        try stop.synchronize();
        if (rep > 0) solo = @min(solo, try cuda.Event.elapsedMs(start, stop));
        try g.n.keep(rows, 1);
        try start.record(e.s);
        try g.n.verifyShared(&seg, timing_ids[0..rows]);
        try g.n.pickSegment(seg[0], 0, out[0..rows]);
        try stop.record(e.s);
        try stop.synchronize();
        if (rep > 0) shared = @min(shared, try cuda.Event.elapsedMs(start, stop));
    }
    std.log.info("GLM one lane, {d} rows (v1.10 _compare_paths; forward and greedy pick, GPU events): one-stream graphs {d:.1} ms, shared (segmented) {d:.1} ms", .{ rows, solo, shared });
}

/// TF_GLM_SHARED_FLAT (default on; 0 off): every shared width costs as 2 rows, so the core keeps every draft in a
/// shared round, as v1.10 does (its multi-stream rounds take each stream's drafts with no depth policy): +1.7% at
/// four streams, the same replies.
pub fn flatten(g: *Glm) void {
    if (std.c.getenv("TF_GLM_SHARED_FLAT")) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) return;
    for (g.shared[1..g.sharedRows()]) |*c| c.ms = g.shared[1].ms;
    std.log.info("GLM shared rounds keep every draft (TF_GLM_SHARED_FLAT; costs flat at {d:.1} ms)", .{g.shared[1].ms});
}

/// A shared round's rows at most: fwd.Engine.shared_rows over the segmented kernels, else the lanes' own max_rows.
pub fn sharedRows(g: *const Glm) usize {
    return if (g.n.e.seg_enabled) fwd.Engine.shared_rows else max_rows;
}

/// Each window width's forward and one chained MTP step, timed on a short dummy prompt (both ranks run it).
pub fn measure(g: *Glm, io: std.Io) !void {
    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + i);
    _ = try g.n.prefill(&prompt, !g.dflash, 0);
    var out: [max_rows]u32 = undefined;
    var ids: [max_rows]u32 = timing_ids[0..max_rows].*;
    for (1..max_rows + 1) |width| {
        var best: f64 = std.math.inf(f64);
        for (0..3) |_| {
            const t0 = now(io);
            try g.n.verify(ids[0..width], 0, out[0..width]);
            best = @min(best, ms(io, t0));
            try g.n.keep(width, 1);
        }
        g.costs[width - 1] = .{ .width = @intCast(width), .ms = best };
    }
    var one: f64 = std.math.inf(f64);
    var three: f64 = std.math.inf(f64);
    if (!g.dflash) for (0..3) |_| {
        var t0 = now(io);
        try g.n.draft(.verify, out[0..1], ids[0..1]);
        one = @min(one, ms(io, t0));
        t0 = now(io);
        try g.n.draft(.verify, out[0..1], ids[0..3]);
        three = @min(three, ms(io, t0));
    };
    g.mtp_ms = if (g.dflash) 0.1 else @max(0.1, (three - one) / 2);
    if (g.dflash) {
        // DFlash2's pieces (Engine._calibrate's "block" and "taps8"): the block pass with its candidates read back, the
        // chain on the host, and a context update of the last window's first 8 rows
        var blk: f64 = std.math.inf(f64);
        var chn: f64 = std.math.inf(f64);
        var tap: f64 = std.math.inf(f64);
        for (0..7) |_| {
            var t0 = now(io);
            try g.n.dflashTurn(.prompt, 0, 2000, true);
            blk = @min(blk, ms(io, t0));
            t0 = now(io);
            var dtoks: [dflash_max]u32 = undefined;
            _ = g.n.dflash.?.chain(7, 2000, 100, null, g.policy, &dtoks);
            chn = @min(chn, ms(io, t0));
            t0 = now(io);
            try g.n.dflashTurn(.verify, 8, 2000, false);
            tap = @min(tap, ms(io, t0));
        }
        std.log.info("GLM DFlash2 ms: block {d:.2}, chain {d:.2}, context update of 8 rows {d:.2}", .{ blk, chn, tap });
    }
    var line: [max_rows * 8]u8 = undefined;
    var w = std.Io.Writer.fixed(&line);
    for (g.costs) |c| w.print(" {d:.1}", .{c.ms}) catch {};
    std.log.info("GLM window ms by width:{s}; MTP step {d:.2} ms", .{ w.buffered(), g.mtp_ms });
}

pub fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

pub fn ms(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e6;
}
