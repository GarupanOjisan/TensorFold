//! The GLM backend's prompt passes: one prompt at a time (prefillFn) or a round over several (prefillManyFn).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const native = @import("glm_native.zig");
const node = @import("glm_node.zig");
const fwd = @import("glm_forward.zig");
const dflash = @import("glm_dflash.zig");
const glm_sample = @import("glm_native_sample.zig");
const glm_round = @import("glm_native_round.zig");
const glm_timing = @import("glm_native_timing.zig");

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
const now = glm_timing.now;
const ms = glm_timing.ms;

pub fn prefillFn(p: *anyopaque, s: *lanes.Stream) anyerror!void {
    const g = of(p);
    const ids = s.prompt();
    var at = try g.admitPrompt(s);
    const ln = g.laneOf(s).?;
    errdefer ln.stream = null;
    g.pass = ids;
    defer {
        g.pass = &.{};
        g.pass_at = null;
    }
    var next: usize = 0; // the planned chunk starts: a resumed pass cuts where a fresh one does
    while (at < ids.len) {
        if (s.isCancelled()) return error.Cancelled;
        const rows = nextCut(s, at, &next, fwd.Engine.prefill_rows);
        try g.n.chunk(at, rows);
        try ln.live.appendSlice(g.gpa, ids[at..][0..rows]);
        at += rows;
        g.pass_at = @intCast(at);
        if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) != null) if (s.reuse.hook) |hk| hk.at(hk.ptr, s, @intCast(at));
    }
    try g.firstOf(s, ln);
}

/// The rows of the prompt chunk from `at`: to its next planned start (`next` walks s.chunks) or its end, at most `cap`.
pub fn nextCut(s: *const lanes.Stream, at: usize, next: *usize, cap: usize) usize {
    while (next.* < s.chunks.len and s.chunks[next.*] <= at) next.* += 1;
    const end: usize = if (next.* < s.chunks.len) s.chunks[next.*] else s.prompt().len;
    return @min(end - at, cap);
}

/// The first token after the bound lane's prompt.
pub fn firstOf(g: *Glm, s: *lanes.Stream, ln: *Lane) !void {
    const k = try candidateCount(s);
    const first = try g.n.first(k);
    var one: [1]u32 = .{first};
    if (k > 0) try g.choose(s, .last, k, 1, s.prompt().len, &one);
    ln.first = g.take(one[0]);
}

/// A prompt's lane (bound) and its begun state: a kept state of its prefix resumed, else empty; where its pass starts.
pub fn admitPrompt(g: *Glm, s: *lanes.Stream) !usize {
    const e = &g.n.e;
    const ids = s.prompt();
    if (ids.len == 0 or ids.len + s.max_new + max_rows > e.st.capacity) return error.PromptTooLong;
    s.cached = 0;
    // a kept state of this prompt's prefix resumes in a lane whose rows hold it, or in a free lane they are
    // copied to; without one the pass starts at 0 in the free lane used longest ago
    const kept: ?*const Snap = if (s.reuse.saved) |saved| @ptrCast(@alignCast(saved)) else null;
    var views: [max_lanes]View = undefined;
    for (g.lanes[0..g.n_lanes], views[0..g.n_lanes]) |*l, *v| v.* = .{
        .busy = l.stream != null,
        .used = l.used,
        .holds = if (kept) |sn| sn.at < ids.len and liveHolds(l, sn) else false,
    };
    const pick = pickLane(views[0..g.n_lanes]) orelse return error.NoFreeLane;
    const ln = &g.lanes[pick.lane];
    try g.n.bind(pick.lane);
    ln.stream = s;
    errdefer ln.stream = null;
    g.clock += 1;
    ln.used = g.clock;
    // v1.8's copy rule in place of the core's suffix lookup (which TF_GLM_COPY_DRAFTS=0 turns off as well)
    s.proposer = null;
    if (s.drafts and ln.copy.cfg.on) {
        ln.copy.reset(s.prompt_len);
        s.proposer = ln.copy.proposer();
    }
    ln.pending_rows = 0;
    ln.held_n = 0;
    ln.dn = 0;
    var at: usize = 0;
    var snap = node.no_snapshot;
    if (kept) |sn| {
        if (views[pick.lane].holds or pick.from != null) {
            if (pick.from) |src| {
                try g.n.copy(src, pick.lane, sn.at);
                ln.live.clearRetainingCapacity();
                try ln.live.appendSlice(g.gpa, g.lanes[src].live.items[0..sn.at]);
            }
            snap = sn.id;
            at = sn.at;
        } else s.reuse_failed = true;
    }
    ln.live.shrinkRetainingCapacity(@min(ln.live.items.len, at));
    try g.n.begin(ids, g.drafts and !g.dflash, snap, at);
    s.cached = @intCast(at);
    return at;
}

/// One prompt-chunk round over `streams` (Backend.prefill_many; v1.10 multi_prefill with FILL's alternation, the host
/// running a decode round between calls): a stream not on a lane yet is admitted (admitPrompt); then one piece a
/// prompt, its next cut exactly as its own pass takes it (nextCut), the fewest rows left first, while the prompt
/// buffers hold them (each piece at a multiple of 64 rows), in one chunk (Node.chunkLanes; one piece: Node.chunk); a
/// prompt that does not fit waits. The streams whose prompt is in (their first token drawn) or that were cancelled
/// move to the front; their count is returned. With the MTP head each prompt fills whole, alone. While a lane outside
/// `streams` holds a stream (it decodes between these calls) a round takes at most busy_rows rows, so its pause stays
/// short; otherwise the whole prompt buffers.
pub fn prefillManyFn(p: *anyopaque, streams: []*lanes.Stream) anyerror!usize {
    const g = of(p);
    if (g.drafts and !g.dflash) {
        for (streams) |s| try prefillFn(p, s);
        return streams.len;
    }
    if (streams.len > max_lanes) return error.NoFreeLane;
    var lane: [max_lanes]usize = undefined;
    for (streams, 0..) |s, i| {
        if (g.laneOf(s) == null) {
            const at = try g.admitPrompt(s);
            const ln = g.laneOf(s).?;
            ln.fill_at = at;
            ln.fill_next = 0;
        }
        lane[i] = for (g.lanes[0..g.n_lanes], 0..) |*ln, li| (if (ln.stream == s) break li) else unreachable;
    }
    defer {
        g.pass = &.{};
        g.pass_at = null;
    }
    var order: [max_lanes]usize = undefined;
    var m: usize = 0;
    for (streams, 0..) |s, i| if (g.lanes[lane[i]].fill_at < s.prompt().len and !s.isCancelled()) {
        order[m] = i;
        m += 1;
    };
    const Left = struct {
        s: []*lanes.Stream,
        l: []const Lane,
        ln: []const usize,
        fn rest(c: @This(), i: usize) usize {
            return c.s[i].prompt().len - c.l[c.ln[i]].fill_at;
        }
        fn less(c: @This(), a: usize, b: usize) bool {
            return c.rest(a) < c.rest(b);
        }
    };
    std.sort.insertion(usize, order[0..m], Left{ .s = streams, .l = &g.lanes, .ln = lane[0..streams.len] }, Left.less);
    const cap = if (g.othersDecode(lane[0..streams.len])) g.busy_rows else fwd.Engine.prefill_rows;
    var items: [max_lanes][3]usize = undefined;
    var picked: [max_lanes]usize = undefined;
    var n: usize = 0;
    var used: usize = 0;
    for (order[0..m]) |i| {
        const ln = &g.lanes[lane[i]];
        var nx = ln.fill_next;
        const rows = nextCut(streams[i], ln.fill_at, &nx, cap);
        const lo = std.mem.alignForward(usize, used, 64);
        if (lo + rows > cap) continue;
        ln.fill_next = nx;
        items[n] = .{ lane[i], ln.fill_at, rows };
        picked[n] = i;
        n += 1;
        used = lo + rows;
    }
    if (n == 1) {
        try g.n.bind(items[0][0]);
        try g.n.chunk(items[0][1], items[0][2]);
    } else if (n > 1) try g.n.chunkLanes(items[0..n]);
    for (items[0..n], picked[0..n]) |it, i| {
        const s = streams[i];
        const ln = &g.lanes[it[0]];
        try ln.live.appendSlice(g.gpa, s.prompt()[it[1]..][0..it[2]]);
        ln.fill_at += it[2];
        if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(ln.fill_at)) != null) if (s.reuse.hook) |hk| {
            try g.n.bind(it[0]); // snap_save keeps the bound lane's state
            g.pass = s.prompt();
            g.pass_at = @intCast(ln.fill_at);
            hk.at(hk.ptr, s, @intCast(ln.fill_at));
        };
    }
    // the finished ones first (their first token drawn), in order
    var k: usize = 0;
    for (0..streams.len) |i| {
        const s = streams[i];
        const ln = &g.lanes[lane[i]];
        if (!s.isCancelled() and ln.fill_at < s.prompt().len) continue;
        if (!s.isCancelled()) {
            try g.n.bind(lane[i]);
            try g.firstOf(s, ln);
        }
        std.mem.swap(*lanes.Stream, &streams[k], &streams[i]);
        std.mem.swap(usize, &lane[k], &lane[i]);
        k += 1;
    }
    return k;
}
