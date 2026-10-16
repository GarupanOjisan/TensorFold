//! The GLM backend's decode rounds: the first token, verify windows, keeps, drafts and the stream's release.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const native = @import("glm_native.zig");
const node = @import("glm_node.zig");
const fwd = @import("glm_forward.zig");
const dflash = @import("glm_dflash.zig");
const glm_sample = @import("glm_native_sample.zig");
const glm_prompt = @import("glm_native_prompt.zig");
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

pub fn firstFn(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
    const g = of(p);
    if (position != s.prompt_len) return error.PositionMismatch;
    return (g.laneOf(s) orelse return error.NoLane).first;
}

pub fn queueFn(p: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
    _ = p;
    _ = s;
    _ = feed;
    _ = position;
    return error.NotPipelined;
}

pub fn readFn(p: *anyopaque, handle: u64) anyerror!u32 {
    const g = of(p);
    if (handle >= g.next or g.next - handle > ring) return error.NoSuchToken;
    return g.drawn[handle % ring];
}

/// One window: the pending token, the drafts held here, then host drafts; every row's greedy token.
pub fn verifyFn(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
    const g = of(p);
    if (windows.len > 1) return g.verifyShared(windows, out);
    if (windows.len != 1) return error.OneStreamAtATime;
    const w = windows[0];
    if (w.parents != null) return error.TreesNotBuilt;
    const ln = try g.enter(w.stream);
    try g.commit(ln, ln.pending_rows); // no keep since the last window: all of its rows stay (Nemotron's bindLane)
    const rows = w.rows();
    if (rows > max_rows) return error.WindowTooWide;
    if (w.held > ln.held_n) return error.NoHeldDrafts;
    const e = &g.n.e;
    for (w.positions, 0..) |pos, r| if (pos != e.st.pos + 1 + r) return error.PositionMismatch;
    var ids: [max_rows]u32 = undefined;
    ids[0] = w.pending;
    @memcpy(ids[1..][0..w.held], ln.held[0..w.held]);
    @memcpy(ids[1 + w.held ..][0..w.tokens.len], w.tokens);
    const k = try candidateCount(w.stream);
    try g.n.verify(ids[0..rows], k, out[0].sampled[0..rows]);
    if (k > 0) try g.choose(w.stream, .window, k, rows, w.positions[0], out[0].sampled[0..rows]);
    @memcpy(out[0].drafts[0 .. rows - 1], ids[1..rows]);
    @memcpy(ln.window[0..rows], ids[0..rows]);
    ln.pending_rows = rows;
    ln.held_n = 0;
}

/// Several streams' windows (a shared round) in one forward, each lane's rows back to back in the windows' order.
pub fn verifyShared(g: *Glm, windows: []const be.Window, out: []be.Verified) !void {
    if (windows.len > max_lanes) return error.TooManyWindows;
    var segs: [max_lanes]fwd.Engine.Seg = undefined;
    var ks: [max_lanes]usize = undefined;
    var ids: [fwd.Engine.shared_rows]u32 = undefined;
    var total: usize = 0;
    for (windows, 0..) |w, i| {
        if (w.parents != null) return error.TreesNotBuilt;
        const ln = try g.enter(w.stream);
        try g.commit(ln, ln.pending_rows); // no keep since its last window: all of its rows stay
        const rows = w.rows();
        if (total + rows > g.sharedRows()) return error.WindowTooWide;
        if (w.held > ln.held_n) return error.NoHeldDrafts;
        for (w.positions, 0..) |pos, r| if (pos != g.n.e.st.pos + 1 + r) return error.PositionMismatch;
        ids[total] = w.pending;
        @memcpy(ids[total + 1 ..][0..w.held], ln.held[0..w.held]);
        @memcpy(ids[total + 1 + w.held ..][0..w.tokens.len], w.tokens);
        segs[i] = .{ .lane = g.n.e.lane, .rows = rows };
        ks[i] = try candidateCount(w.stream);
        total += rows;
    }
    const n = windows.len;
    try g.n.verifyShared(segs[0..n], ids[0..total]);
    // one pick over every row when the lanes draw alike (greedy, or the same top-k), as v1.10's packed all-gather; a
    // whole-vocabulary draw (top_k off) reads the bound lane's rows, so it picks lane by lane
    var one = n > 1 and ks[0] != Node.whole_vocabulary;
    for (ks[1..n]) |k| one = one and k == ks[0];
    var all: [fwd.Engine.shared_rows]u32 = undefined;
    if (one) try g.n.pickRows(total, ks[0], all[0..total]);
    var at: usize = 0;
    for (windows, out, segs[0..n], ks[0..n]) |w, *o, sg, k| {
        const rows = sg.rows;
        if (one) {
            if (k > 0) try g.chooseIn(w.stream.sampling.?, k, total, at, rows, w.positions[0], o.sampled[0..rows]) else @memcpy(o.sampled[0..rows], all[at..][0..rows]);
        } else {
            try g.n.bind(sg.lane); // a draw's `nucleus` reads the bound lane's rows
            try g.n.pickSegment(sg, k, o.sampled[0..rows]);
            if (k > 0) try g.choose(w.stream, .window, k, rows, w.positions[0], o.sampled[0..rows]);
        }
        @memcpy(o.drafts[0 .. rows - 1], ids[at + 1 ..][0 .. rows - 1]);
        const ln = &g.lanes[sg.lane];
        @memcpy(ln.window[0..rows], ids[at..][0..rows]);
        ln.pending_rows = rows;
        ln.held_n = 0;
        at += rows;
    }
}

pub fn keepFn(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
    const g = of(p);
    if (paths.len != windows.len) return error.PathsWithoutWindows;
    for (paths, windows) |path, w| {
        if (path.len == 0) return error.EmptyPath;
        for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        try g.commit(try g.enter(w.stream), path.len);
    }
}

/// DFlash2 in place of the MTP head (v1.8's dflash_decode): the kept rows' taps join its context, then, unless a copy is
/// ahead (v1.8 takes the copy first and runs no block), its block pass and chain (its own stop rule, not `r.depth`, up to
/// TF_GLM_DFLASH_POLICY's most) give the next window's drafts as host tokens (`treeFn`). The first round after a prompt
/// is taken as `r.depth` held drafts (the core does not ask for the chain there), filled up with the pending token.
pub fn draftDflash(g: *Glm, ln: *Lane, r: be.DraftRequest) !void {
    const t = try g.dflashTaps(ln, r, true, true);
    g.dflashChain(ln, r, t, if (t.want > 0) &g.n.dflash.? else null);
}

/// A DFlash2 turn's first half: the kept rows committed and joined to the lane's context, and how many drafts it wants.
pub const Turn = struct { want: usize, pending: u32, kept: usize = 0 };

/// The bound lane's DFlash2 turn up to its chain: commit, the context update and, with `block`, its own block pass.
/// `turn` false: no DFlash2 turn here (the caller adds the kept rows to the context, Node.dflashTapsLanes).
pub fn dflashTaps(g: *Glm, ln: *Lane, r: be.DraftRequest, block: bool, turn: bool) !Turn {
    const s = r.stream;
    var from: node.From = .prompt;
    var keep: usize = 0;
    var pending: u32 = undefined;
    if (r.rows) |rows| {
        for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
        try g.commit(ln, rows.len);
        if (r.position != g.n.e.st.pos + 1) return error.PositionMismatch;
        from = .verify;
        keep = rows.len;
        pending = r.follow[rows.len - 1];
    } else {
        if (r.position != g.n.e.st.pos + 1) return error.PositionMismatch;
        pending = if (r.follow.len > 0) r.follow[0] else try g.value(r.first orelse return error.NoFirstToken);
    }
    const room: i64 = s.draftRoom() - 1;
    var want: usize = if (room > 0) @min(g.policy.most, @as(usize, @intCast(room))) else 0;
    if (r.depth == 0) want = 0; // forced rows ahead: absorb only
    if (want > 0) if (s.proposer) |pr| {
        const width: i64 = @min(@as(i64, @intCast(g.copy_cfg.most)), room);
        const found = pr.propose(s.context.items, width) catch @as([]const u32, &.{});
        if (found.len >= 2 and pr.lastMatch() >= 8) want = 0; // the core's own copy test
    };
    if (turn) try g.n.dflashTurn(from, keep, pending, block and want > 0);
    return .{ .want = want, .pending = pending, .kept = keep };
}

/// A DFlash2 turn's second half: the chain from `dft`'s candidates (null: no block pass ran, no drafts), held.
pub fn dflashChain(g: *Glm, ln: *Lane, r: be.DraftRequest, t: Turn, dft: ?*const dflash.Drafter) void {
    const s = r.stream;
    ln.dn = if (dft) |d| d.chain(t.want, t.pending, r.position, if (s.sampling) |sm| .{ .seed = sm.seed, .temperature = sm.temperature } else null, g.policy, &ln.dtoks) else 0;
    const held = @max(ln.dn, r.depth);
    @memcpy(ln.held[0..ln.dn], ln.dtoks[0..ln.dn]);
    for (ln.dn..held) |i| ln.held[i] = t.pending;
    ln.held_n = held;
}

/// A shared round's DFlash2 turns (facts.draft_streams): each lane's context update on its own, then the block passes of
/// every lane that wants drafts in one (Node.dflashBlocks; one lane alone takes its own captured pass), then the chains.
pub fn draftDflashLanes(g: *Glm, requests: []const be.DraftRequest) !void {
    if (requests.len > max_lanes) return error.TooManyWindows;
    var turns: [max_lanes]Turn = undefined;
    var idx: [max_lanes]usize = undefined;
    var blocked: [max_lanes]bool = @splat(false);
    var ls: [max_lanes]usize = undefined;
    var pend: [max_lanes]u32 = undefined;
    var m: usize = 0;
    // every lane's kept rows in one commit (Node.keepLanes), before the turns: each turn's own commit then finds nothing
    var keeps: [max_lanes]fwd.Engine.Keep = undefined;
    var nk: usize = 0;
    for (requests) |r| {
        const rows = r.rows orelse continue;
        for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
        for (g.lanes[0..g.n_lanes], 0..) |*ln, li| if (ln.stream == r.stream and ln.pending_rows > 0) {
            keeps[nk] = .{ .lane = li, .rows = ln.pending_rows, .keep = rows.len };
            nk += 1;
        };
    }
    if (nk > 0) {
        try g.n.keepLanes(keeps[0..nk]);
        for (keeps[0..nk]) |k| {
            const ln = &g.lanes[k.lane];
            try ln.live.appendSlice(g.gpa, ln.window[0..k.keep]);
            ln.pending_rows = 0;
        }
    }
    // every lane's kept rows into its drafter's context in one pass (Node.dflashTapsLanes), then the block passes
    var taps: [max_lanes][2]usize = undefined;
    var nt: usize = 0;
    for (requests, 0..) |r, i| {
        const ln = try g.enter(r.stream);
        idx[i] = g.n.e.lane;
        turns[i] = try g.dflashTaps(ln, r, false, false);
        if (turns[i].kept > 0) {
            taps[nt] = .{ idx[i], turns[i].kept };
            nt += 1;
        }
    }
    if (nt > 0) try g.n.dflashTapsLanes(taps[0..nt]);
    for (0..requests.len) |i| {
        if (turns[i].want > 0 and (try g.n.drafterOf(idx[i])).context_end > 0) {
            blocked[i] = true;
            ls[m] = idx[i];
            pend[m] = turns[i].pending;
            m += 1;
        }
    }
    if (m == 1) {
        try g.n.bind(ls[0]);
        try g.n.dflashTurn(.prompt, 0, pend[0], true);
    } else if (m > 1) try g.n.dflashBlocks(ls[0..m], pend[0..m]);
    for (requests, 0..) |r, i| g.dflashChain(&g.lanes[idx[i]], r, turns[i], if (blocked[i]) try g.n.drafterOf(idx[i]) else null);
}

pub fn treeFn(p: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
    const g = of(p);
    if (!g.dflash) return null;
    const ln = g.laneOf(s) orelse return error.NoLane;
    return .{ .count = @intCast(ln.dn), .tokens = try gpa.dupe(u32, ln.dtoks[0..ln.dn]) };
}

/// Absorb the kept rows with their following tokens, then hold `depth` drafts for the next window.
pub fn draftFn(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const g = of(p);
    if (g.dflash and requests.len > 1 and g.n.batch != null) return g.draftDflashLanes(requests);
    for (requests) |r| {
        const ln = try g.enter(r.stream);
        if (g.dflash) {
            try g.draftDflash(ln, r);
            continue;
        }
        if (r.depth > max_rows - 1) return error.DepthPastWindow;
        if (r.rows) |rows| {
            for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
            try g.commit(ln, rows.len);
            if (r.position != g.n.e.st.pos + 1) return error.PositionMismatch;
            try g.n.draft(.verify, r.follow[0..rows.len], ln.held[0..r.depth]);
        } else {
            if (r.position != g.n.e.st.pos + 1) return error.PositionMismatch;
            const first = if (r.follow.len > 0) r.follow[0] else try g.value(r.first orelse return error.NoFirstToken);
            try g.n.draft(.prompt, &.{first}, ln.held[0..r.depth]);
        }
        ln.held_n = r.depth;
    }
}

/// The stream leaves its lane; the lane's caches and live tokens stay for a later turn of its conversation.
pub fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
    const g = of(p);
    const ln = g.laneOf(s) orelse return;
    ln.stream = null;
    ln.pending_rows = 0;
    ln.held_n = 0;
    ln.dn = 0;
}
