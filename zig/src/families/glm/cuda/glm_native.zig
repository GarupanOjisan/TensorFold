//! GLM-5.3-Flash on two CUDA ranks behind the native family interface (native/cuda.zig): this process is rank 0
//! and serves; rank 1 runs `tf-glm follow` on the other Spark. A stream a lane (each lane one sequence's caches and
//! DFlash2 context), the lanes' rounds taking turns; an idle lane keeps its caches' rows, so a later turn of its
//! conversation resumes there. Windows of up to 16 rows, MTP drafts or DFlash2's (TF_GLM_DRAFTER_DIR; glm_dflash.zig)
//! and v1.8's prompt-lookup (copy) drafts (glm_copy.zig). TF_GLM_PORT (default 29660) is the port rank 0 listens on for rank 1.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const node = @import("glm_node.zig");
const fwd = @import("glm_forward.zig");
const copy = @import("glm_copy.zig");
const dflash = @import("glm_dflash.zig");

const be = lanes.backend;
const Node = node.Node;
const max_rows = fwd.Engine.max_rows;
/// Drafts the MTP head chains at most: the 8-row window of earlier builds (the rows past it serve copies).
const head_drafts = 7;
const dflash_max = 15;
/// sampling.WIDER: the candidates a rank sends when a row's nucleus runs past the first (union rule); then the whole shards.
const nucleus_wider = fwd.Engine.max_nucleus;

const NucEnt = struct {
    v: f64,
    id: i64,
    m: i64,

    fn before(_: void, a: NucEnt, b: NucEnt) bool {
        return a.v > b.v or (a.v == b.v and a.id < b.id);
    }
};

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{"exl3-mcg-4bit"};
pub const default_context: i64 = 1048576;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = fwd.Engine.prefill_rows;
pub const promptGrid = fwd.promptGrid;
/// Sequences' caches allocated at load (`grow`), a stream each.
pub const max_lanes: u32 = fwd.Engine.max_lanes;
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
    return total / 2 + (4 << 30) + if (node.drafterDir() != null) @as(u64, 2 << 30) else 0; // DFlash2's 4-bit layers: 1.8 GiB a rank
}

const ring = 64;

/// glm_copy.zig as the lane core's proposer (lanes/proposer.zig): one a lane, reset by each prompt pass there. The core asks twice a round with the same context (the head's depth rule, then the next plan): the second
/// answer is the first.
const CopyRule = struct {
    cfg: copy.Settings,
    prompt: usize = 0,
    buf: [copy.max_drafts]u32 = undefined,
    memo_len: usize = std.math.maxInt(usize),
    memo_room: i64 = 0,
    memo_n: usize = 0,

    const vtable: lanes.proposer.Proposer.VTable = .{ .propose = proposeFn, .last_match = lastMatchFn };

    fn reset(r: *CopyRule, prompt: usize) void {
        r.prompt = prompt;
        r.memo_len = std.math.maxInt(usize);
    }

    fn proposer(r: *CopyRule) lanes.proposer.Proposer {
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
const Lane = struct {
    stream: ?*lanes.Stream = null,
    copy: CopyRule,
    dtoks: [dflash_max]u32 = undefined, // the chain DFlash2 picked for the stream's next window
    dn: usize = 0,
    held: [max_rows]u32 = undefined,
    held_n: usize = 0,
    pending_rows: usize = 0, // the last verify's rows, until a keep or a draft commits them
    window: [max_rows]u32 = undefined, // the last verify's rows, the tokens a commit makes live
    /// The tokens whose attention-cache rows are committed, from position 0: a kept state resumes only while the
    /// rows below its position still hold its tokens (the caches are positional; only KDA's state is copied).
    live: std.ArrayList(u32) = .empty,
    first: u64 = 0, // the handle of the stream's first token
    used: u64 = 0, // when a stream last took the lane (least recently used goes first)
};

const Glm = struct {
    gpa: std.mem.Allocator,
    n: *Node,
    drafts: bool,
    dflash: bool = false, // DFlash2 drafts (the MTP head is not loaded)
    policy: dflash.Policy = .{},
    nucleus_first: usize = 1024, // TF_GLM_NUCLEUS: the candidates a rank sends first for a top_k-off draw (sampling.NUCLEUS)
    nucleus_union: bool = true, // TENSORFOLD_NUCLEUS_UNION
    copy_cfg: copy.Settings,
    lanes: [max_lanes]Lane = undefined,
    n_lanes: usize = 1,
    clock: u64 = 0,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    costs: [max_rows]lanes.config.Cost = undefined,
    /// Shared windows' ms by total rows: the lone windows' until `grow` times them over two lanes (the facts point here,
    /// and the lane core reads them after `grow`).
    shared: [max_rows]lanes.config.Cost = undefined,
    io: std.Io = undefined,
    mtp_ms: f64 = 1.0,
    vtable: be.Backend.VTable = undefined,
    pass: []const u32 = &.{}, // the prompt a prompt pass is running, and where it stands between chunks
    pass_at: ?u32 = null,
    next_snap: u32 = 0,

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

    /// The lane stream `s` uses, bound on both ranks.
    fn enter(g: *Glm, s: *const lanes.Stream) !*Lane {
        for (g.lanes[0..g.n_lanes], 0..) |*ln, i| if (ln.stream == s) {
            try g.n.bind(i);
            return ln;
        };
        return error.NoLane;
    }

    fn laneOf(g: *Glm, s: *const lanes.Stream) ?*Lane {
        for (g.lanes[0..g.n_lanes]) |*ln| if (ln.stream == s) return ln;
        return null;
    }

    /// Commit the bound lane's last verify's first `keep` rows once (keep and the next draft both name them).
    fn commit(g: *Glm, ln: *Lane, keep: usize) !void {
        if (ln.pending_rows == 0) return;
        try g.n.keep(ln.pending_rows, keep);
        try ln.live.appendSlice(g.gpa, ln.window[0..keep]);
        ln.pending_rows = 0;
    }

    /// Each row's draw over the vocabulary (Node.whole_vocabulary): top_k off: sampling.nucleus_rows, its candidates a rank first
    /// (`nucleus_first`, then `nucleus_wider`: TENSORFOLD_NUCLEUS_UNION, v1.10 patch 0034's rule), whole shards when a row's nucleus
    /// runs past them; else choose_rows over every token (a top_k past the candidates a rank gives). Row r keyed at position0 + r.
    fn chooseWhole(g: *Glm, sm: lanes.sampling.Sampling, which: Node.Rows, rows: usize, position0: u64, out: []u32) !void {
        if (sm.top_k == 0) {
            const stages = [_]usize{ g.nucleus_first, nucleus_wider };
            // without the union rule (stock v1.8) a row whose nucleus is not inside one rank's candidates goes straight to the shards
            for (stages[0 .. if (g.nucleus_union) 2 else 1]) |count| {
                const words = try g.n.nucleus(which, rows, sm.temperature, count);
                if (try g.drawNucleus(sm, rows, position0, words, count, out)) return;
            }
        }
        _ = try g.n.nucleus(which, rows, sm.temperature, 0);
        const host = g.n.e.full_host;
        const n = g.n.e.w.head.n;
        const row = try g.gpa.alloc(u16, 2 * n);
        defer g.gpa.free(row);
        for (0..rows) |r| {
            for (0..2) |rank| @memcpy(row[rank * n ..][0..n], host[(rank * rows + r) * n ..][0..n]);
            if (sm.top_k == 0) {
                out[r] = try nucleus(g.gpa, row, position0 + r, sm);
                continue;
            }
            const values = try g.gpa.alloc(f64, 2 * n);
            defer g.gpa.free(values);
            const ids = try g.gpa.alloc(u64, 2 * n);
            defer g.gpa.free(ids);
            for (row, values, ids, 0..) |b, *v, *id, i| {
                v.* = bf16Value(b);
                id.* = i;
            }
            out[r] = @intCast(try lanes.sampling.choose(g.gpa, values, ids, position0 + r, sm));
        }
    }

    /// sampling._draw over the ranks' `count` best candidates (`words`, Engine.nucleusGather's layout): false when some row's nucleus
    /// is not inside them (a wider gather or the whole shards decide it). With the union rule (TENSORFOLD_NUCLEUS_UNION, default
    /// on) a row is covered when the merged walk to the cumulative mass ceil(top_p total) stops above every rank's last candidate:
    /// no token left out can be in the nucleus, so the draw is the whole vocabulary's.
    fn drawNucleus(g: *Glm, sm: lanes.sampling.Sampling, rows: usize, position0: u64, words: []const u64, count: usize, out: []u32) !bool {
        const per_row = 3 * count + 2;
        const cut = sm.top_p > 0.0 and sm.top_p < 1.0;
        const union_rule = cut and g.nucleus_union;
        const min_log = sm.minLog();
        const ents = try g.gpa.alloc(NucEnt, 2 * count);
        defer g.gpa.free(ents);
        for (0..rows) |r| {
            var total: u64 = 0;
            var top = -std.math.inf(f64);
            var lows: [2]f64 = undefined;
            for (0..2) |k| {
                const at = (k * rows + r) * per_row;
                const vals = words[at..][0..count];
                const ids = words[at + count ..][0..count];
                const mass = words[at + 2 * count ..][0..count];
                total += words[at + 3 * count];
                lows[k] = std.math.inf(f64);
                for (0..count) |j| {
                    const v: f64 = @bitCast(vals[j]);
                    ents[k * count + j] = .{ .v = v, .id = @bitCast(ids[j]), .m = @bitCast(mass[j]) };
                    top = @max(top, v);
                    lows[k] = @min(lows[k], v);
                }
            }
            const need: i64 = if (cut) @intCast(ceilShare(sm.top_p, total)) else 0;
            const floor = top + min_log;
            std.mem.sort(NucEnt, ents, {}, NucEnt.before);
            if (cut and union_rule) {
                var cum: i64 = 0;
                var stop: ?usize = null;
                for (ents, 0..) |en, i| {
                    cum += en.m;
                    if (cum >= need) {
                        stop = i;
                        break;
                    }
                }
                const at = stop orelse return false;
                for (lows) |low| if (!(ents[at].v > low)) return false;
            } else for (0..2) |k| {
                // a rank's own share must end inside its candidates (min_p alone: a candidate below its cut)
                var cum: i64 = 0;
                var covered = false;
                if (cut) {
                    for (ents) |en| {
                        if (en.id < @as(i64, 0) or @as(usize, @intCast(en.id)) / g.n.e.w.head.n != k) continue;
                        cum += en.m;
                        if (cum >= need) {
                            covered = en.v > lows[k];
                            break;
                        }
                    }
                } else covered = sm.min_p > 0.0 and lows[k] < floor;
                if (!covered) return false;
            }
            var keep: usize = ents.len;
            if (cut) {
                var cum: i64 = 0;
                var below: usize = 0;
                for (ents) |en| {
                    cum += en.m;
                    if (cum < need) below += 1 else break;
                }
                keep = below + 1;
            }
            if (sm.min_p > 0.0) {
                var n: usize = 0;
                for (ents) |en| {
                    if (en.v >= floor) n += 1 else break;
                }
                keep = @min(keep, n);
            }
            var best: usize = 0;
            var best_score = -std.math.inf(f64);
            for (ents[0..keep], 0..) |en, j| {
                const score = en.v - libm_log(-libm_log(lanes.sampling.uniform(sm.seed, position0 + r, @intCast(en.id))));
                if (j == 0 or score > best_score) {
                    best = j;
                    best_score = score;
                }
            }
            out[r] = @intCast(ents[best].id);
        }
        return true;
    }

    /// Each row's draw among the gathered candidates (exact_sampling.choose_rows), row r keyed at position0 + r.
    fn choose(g: *Glm, s: *lanes.Stream, which: Node.Rows, k: usize, rows: usize, position0: u64, out: []u32) !void {
        const sm = s.sampling.?;
        if (k == Node.whole_vocabulary) return g.chooseWhole(sm, which, rows, position0, out);
        const host = &g.n.e.cand_host;
        var values: [2 * fwd.Engine.max_k]f64 = undefined;
        var ids: [2 * fwd.Engine.max_k]u64 = undefined;
        for (0..rows) |r| {
            var m: usize = 0;
            for (0..2) |rank| {
                const base = (rank * rows + r) * 2 * k;
                for (0..k) |j| {
                    values[m] = @as(f32, @bitCast(host[base + 2 * j]));
                    ids[m] = host[base + 2 * j + 1];
                    m += 1;
                }
            }
            out[r] = @intCast(try lanes.sampling.choose(g.gpa, values[0..m], ids[0..m], position0 + r, sm));
        }
    }

    fn prefillFn(p: *anyopaque, s: *lanes.Stream) anyerror!void {
        const g = of(p);
        const e = &g.n.e;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + max_rows > e.st.capacity) return error.PromptTooLong;
        const k = try candidateCount(s);
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
        g.pass = ids;
        defer {
            g.pass = &.{};
            g.pass_at = null;
        }
        var next: usize = 0; // the planned chunk starts: a resumed pass cuts where a fresh one does
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            while (next < s.chunks.len and s.chunks[next] <= at) next += 1;
            const end: usize = if (next < s.chunks.len) s.chunks[next] else ids.len;
            const rows = @min(end - at, fwd.Engine.prefill_rows);
            try g.n.chunk(at, rows);
            try ln.live.appendSlice(g.gpa, ids[at..][0..rows]);
            at += rows;
            g.pass_at = @intCast(at);
            if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) != null) if (s.reuse.hook) |hk| hk.at(hk.ptr, s, @intCast(at));
        }
        const first = try g.n.first(k);
        var one: [1]u32 = .{first};
        if (k > 0) try g.choose(s, .last, k, 1, ids.len, &one);
        ln.first = g.take(one[0]);
    }

    fn firstFn(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const g = of(p);
        if (position != s.prompt_len) return error.PositionMismatch;
        return (g.laneOf(s) orelse return error.NoLane).first;
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
    fn verifyShared(g: *Glm, windows: []const be.Window, out: []be.Verified) !void {
        if (windows.len > max_lanes) return error.TooManyWindows;
        var segs: [max_lanes]fwd.Engine.Seg = undefined;
        var ks: [max_lanes]usize = undefined;
        var ids: [max_rows]u32 = undefined;
        var total: usize = 0;
        for (windows, 0..) |w, i| {
            if (w.parents != null) return error.TreesNotBuilt;
            const ln = try g.enter(w.stream);
            try g.commit(ln, ln.pending_rows); // no keep since its last window: all of its rows stay
            const rows = w.rows();
            if (total + rows > max_rows) return error.WindowTooWide;
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
        var at: usize = 0;
        for (windows, out, segs[0..n], ks[0..n]) |w, *o, sg, k| {
            const rows = sg.rows;
            try g.n.bind(sg.lane); // a draw's `nucleus` reads the bound lane's rows
            try g.n.pickSegment(sg, k, o.sampled[0..rows]);
            if (k > 0) try g.choose(w.stream, .window, k, rows, w.positions[0], o.sampled[0..rows]);
            @memcpy(o.drafts[0 .. rows - 1], ids[at + 1 ..][0 .. rows - 1]);
            const ln = &g.lanes[sg.lane];
            @memcpy(ln.window[0..rows], ids[at..][0..rows]);
            ln.pending_rows = rows;
            ln.held_n = 0;
            at += rows;
        }
    }

    fn keepFn(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
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
    fn draftDflash(g: *Glm, ln: *Lane, r: be.DraftRequest) !void {
        const t = try g.dflashTaps(ln, r, true);
        g.dflashChain(ln, r, t, if (t.want > 0) &g.n.dflash.? else null);
    }

    /// A DFlash2 turn's first half: the kept rows committed and joined to the lane's context, and how many drafts it wants.
    const Turn = struct { want: usize, pending: u32 };

    /// The bound lane's DFlash2 turn up to its chain: commit, the context update and, with `block`, its own block pass.
    fn dflashTaps(g: *Glm, ln: *Lane, r: be.DraftRequest, block: bool) !Turn {
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
        try g.n.dflashTurn(from, keep, pending, block and want > 0);
        return .{ .want = want, .pending = pending };
    }

    /// A DFlash2 turn's second half: the chain from `dft`'s candidates (null: no block pass ran, no drafts), held.
    fn dflashChain(g: *Glm, ln: *Lane, r: be.DraftRequest, t: Turn, dft: ?*const dflash.Drafter) void {
        const s = r.stream;
        ln.dn = if (dft) |d| d.chain(t.want, t.pending, r.position, if (s.sampling) |sm| .{ .seed = sm.seed, .temperature = sm.temperature } else null, g.policy, &ln.dtoks) else 0;
        const held = @max(ln.dn, r.depth);
        @memcpy(ln.held[0..ln.dn], ln.dtoks[0..ln.dn]);
        for (ln.dn..held) |i| ln.held[i] = t.pending;
        ln.held_n = held;
    }

    /// A shared round's DFlash2 turns (facts.draft_streams): each lane's context update on its own, then the block passes of
    /// every lane that wants drafts in one (Node.dflashBlocks; one lane alone takes its own captured pass), then the chains.
    fn draftDflashLanes(g: *Glm, requests: []const be.DraftRequest) !void {
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
        for (requests, 0..) |r, i| {
            const ln = try g.enter(r.stream);
            idx[i] = g.n.e.lane;
            turns[i] = try g.dflashTaps(ln, r, false);
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

    fn treeFn(p: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const g = of(p);
        if (!g.dflash) return null;
        const ln = g.laneOf(s) orelse return error.NoLane;
        return .{ .count = @intCast(ln.dn), .tokens = try gpa.dupe(u32, ln.dtoks[0..ln.dn]) };
    }

    /// Absorb the kept rows with their following tokens, then hold `depth` drafts for the next window.
    fn draftFn(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
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
    fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
        const g = of(p);
        const ln = g.laneOf(s) orelse return;
        ln.stream = null;
        ln.pending_rows = 0;
        ln.held_n = 0;
        ln.dn = 0;
    }

    fn backend(g: *Glm) be.Backend {
        g.vtable = .{ .prefill = prefillFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .tree = treeFn };
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
            // several live streams share a forward (verifyShared), up to a window's rows; the lanes `grow` allocated cap
            // how many streams are live, so one lane never shares
            .hidden_rows = true,
            .draft_streams = true, // a shared round's drafts in one call: DFlash2's block passes share one (draftDflashLanes)
            .streams_exact = true,
            .batch_rows = max_rows,
            .max_streams = max_lanes,
        };
    }

    /// Shared windows of 2 to max_rows rows over lanes 0 and 1 (half each), timed on whatever their caches hold (both
    /// ranks run them; no commit, so the lanes stay where they were). Serving overwrites the rows they wrote.
    fn measureShared(g: *Glm) !void {
        var ids: [max_rows]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast(3000 + 37 * i);
        var out: [max_rows]u32 = undefined;
        g.shared[0] = g.costs[0];
        for (2..max_rows + 1) |total| {
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
        std.log.info("GLM shared window ms by rows (two lanes): 2 {d:.1}, 4 {d:.1}, 8 {d:.1}, 16 {d:.1}", .{ g.shared[1].ms, g.shared[3].ms, g.shared[7].ms, g.shared[max_rows - 1].ms });
    }

    /// Each window width's forward and one chained MTP step, timed on a short dummy prompt (both ranks run it).
    fn measure(g: *Glm, io: std.Io) !void {
        var prompt: [64]u32 = undefined;
        for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + i);
        _ = try g.n.prefill(&prompt, !g.dflash, 0);
        var out: [max_rows]u32 = undefined;
        var ids: [max_rows]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast(2000 + i);
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
};

/// A stream's candidates a rank (top_k + MARGIN); 0 for greedy. top_k 0 (the full-vocabulary nucleus) is not ported.
/// k candidates a rank (top_k + exact_sampling.MARGIN), 0 for greedy, or the whole vocabulary: top_k off
/// (sampling.nucleus_rows) or past the candidates a rank gives.
fn candidateCount(s: *const lanes.Stream) !usize {
    const sm = s.sampling orelse return 0;
    if (sm.temperature <= 0) return 0;
    const k = @as(usize, sm.top_k) + 8;
    if (sm.top_k == 0 or k > fwd.Engine.max_k) return Node.whole_vocabulary;
    return k;
}

/// sampling.nucleus_rows for one row over the whole vocabulary `row` (bf16 bits by token id). Python reads each
/// rank's top 1,024 first and every whole shard when a row's nucleus runs past them, so its draw is this one: tokens
/// in (value desc, id asc) order with v = logit / T in float64, mass floor(exp(v - top) 2^40); keep a token while the
/// mass before it is below ceil(top_p total) (top_p in (0, 1)), and while v >= top + ln(min_p); the Gumbel race
/// v - ln(-ln u(seed, position, id)) among them, the first maximum in that order. Both cuts keep a prefix of the
/// order, so the kept set is found by selection (no sort). Tokens 42 or more below the top are left out: their mass
/// is 0 (below 2^-40) and their race score stays under the top token's (the race's noise spans 41.1).
fn nucleus(gpa: std.mem.Allocator, row: []const u16, position: u64, s: lanes.sampling.Sampling) !u32 {
    const t = @max(s.temperature, 1e-6);
    var top = -std.math.inf(f64);
    for (row) |b| top = @max(top, bf16Value(b) / t);
    var cands: std.ArrayList(Tok) = .empty;
    defer cands.deinit(gpa);
    try cands.ensureTotalCapacity(gpa, row.len);
    var total: u64 = 0;
    const floor = top + s.minLog();
    for (row, 0..) |b, i| {
        const v = bf16Value(b) / t;
        if (v < top - 42.0 or v < floor) continue; // min_p's cut keeps a prefix too: drop its tail now
        // floor(exp(d) 2^40) is 0 below d = -40 ln 2 (-27.7): no exp for those
        const mass: u64 = if (v - top < -28.0) 0 else @intFromFloat(@floor(libm_exp(v - top) * 0x1p40));
        cands.appendAssumeCapacity(.{ .v = v, .id = @intCast(i), .mass = mass });
    }
    // the cut's total is every token's mass, the ones below min_p's floor included
    for (row) |b| {
        const v = bf16Value(b) / t;
        if (v >= floor and v >= top - 42.0) continue;
        if (v - top >= -28.0) total += @intFromFloat(@floor(libm_exp(v - top) * 0x1p40));
    }
    for (cands.items) |c| total += c.mass;
    var kept = cands.items;
    if (s.top_p > 0.0 and s.top_p < 1.0) kept = kept[0..prefixWithin(kept, ceilShare(s.top_p, total))];
    var best = kept[0];
    var best_score = raceScore(best, position, s.seed);
    for (kept[1..]) |c| {
        const score = raceScore(c, position, s.seed);
        if (score > best_score or (score == best_score and before(c, best))) {
            best = c;
            best_score = score;
        }
    }
    return best.id;
}

const Tok = struct { v: f64, id: u32, mass: u64 };

fn before(a: Tok, b: Tok) bool {
    return a.v > b.v or (a.v == b.v and a.id < b.id);
}

fn raceScore(c: Tok, position: u64, seed: u64) f64 {
    return c.v - libm_log(-libm_log(lanes.sampling.uniform(seed, position, c.id)));
}

/// Reorders `a` so its kept tokens come first and returns how many: a token is kept when the mass of the tokens
/// before it in (value desc, id asc) order is below `need` (quickselect on that order, expected linear).
fn prefixWithin(a: []Tok, need: u64) usize {
    var lo: usize = 0;
    var hi: usize = a.len;
    var rem = need;
    while (hi > lo) {
        // pivot: the middle element, moved to the end; partition [lo, hi - 1) into before-pivot and after-pivot
        std.mem.swap(Tok, &a[lo + (hi - lo) / 2], &a[hi - 1]);
        const p = a[hi - 1];
        var m = lo;
        var sum: u64 = 0;
        for (lo..hi - 1) |i| {
            if (before(a[i], p)) {
                sum += a[i].mass;
                std.mem.swap(Tok, &a[i], &a[m]);
                m += 1;
            }
        }
        std.mem.swap(Tok, &a[m], &a[hi - 1]); // pivot at m, [lo, m) before it, (m, hi) after it
        if (sum >= rem) {
            hi = m; // the kept tokens all lie before the pivot
            continue;
        }
        // [lo, m) and the pivot are kept; the rest only while mass remains
        if (sum + p.mass >= rem) return m + 1;
        rem -= sum + p.mass;
        lo = m + 1;
    }
    return lo;
}

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;
const libm_exp = exp;
const libm_log = log;

fn bf16Value(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

/// ceil(Fraction(p) * total) exactly: p's double as m 2^-s.
fn ceilShare(p: f64, total: u64) u64 {
    const f = std.math.frexp(p); // p = f.significand 2^f.exponent, significand in [0.5, 1)
    const m: u128 = @intFromFloat(std.math.ldexp(f.significand, 53));
    const shift: i32 = 53 - f.exponent;
    const prod = m * total;
    if (shift <= 0) return @intCast(prod << @intCast(-shift));
    if (shift >= 128) return @intFromBool(prod > 0);
    const sh: u7 = @intCast(shift);
    return @intCast((prod + ((@as(u128, 1) << sh) - 1)) >> sh);
}

test {
    _ = @import("glm_segments.zig");
    _ = @import("glm_kernels.zig");
}

test "ceilShare is ceil(p * total) over the double's exact value" {
    try std.testing.expectEqual(@as(u64, 95), ceilShare(0.95, 100)); // 0.95's double sits just below 0.95
    try std.testing.expectEqual(@as(u64, 50), ceilShare(0.5, 100));
    try std.testing.expectEqual(@as(u64, 1), ceilShare(0.5, 1));
    try std.testing.expectEqual(@as(u64, 0), ceilShare(0.5, 0));
}

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn ms(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e6;
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
    g.* = .{ .nucleus_union = !envOff("TENSORFOLD_NUCLEUS_UNION"), .nucleus_first = envCount("TF_GLM_NUCLEUS", 1024), .gpa = gpa, .n = n, .drafts = o.drafts and (n.w.mtp != null or n.dflash != null), .dflash = n.dflash != null, .policy = try dflash.Policy.fromEnv(), .copy_cfg = try copy.Settings.fromEnv(n.e.wide_graph) };
    for (&g.lanes) |*ln| ln.* = .{ .copy = .{ .cfg = g.copy_cfg } };
    if (g.dflash and g.drafts) std.log.info("GLM drafts: DFlash2 (block {d}, up to {d} drafts a round, stop at {d:.2} of the noisy chain, beta {d:.1}); the MTP head is not loaded", .{ n.w.draft.?.block, g.policy.most, g.policy.confidence, g.policy.beta });
    if (g.drafts and g.copy_cfg.on) std.log.info("GLM copy drafts: up to {d} a round, match {d} tokens, {d} inside the reply, windows of 9 to 15 rows padded to {d}", .{ g.copy_cfg.most, g.copy_cfg.match, g.copy_cfg.reply_match, g.copy_cfg.pad });
    g.io = io;
    try g.measure(io);
    g.shared = g.costs;
    return .{ .backend = g.backend(), .facts = g.facts(), .rows = if (g.drafts) max_rows else 1, .stream_bytes = n.laneBytes(), .ctx = g, .deinit = release };
}

/// Lanes up to `total` on both ranks, allocated now (native/cuda.zig calls this once, after the memory admission
/// and before serving): how many there are.
pub fn grow(p: *anyopaque, total: u32) !u32 {
    const g = Glm.of(p);
    g.n_lanes = try g.n.grow(total);
    if (g.n_lanes > 1 and g.drafts) try g.measureShared();
    return @intCast(g.n_lanes);
}

/// A lane's state for the pick: whether a stream uses it, when one last took it, and whether its rows hold the
/// request's kept state.
const View = struct { busy: bool, used: u64, holds: bool };
/// The lane a prompt pass runs in, and the lane whose rows it copies first (null: none).
const Pick = struct { lane: usize, from: ?usize = null };

/// A free lane holding the kept state; else the free lane used longest ago, copying from a busy lane holding it.
fn pickLane(v: []const View) ?Pick {
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

/// Whether lane `ln`'s attention caches' rows still hold kept state `sn`'s tokens: the `at` before it and the one
/// its MTP rows read past it.
fn liveHolds(ln: *const Lane, sn: *const Snap) bool {
    return ln.live.items.len > sn.at and std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ln.live.items[0 .. sn.at + 1])) == sn.key;
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OneStreamAtATime => "this build of the GLM CUDA engine runs one request's window at a time",
        error.NoFreeLane => "every lane of the GLM CUDA engine is in use: retry when a request finishes",
        else => null,
    };
}

// -- kept prompt states (prompt_cache.Snapshots, wired by native/cuda.zig) ------------------------------------------
/// A kept state: KDA's state after `at` prompt tokens on both ranks under `id`, and the hash of the `at + 1` tokens
/// the attention caches must still hold for it (liveHolds).
const Snap = struct { id: u32, at: u32, key: u64 };

/// prompt_cache.Rules: the MTP head's row at-1 reads token at; prompt kernels change bits with a chunk's rows, so
/// states are kept and resumed at the request's planned chunk starts only.
pub const cache_lookahead: u32 = 1;
pub const cache_planned = true;

pub fn snapBytes(p: *anyopaque, at: u32) u64 {
    _ = at;
    return Glm.of(p).n.snapshotBytes();
}

pub fn snapSave(p: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
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
pub fn snapRestore(p: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = p;
    _ = owner;
    _ = saved;
    return error.RestoredByThePromptPass;
}

pub fn snapDrop(p: *anyopaque, saved: *anyopaque) void {
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
