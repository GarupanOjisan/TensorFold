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

    /// Each row's draw over the whole vocabulary (Node.whole_vocabulary): nucleus_rows for top_k off, else
    /// choose_rows over every token (top_k past the candidates a rank gives). Row r keyed at position0 + r.
    fn chooseWhole(g: *Glm, sm: lanes.sampling.Sampling, rows: usize, position0: u64, out: []u32) !void {
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

    /// Each row's draw among the gathered candidates (exact_sampling.choose_rows), row r keyed at position0 + r.
    fn choose(g: *Glm, s: *lanes.Stream, k: usize, rows: usize, position0: u64, out: []u32) !void {
        const sm = s.sampling.?;
        if (k == Node.whole_vocabulary) return g.chooseWhole(sm, rows, position0, out);
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
        g.stream = s;
        g.pending_rows = 0;
        g.held_n = 0;
        const first = try g.n.prefill(ids, g.drafts, k);
        var one: [1]u32 = .{first};
        if (k > 0) try g.choose(s, k, 1, ids.len, &one);
        _ = g.take(one[0]);
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
        try g.commit(g.pending_rows); // no keep since the last window: all of its rows stay (Nemotron's bindLane)
        const rows = w.rows();
        if (rows > max_rows) return error.WindowTooWide;
        if (w.held > g.held_n) return error.NoHeldDrafts;
        const e = &g.n.e;
        for (w.positions, 0..) |pos, r| if (pos != e.st.pos + 1 + r) return error.PositionMismatch;
        var ids: [max_rows]u32 = undefined;
        ids[0] = w.pending;
        @memcpy(ids[1..][0..w.held], g.held[0..w.held]);
        @memcpy(ids[1 + w.held ..][0..w.tokens.len], w.tokens);
        const k = try candidateCount(w.stream);
        try g.n.verify(ids[0..rows], k, out[0].sampled[0..rows]);
        if (k > 0) try g.choose(w.stream, k, rows, w.positions[0], out[0].sampled[0..rows]);
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
        _ = try g.n.prefill(&prompt, true, 0);
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
