//! The GLM backend's token draws: top-k among the gathered candidates and the whole-vocabulary nucleus.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const native = @import("glm_native.zig");
const node = @import("glm_node.zig");
const fwd = @import("glm_forward.zig");
const dflash = @import("glm_dflash.zig");
const glm_prompt = @import("glm_native_prompt.zig");
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
const now = glm_timing.now;
const ms = glm_timing.ms;

/// Each row's draw over the vocabulary (Node.whole_vocabulary): top_k off: sampling.nucleus_rows, its candidates a rank first
/// (`nucleus_first`, then `nucleus_wider`: TENSORFOLD_NUCLEUS_UNION, v1.10 patch 0034's rule), whole shards when a row's nucleus
/// runs past them; else choose_rows over every token (a top_k past the candidates a rank gives). Row r keyed at position0 + r.
pub fn chooseWhole(g: *Glm, sm: lanes.sampling.Sampling, which: Node.Rows, rows: usize, position0: u64, out: []u32) !void {
    if (sm.top_k == 0) {
        const stages = [_]usize{ g.nucleus_first, nucleus_wider };
        // without the union rule (stock v1.8) a row whose nucleus is not inside one rank's candidates goes straight to the shards
        for (stages[0..if (g.nucleus_union) 2 else 1]) |count| {
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
pub fn drawNucleus(g: *Glm, sm: lanes.sampling.Sampling, rows: usize, position0: u64, words: []const u64, count: usize, out: []u32) !bool {
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
pub fn choose(g: *Glm, s: *lanes.Stream, which: Node.Rows, k: usize, rows: usize, position0: u64, out: []u32) !void {
    const sm = s.sampling.?;
    if (k == Node.whole_vocabulary) return g.chooseWhole(sm, which, rows, position0, out);
    return g.chooseIn(sm, k, rows, 0, rows, position0, out);
}

/// choose's top-k draw for rows off .. off + rows of the `all` rows whose candidates the last pick gathered.
pub fn chooseIn(g: *Glm, sm: lanes.sampling.Sampling, k: usize, all: usize, off: usize, rows: usize, position0: u64, out: []u32) !void {
    const host = &g.n.e.cand_host;
    var values: [2 * fwd.Engine.max_k]f64 = undefined;
    var ids: [2 * fwd.Engine.max_k]u64 = undefined;
    for (0..rows) |r| {
        var m: usize = 0;
        for (0..2) |rank| {
            const base = (rank * all + off + r) * 2 * k;
            for (0..k) |j| {
                values[m] = @as(f32, @bitCast(host[base + 2 * j]));
                ids[m] = host[base + 2 * j + 1];
                m += 1;
            }
        }
        out[r] = @intCast(try lanes.sampling.choose(g.gpa, values[0..m], ids[0..m], position0 + r, sm));
    }
}

pub const NucEnt = struct {
    v: f64,
    id: i64,
    m: i64,

    fn before(_: void, a: NucEnt, b: NucEnt) bool {
        return a.v > b.v or (a.v == b.v and a.id < b.id);
    }
};

/// A stream's candidates a rank (top_k + MARGIN); 0 for greedy. top_k 0 (the full-vocabulary nucleus) is not ported.
/// k candidates a rank (top_k + exact_sampling.MARGIN), 0 for greedy, or the whole vocabulary: top_k off
/// (sampling.nucleus_rows) or past the candidates a rank gives.
pub fn candidateCount(s: *const lanes.Stream) !usize {
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
pub fn nucleus(gpa: std.mem.Allocator, row: []const u16, position: u64, s: lanes.sampling.Sampling) !u32 {
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

pub const Tok = struct { v: f64, id: u32, mass: u64 };

pub fn before(a: Tok, b: Tok) bool {
    return a.v > b.v or (a.v == b.v and a.id < b.id);
}

pub fn raceScore(c: Tok, position: u64, seed: u64) f64 {
    return c.v - libm_log(-libm_log(lanes.sampling.uniform(seed, position, c.id)));
}

/// Reorders `a` so its kept tokens come first and returns how many: a token is kept when the mass of the tokens
/// before it in (value desc, id asc) order is below `need` (quickselect on that order, expected linear).
pub fn prefixWithin(a: []Tok, need: u64) usize {
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
pub const libm_exp = exp;
pub const libm_log = log;

pub fn bf16Value(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

/// ceil(Fraction(p) * total) exactly: p's double as m 2^-s.
pub fn ceilShare(p: f64, total: u64) u64 {
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
