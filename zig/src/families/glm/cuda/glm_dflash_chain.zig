//! The DFlash2 drafter's chain pick on the host: the stop policy, the keyed noise and the confidence rule.
const std = @import("std");
const Drafter = @import("glm_dflash.zig").Drafter;

/// EDGE and NOISE of dflash2.py: the selector's edge weight and the target's keyed noise weight when sampling.
const edge_weight: f64 = 0.6;
const noise_weight: f64 = 0.7;

/// The target's keyed sampling a chain's noise follows (lanes.sampling.Sampling's seed and temperature).
pub const Sampling = struct { seed: u64, temperature: f64 };

/// lanes.sampling.uniform (Python exact_sampling.uniform): a uniform (0, 1] double keyed by (seed, position, id).
fn mix(v: u64) u64 {
    var x = v;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

fn uniform(seed: u64, position: u64, id: u64) f64 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (position *% 0xD1B54A32D192ED03));
    x = mix(x ^ id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

/// engine.encode_policy's DFlash2 stop rule `fncN:P[:B]` (the product of the noise-aware confidences holds P) or `fcN:P`
/// (the product of the plain ones); TF_GLM_DFLASH_POLICY, v1.8's production setting `fnc7:0.3`.
pub const Policy = struct {
    most: usize = 7,
    confidence: f64 = 0.3,
    beta: f64 = 2.0, // dflash2.NOISY_BETA; 0: the plain softmax share

    pub fn fromEnv() !Policy {
        const v = std.c.getenv("TF_GLM_DFLASH_POLICY") orelse return .{};
        const spec = std.mem.trim(u8, std.mem.span(v), " ");
        if (spec.len == 0) return .{};
        const noisy = std.mem.startsWith(u8, spec, "fnc");
        if (!noisy and !std.mem.startsWith(u8, spec, "fc")) {
            std.log.err("TF_GLM_DFLASH_POLICY is fncN:P[:B] or fcN:P, not {s}", .{spec});
            return error.BadDraftPolicy;
        }
        var it = std.mem.splitScalar(u8, spec[if (noisy) 3 else 2..], ':');
        const most = std.fmt.parseInt(usize, it.next() orelse "", 10) catch return error.BadDraftPolicy;
        const conf = std.fmt.parseFloat(f64, it.next() orelse "") catch return error.BadDraftPolicy;
        const beta: f64 = if (noisy) (if (it.next()) |b| (std.fmt.parseFloat(f64, b) catch return error.BadDraftPolicy) else 2.0) else 0.0;
        if (most < 1 or most > 15 or conf < 0 or conf > 1) return error.BadDraftPolicy;
        return .{ .most = most, .confidence = conf, .beta = beta };
    }
};

/// dflash2.noisy_confidence: softmax((score + g) / beta) at the pick j (g null: greedy).
fn noisyConfidence(score: []const f64, noise: ?[]const f64, j: usize, beta: f64) f64 {
    var z: [32]f64 = undefined;
    var top = -std.math.inf(f64);
    for (score, 0..) |s, i| {
        z[i] = (if (noise) |g| s + g[i] else s) / beta;
        top = @max(top, z[i]);
    }
    var sum: f64 = 0;
    var ej: f64 = 0;
    for (score, 0..) |_, i| {
        const e = exp(z[i] - top);
        if (i == j) ej = e;
        sum += e;
    }
    return ej / sum;
}

/// dflash2.Drafter.chain: pick candidates with the selector's edges and the target's keyed noise; stop below the
/// cumulative confidence, always keeping the first draft. `first` is the position of the first draft. Up to `depth` tokens.
pub fn chain(self: *const Drafter, depth: usize, anchor: u32, first: u64, sampling: ?Sampling, policy: Policy, out: []u32) usize {
    const dr = self.dr;
    const k = dr.top_k;
    const sr = dr.sel_rank;
    const sampled = if (sampling) |s| s.temperature > 0 else false;
    const temp: f64 = if (sampled) sampling.?.temperature else 1.0;
    var n: usize = 0;
    var prev: u32 = anchor;
    var chain_p: f64 = 1.0;
    for (0..depth) |d| {
        const toks = self.cand_tokens[d * k ..][0..k];
        const vals = self.cand_values[d * k ..][0..k];
        const proj = self.cand_proj[d * sr ..][0..sr];
        const pred = dr.pred[@as(usize, prev) * sr ..][0..sr];
        var score: [32]f64 = undefined;
        var noise: [32]f64 = undefined;
        var pick: [32]f64 = undefined;
        var q: [512]f64 = undefined;
        for (0..sr) |i| q[i] = @as(f64, pred[i]) * proj[i];
        for (0..k) |j| {
            const succ = dr.succ[@as(usize, toks[j]) * sr ..][0..sr];
            var edge: f64 = 0;
            for (0..sr) |i| edge += @as(f64, succ[i]) * q[i];
            score[j] = (vals[j] + edge_weight * edge) / temp;
            if (sampled) {
                noise[j] = -log(-log(uniform(sampling.?.seed, first + d, toks[j])));
                pick[j] = score[j] + noise_weight * noise[j];
            } else pick[j] = score[j];
        }
        var best: usize = 0;
        for (1..k) |j| if (pick[j] > pick[best]) {
            best = j;
        };
        if (policy.beta > 0) {
            chain_p *= noisyConfidence(score[0..k], if (sampled) noise[0..k] else null, best, policy.beta);
            if (d > 0 and chain_p < policy.confidence) break;
        } else if (policy.confidence > 0) {
            var top = -std.math.inf(f64);
            for (score[0..k]) |s| top = @max(top, s);
            var sum: f64 = 0;
            var ej: f64 = 0;
            for (score[0..k], 0..) |s, j| {
                const e = exp(s - top);
                if (j == best) ej = e;
                sum += e;
            }
            chain_p *= ej / sum;
            if (d > 0 and chain_p < policy.confidence) break;
        }
        prev = toks[best];
        out[n] = prev;
        n += 1;
    }
    return n;
}
