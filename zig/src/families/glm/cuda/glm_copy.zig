//! v1.8's prompt-lookup ("copy") drafts (glm5_next/cuda/copy_drafts.py, patches 0007, 0013 and 0032): when the context's last
//! `match` tokens (the pending token included) occurred before, the tokens that followed their latest earlier occurrence
//! are verified in place of the MTP head's drafts. They only propose; verification keeps the serial sample.
//!
//! Three rules differ from the lanes core's `SuffixLookup` (a 3-gram index, the longest match of at least 4 tokens, silent for
//! 16 rounds after four missed ones), so GLM brings its own (glm_native.zig hands it to the stream):
//! - which occurrence: the latest one with `k` tokens after it, else the earliest (not the longest match);
//! - an occurrence that starts inside the reply must match `reply_match` tokens (COPY_CODE: 16), not `match` (8): code replies
//!   repeat short boilerplate whose continuations differ;
//! - a window of 9 to 15 rows is padded to the CUDA-graph width (16) by repeating its last draft, when the room allows.
//! There is no silence, and a copy of one token is a copy (the lanes core wants two).
const std = @import("std");

/// Verify windows up to this many rows always replay CUDA graphs (engine.GRAPH_ROWS).
pub const graphed = 8;
/// The most drafts a round: a 16-row window (engine.MAX_ROWS - 1).
pub const max_drafts = 15;

pub const Settings = struct {
    on: bool = true, // TF_GLM_COPY_DRAFTS (production: 1)
    match: usize = 8, // TF_GLM_COPY_MATCH
    most: usize = max_drafts, // TF_GLM_COPY_MAX (production: 15; Python's own default is 5)
    reply_match: usize = 16, // TF_GLM_COPY_REPLY_MATCH (production COPY_CODE=1: 16; 0: the prompt's rule everywhere)
    pad: usize = 16, // the engine's TF_GLM_WIDE_GRAPHS width (0: no padding)

    /// The v1.8 production settings, from the TF_GLM_COPY_* variables; `wide_graph` is the engine's width.
    pub fn fromEnv(wide_graph: usize) !Settings {
        var s: Settings = .{ .pad = wide_graph };
        const on = std.c.getenv("TF_GLM_COPY_DRAFTS");
        if (on) |v| {
            const t = std.mem.trim(u8, std.mem.span(v), " ");
            if (t.len == 0 or std.mem.eql(u8, t, "0")) s.on = false else if (!std.mem.eql(u8, t, "1")) {
                std.log.err("TF_GLM_COPY_DRAFTS is 0 or 1, not {s}", .{t});
                return error.BadCopySettings;
            }
        }
        s.match = try number("TF_GLM_COPY_MATCH", s.match, 2, 64);
        s.most = try number("TF_GLM_COPY_MAX", s.most, 1, max_drafts);
        s.reply_match = try number("TF_GLM_COPY_REPLY_MATCH", s.reply_match, 0, 64);
        if (s.reply_match != 0 and s.reply_match <= s.match) {
            std.log.err("TF_GLM_COPY_REPLY_MATCH is 0 (off) or more than TF_GLM_COPY_MATCH ({d}), not {d}", .{ s.match, s.reply_match });
            return error.BadCopySettings;
        }
        return s;
    }

    fn number(name: [*:0]const u8, default: usize, low: usize, high: usize) !usize {
        const v = std.c.getenv(name) orelse return default;
        const t = std.mem.trim(u8, std.mem.span(v), " ");
        if (t.len == 0) return default;
        const n = std.fmt.parseInt(usize, t, 10) catch 0;
        if (n < low or n > high) {
            std.log.err("{s} is {d} to {d}, not {s}", .{ name, low, high, t });
            return error.BadCopySettings;
        }
        return n;
    }

    /// Whether `ctx[i .. i + match]` is an earlier occurrence of the context's last `match` tokens (`prompt` leading
    /// tokens are the prompt's: an occurrence starting past them also needs the tokens before it to match).
    fn occurs(s: Settings, ctx: []const u32, prompt: usize, i: usize) bool {
        const n = s.match;
        const len = ctx.len;
        if (!std.mem.eql(u32, ctx[i .. i + n], ctx[len - n ..])) return false;
        if (s.reply_match > n and i >= prompt) {
            const extra = s.reply_match - n;
            if (i < extra or len - n < extra) return false;
            for (1..extra + 1) |k| if (ctx[i - k] != ctx[len - n - k]) return false;
        }
        return true;
    }

    /// CopyDrafts.propose: up to `min(most, room)` drafts, what followed the latest earlier occurrence that has that many
    /// tokens after it, else the most after any occurrence (the earliest); none when the suffix never occurred before. A
    /// window of more than `graphed` rows short of the `pad` width is filled up to it by repeating the last draft, when
    /// `room` allows. `ctx` is the prompt, the reply so far and the pending token; the result lives in `buf`.
    pub fn propose(s: Settings, ctx: []const u32, prompt: usize, room: usize, buf: *[max_drafts]u32) []const u32 {
        const k = @min(s.most, room);
        const n = s.match;
        const len = ctx.len;
        if (!s.on or k < 1 or len <= n) return buf[0..0];
        const last = ctx[len - 1];
        var found: ?usize = null;
        if (len >= n + k) { // the latest start with k tokens after its match
            var i = len - n - k + 1;
            while (i > 0) {
                i -= 1;
                if (ctx[i + n - 1] == last and s.occurs(ctx, prompt, i)) {
                    found = i;
                    break;
                }
            }
        }
        if (found == null) { // none: the earliest start, which leaves fewer than k tokens
            var i = if (len >= n + k) len - n - k + 1 else 0;
            while (i < len - n) : (i += 1) {
                if (ctx[i + n - 1] == last and s.occurs(ctx, prompt, i)) {
                    found = i;
                    break;
                }
            }
        }
        const at = (found orelse return buf[0..0]) + n;
        const count = @min(k, len - at);
        @memcpy(buf[0..count], ctx[at..][0..count]);
        const rows = 1 + count;
        if (s.pad > graphed and rows > graphed and rows < s.pad and s.pad - 1 <= room) {
            @memset(buf[count .. s.pad - 1], buf[count - 1]);
            return buf[0 .. s.pad - 1];
        }
        return buf[0..count];
    }
};

test "the latest occurrence with room, else the earliest" {
    const s: Settings = .{ .match = 2, .most = 3, .reply_match = 0, .pad = 0 };
    var buf: [max_drafts]u32 = undefined;
    // the suffix (1 2) occurred at 0 and 5, both with 3 tokens after them: the later one
    const ctx = [_]u32{ 1, 2, 5, 6, 7, 1, 2, 8, 9, 1, 2 };
    try std.testing.expectEqualSlices(u32, &.{ 8, 9, 1 }, s.propose(&ctx, 0, 10, &buf));
    // room 1 is one draft, from the same occurrence
    try std.testing.expectEqualSlices(u32, &.{8}, s.propose(&ctx, 0, 1, &buf));
    // (3 4) occurred once, at 0, with only 3 tokens after it where 4 are wanted: the earliest start, short
    const wide: Settings = .{ .match = 2, .most = 4, .reply_match = 0, .pad = 0 };
    const short = [_]u32{ 3, 4, 7, 3, 4 };
    try std.testing.expectEqualSlices(u32, &.{ 7, 3, 4 }, wide.propose(&short, 0, 10, &buf));
    // never seen
    const fresh = [_]u32{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(usize, 0), s.propose(&fresh, 0, 10, &buf).len);
}

test "an occurrence inside the reply needs the reply's longer match" {
    const s: Settings = .{ .match = 2, .most = 2, .reply_match = 4, .pad = 0 };
    var buf: [max_drafts]u32 = undefined;
    // prompt 9 9, reply 1 2 5 7 1 2: the occurrence of (1 2) at 2 starts in the reply, and the token before it (9) is not
    // the one before the suffix (7)
    const ctx = [_]u32{ 9, 9, 1, 2, 5, 7, 1, 2 };
    try std.testing.expectEqual(@as(usize, 0), s.propose(&ctx, 2, 10, &buf).len);
    // with the tokens before it equal, it copies
    const same = [_]u32{ 9, 9, 6, 6, 1, 2, 5, 6, 6, 1, 2 };
    try std.testing.expectEqualSlices(u32, &.{ 5, 6 }, s.propose(&same, 2, 10, &buf));
    // an occurrence in the prompt keeps the short match
    try std.testing.expectEqualSlices(u32, &.{ 5, 7 }, s.propose(&.{ 1, 2, 5, 7, 4, 1, 2 }, 6, 10, &buf));
}

test "a window of 9 to 15 rows is padded to the graph width when room allows" {
    const s: Settings = .{ .match = 2, .most = 15, .reply_match = 0, .pad = 16 };
    var buf: [max_drafts]u32 = undefined;
    var ctx: [30]u32 = undefined;
    for (&ctx, 0..) |*t, i| t.* = @intCast(100 + i);
    ctx[28] = 100;
    ctx[29] = 101; // the suffix (100 101) occurred at 0
    // 28 tokens follow the match at 0, so 15 drafts are plain
    try std.testing.expectEqual(@as(usize, 15), s.propose(&ctx, 0, 40, &buf).len);
    // room 10: ten drafts, 11 rows, below the pad's room (15): not padded
    try std.testing.expectEqual(@as(usize, 10), s.propose(&ctx, 0, 10, &buf).len);
    // a match with only 14 tokens after it (earliest, short): 15 rows, padded to 16 by repeating the last draft
    var short: [16]u32 = undefined;
    for (&short, 0..) |*t, i| t.* = @intCast(200 + i);
    short[14] = 200;
    short[15] = 201;
    const got = s.propose(&short, 0, 40, &buf);
    try std.testing.expectEqual(@as(usize, 15), got.len);
    try std.testing.expectEqual(@as(u32, 201), got[14]); // 14 copied tokens end in 201, the pad repeats it
    try std.testing.expectEqual(got[13], got[14]);
}
