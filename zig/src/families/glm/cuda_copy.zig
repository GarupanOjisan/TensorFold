//! GLM CUDA copy-window knobs over the decode window: a copy cap, the reply match and the opt-in extended copies.
const std = @import("std");
const lanes = @import("lanes");
const state = @import("cuda_state.zig");

/// Drafted tokens a copy window holds at most: the decode window without its pending token.
pub const max_copy: u32 = state.window_rows - 1;
/// A copy whose continuation starts in the reply needs this many matching tokens (the recipe's measured bar).
pub const reply_default: i64 = 16;
/// With the opt-in flag, a match at least this long may take the whole copy cap in one round.
pub const strong_match: i64 = 16;

pub const Knobs = struct {
    copy_max: ?u32 = null, // null: the decode window's own width
    reply_match: i64 = reply_default,
    extend: bool = false,
};

/// `TF_GLM_COPY_MAX` unset keeps the decode window (15 and 7 are the earlier caps). Reply match is 16. Extend is `1`.
pub fn read() Knobs {
    return .{
        .copy_max = clampCopy(envInt("TF_GLM_COPY_MAX")),
        .reply_match = clampReply(envInt("TF_GLM_COPY_REPLY_MATCH")),
        .extend = envOne("TF_GLM_COPY_EXTEND"),
    };
}

/// `m` with the environment's copy knobs applied.
pub fn fromEnv(m: lanes.Model) lanes.Model {
    return apply(m, read());
}

/// Values outside 1..max_copy clamp into that range. Unset or unparseable text sets no cap.
pub fn clampCopy(v: ?i64) ?u32 {
    const n = v orelse return null;
    if (n < 1) return 1;
    if (n > max_copy) return max_copy;
    return @intCast(n);
}

/// Values outside 1..64 clamp into that range. Unparseable text keeps the default.
pub fn clampReply(v: ?i64) i64 {
    const n = v orelse return reply_default;
    if (n < 1) return 1;
    if (n > 64) return 64;
    return n;
}

/// Drafting facts with the copy cap applied. A cap never narrows the window below the head's own `drafts`.
pub fn apply(m: lanes.Model, k: Knobs) lanes.Model {
    var out = m;
    if (!m.mtp) return out;
    if (k.copy_max) |cap| out.exact_width = @min(m.exact_width, @max(cap, m.drafts) + 1);
    out.reply_match = k.reply_match;
    if (!k.extend) return out;
    out.context_copies = true;
    out.root_copies = true;
    out.strong_match = strong_match;
    out.strong_copy = out.exact_width - 1;
    return out;
}

fn envInt(name: [*:0]const u8) ?i64 {
    const v = std.c.getenv(name) orelse return null;
    return std.fmt.parseInt(i64, std.mem.span(v), 10) catch null;
}

fn envOne(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

test "copy caps clamp, unset keeps the decode window, and a cap keeps the head's drafts" {
    try std.testing.expectEqual(@as(?u32, null), clampCopy(null));
    try std.testing.expectEqual(@as(?u32, 7), clampCopy(7));
    try std.testing.expectEqual(@as(?u32, 15), clampCopy(15));
    try std.testing.expectEqual(@as(?u32, 1), clampCopy(0));
    try std.testing.expectEqual(@as(?u32, max_copy), clampCopy(400));
    try std.testing.expectEqual(@as(i64, 16), clampReply(null));
    try std.testing.expectEqual(@as(i64, 64), clampReply(100));
    const block = lanes.Model{ .exact_width = state.window_rows, .mtp = true, .speculate = true, .drafts = 7, .first_copy_rows = 4 };
    const unset = apply(block, .{});
    try std.testing.expectEqual(@as(u32, state.window_rows), unset.exact_width);
    try std.testing.expectEqual(@as(u32, 7), unset.drafts);
    try std.testing.expectEqual(reply_default, unset.reply_match);
    try std.testing.expect(!unset.context_copies and !unset.root_copies);
    try std.testing.expectEqual(@as(u32, 0), unset.strong_copy);
    try std.testing.expectEqual(@as(u32, 16), apply(block, .{ .copy_max = 15 }).exact_width);
    try std.testing.expectEqual(@as(u32, 8), apply(block, .{ .copy_max = 7 }).exact_width);
    try std.testing.expectEqual(@as(u32, 8), apply(block, .{ .copy_max = 3 }).exact_width); // DFlash2's 7 drafts stay
    var chain = block;
    chain.drafts = 15;
    try std.testing.expectEqual(@as(u32, 16), apply(chain, .{ .copy_max = 7 }).exact_width); // the MTP chain's 15 stay
    const ext = apply(block, .{ .copy_max = 15, .extend = true });
    try std.testing.expect(ext.context_copies and ext.root_copies);
    try std.testing.expectEqual(strong_match, ext.strong_match);
    try std.testing.expectEqual(@as(u32, 15), ext.strong_copy);
    try std.testing.expectEqual(max_copy, apply(block, .{ .extend = true }).strong_copy);
    const idle = apply(.{ .exact_width = 1 }, .{ .copy_max = 15 });
    try std.testing.expectEqual(@as(u32, 1), idle.exact_width);
}
