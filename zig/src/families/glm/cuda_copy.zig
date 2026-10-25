//! GLM CUDA copy-window knobs. DFlash and the MTP head stay on the 8-row window; copies may use 16.
const std = @import("std");
const lanes = @import("lanes");
const state = @import("cuda_state.zig");

/// Rows the decode buffers, sampler and KDA save hold. Head windows stay at `window_rows`.
pub const copy_rows = state.copy_rows;
/// Drafted tokens a copy window holds at most.
pub const max_copy: u32 = copy_rows - 1;
/// A copy whose continuation starts in the reply needs this many matching tokens (the recipe's measured bar).
pub const reply_default: i64 = 16;
/// With the opt-in flag, a match at least this long may take the whole copy cap in one round.
pub const strong_match: i64 = 16;

pub const Knobs = struct {
    copy_max: u32 = max_copy,
    reply_match: i64 = reply_default,
    extend: bool = false,
};

/// `TF_GLM_COPY_MAX` defaults to 15 (7 is the old cap). Reply match defaults to 16. Extend is `1`.
pub fn read() Knobs {
    return .{
        .copy_max = clampCopy(envInt("TF_GLM_COPY_MAX")),
        .reply_match = clampReply(envInt("TF_GLM_COPY_REPLY_MATCH")),
        .extend = envOne("TF_GLM_COPY_EXTEND"),
    };
}

/// Values outside 1..15 clamp into that range. Unparseable text keeps the default.
pub fn clampCopy(v: ?i64) u32 {
    const n = v orelse return max_copy;
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

/// Drafting facts with the copy cap applied. Head `drafts` is left as the caller set it (DFlash 7, MTP 3).
pub fn apply(m: lanes.Model, k: Knobs) lanes.Model {
    var out = m;
    if (!m.mtp) return out;
    out.exact_width = k.copy_max + 1;
    out.reply_match = k.reply_match;
    if (!k.extend) return out;
    out.context_copies = true;
    out.root_copies = true;
    out.strong_match = strong_match;
    out.strong_copy = k.copy_max;
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

test "copy caps clamp, and the head draft count is not the copy window" {
    try std.testing.expectEqual(@as(u32, 15), clampCopy(null));
    try std.testing.expectEqual(@as(u32, 7), clampCopy(7));
    try std.testing.expectEqual(@as(u32, 1), clampCopy(0));
    try std.testing.expectEqual(@as(u32, 15), clampCopy(40));
    try std.testing.expectEqual(@as(i64, 16), clampReply(null));
    try std.testing.expectEqual(@as(i64, 64), clampReply(100));
    const head = lanes.Model{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = 7, .first_copy_rows = 4 };
    const wide = apply(head, .{});
    try std.testing.expectEqual(@as(u32, 16), wide.exact_width);
    try std.testing.expectEqual(@as(u32, 7), wide.drafts);
    try std.testing.expect(!wide.context_copies and !wide.root_copies);
    try std.testing.expectEqual(@as(u32, 0), wide.strong_copy);
    const old = apply(head, .{ .copy_max = 7 });
    try std.testing.expectEqual(@as(u32, 8), old.exact_width);
    const ext = apply(head, .{ .extend = true });
    try std.testing.expect(ext.context_copies and ext.root_copies);
    try std.testing.expectEqual(strong_match, ext.strong_match);
    try std.testing.expectEqual(@as(u32, 15), ext.strong_copy);
    const idle = apply(.{ .exact_width = 1 }, .{});
    try std.testing.expectEqual(@as(u32, 1), idle.exact_width);
}
