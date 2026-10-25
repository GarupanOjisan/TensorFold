//! How many tokens a copy may draft, including a reply-only match and a strong-match extension.
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;

/// The ramp this round would use before a strong-match extension (shared rounds included).
pub fn ramp(e: *const Engine, s: *const Stream) u32 {
    const width = s.copy_width orelse e.cfg.first_copy;
    return width;
}

/// Tokens to ask the proposer for. A configured strong cap asks for that, and `take` shortens a weak match.
pub fn ask(e: *const Engine, s: *const Stream) u32 {
    const base = ramp(e, s);
    return if (e.cfg.strong_copy > base) e.cfg.strong_copy else base;
}

/// The ramp after this copy: a full accept doubles plus one, up to max_copy; a miss returns to the first width.
pub fn nextWidth(current: u32, first: u32, max_copy: u32, full: bool) u32 {
    return if (full) @min(max_copy, 2 * current + 1) else first;
}

/// The copy to keep. A continuation that starts in the reply needs `reply_match`. A short match stays on the ramp.
pub fn take(e: *const Engine, s: *const Stream, match: i64, end: ?usize, copied: []const u32) []const u32 {
    const least: usize = if (e.alone and e.cfg.context_copies) 1 else 2;
    if (copied.len < least or match < e.cfg.enter_match) return &.{};
    if (e.cfg.reply_match > e.cfg.enter_match) {
        if (end) |at| if (at >= s.prompt_len and match < e.cfg.reply_match) return &.{};
    }
    if (e.cfg.strong_match > 0 and match >= e.cfg.strong_match) return copied;
    if (e.cfg.strong_copy == 0) return copied;
    const n = @min(copied.len, @as(usize, ramp(e, s)));
    return if (n >= least) copied[0..n] else &.{};
}

const std = @import("std");

test "a full accept grows the ramp, and a miss returns to the first width" {
    try std.testing.expectEqual(@as(u32, 7), nextWidth(3, 3, 15, true));
    try std.testing.expectEqual(@as(u32, 15), nextWidth(7, 3, 15, true));
    try std.testing.expectEqual(@as(u32, 15), nextWidth(15, 3, 15, true));
    try std.testing.expectEqual(@as(u32, 7), nextWidth(7, 3, 7, true));
    try std.testing.expectEqual(@as(u32, 3), nextWidth(7, 3, 15, false));
}
