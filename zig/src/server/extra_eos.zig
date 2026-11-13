//! TF_EXTRA_EOS: a comma-separated list of the tokenizer's tokens (by their text, e.g. `<|assistant|>`) that end a reply
//! like the checkpoint's own EOS ids, added after them. A checkpoint whose generation_config leaves out a turn opener
//! the model can write (GLM-5.3: `<|assistant|>`) otherwise streams it as text and starts a turn of its own until the
//! length cap. Unset or empty: the EOS list is the checkpoint's. A name the tokenizer does not hold refuses the start.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The list TF_EXTRA_EOS asks for, or null when unset.
pub fn fromEnv() ?[]const u8 {
    const v = std.c.getenv("TF_EXTRA_EOS") orelse return null;
    return std.mem.span(v);
}

/// `base` followed by each token `raw` names that it does not already hold; `unknown` gets the first name not found.
pub fn merge(a: Allocator, base: []const u32, raw: []const u8, ctx: anytype, comptime tokenId: fn (@TypeOf(ctx), []const u8) ?u32, unknown: *[]const u8) error{ OutOfMemory, UnknownExtraEos }![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, base);
    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |part| {
        const name = std.mem.trim(u8, part, " \t");
        if (name.len == 0) continue;
        const id = tokenId(ctx, name) orelse {
            unknown.* = name;
            return error.UnknownExtraEos;
        };
        if (std.mem.indexOfScalar(u32, out.items, id) == null) try out.append(a, id);
    }
    return out.toOwnedSlice(a);
}

const Fake = struct {
    fn id(_: Fake, name: []const u8) ?u32 {
        if (std.mem.eql(u8, name, "<|assistant|>")) return 154828;
        if (std.mem.eql(u8, name, "<|user|>")) return 154827;
        return null;
    }
};

test "extra eos: names add their ids once, after the checkpoint's; an unknown name refuses" {
    const a = std.testing.allocator;
    const base = [_]u32{ 154820, 154827, 154829 };
    var unknown: []const u8 = "";
    const got = try merge(a, &base, " <|assistant|>, <|user|>,,<|assistant|>", Fake{}, Fake.id, &unknown);
    defer a.free(got);
    try std.testing.expectEqualSlices(u32, &.{ 154820, 154827, 154829, 154828 }, got);
    const same = try merge(a, &base, "", Fake{}, Fake.id, &unknown);
    defer a.free(same);
    try std.testing.expectEqualSlices(u32, &base, same);
    try std.testing.expectError(error.UnknownExtraEos, merge(a, &base, "<|assistant|>,<|nope|>", Fake{}, Fake.id, &unknown));
    try std.testing.expectEqualStrings("<|nope|>", unknown);
}
