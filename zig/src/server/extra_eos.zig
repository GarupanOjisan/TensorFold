//! TF_EXTRA_EOS: comma-separated token texts (GLM-5.3: `<|assistant|>`) that end a reply like the checkpoint's EOS ids.
const std = @import("std");
const Allocator = std.mem.Allocator;
const log = @import("log.zig");

/// The list TF_EXTRA_EOS asks for, or null when unset.
pub fn fromEnv() ?[]const u8 {
    const v = std.c.getenv("TF_EXTRA_EOS") orelse return null;
    return std.mem.span(v);
}

/// `base` with TF_EXTRA_EOS's tokens merged in and logged; unset returns `base`, an unknown name is logged and refused.
pub fn withEnv(a: Allocator, base: []const u32, ctx: anytype, comptime tokenId: fn (@TypeOf(ctx), []const u8) ?u32) error{ OutOfMemory, UnknownExtraEos }![]const u32 {
    const raw = fromEnv() orelse return base;
    var unknown: []const u8 = "";
    const eos = merge(a, base, raw, ctx, tokenId, &unknown) catch |e| {
        if (e == error.UnknownExtraEos) log.line("TF_EXTRA_EOS: the tokenizer has no token {s}", .{unknown});
        return e;
    };
    log.line("TF_EXTRA_EOS: a reply also ends at {s} ({d} stop ids)", .{ raw, eos.len });
    return eos;
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
