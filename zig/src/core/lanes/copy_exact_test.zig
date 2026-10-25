//! GLM-shaped copy windows: drafted tokens match a `"draft": false` run, greedy and sampled.
const std = @import("std");
const Config = @import("config.zig").Config;
const Model = @import("config.zig").Model;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const Sampling = @import("sampling.zig").Sampling;
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const fake = @import("fake.zig");

const gpa = std.testing.allocator;
const eos = [_]u32{96};
const sampled: Sampling = .{ .seed = 11, .temperature = 0.7, .top_k = 0, .top_p = 0.9 };

fn glmModel() Model {
    return .{
        .exact_width = 16,
        .first_copy_rows = 4,
        .mtp = true,
        .speculate = true,
        .speculate_early = false,
        .drafts = 7,
        .hidden_rows = true,
        .batch_rows = 32,
        .max_streams = 16,
        .copy_match = 8,
        .reply_match = 16,
        .streams_exact = true,
    };
}

const Out = struct { tokens: []u32, drafted: u64 };

fn run(prompts: []const []const u32, sampling: ?Sampling, drafts: bool) ![]Out {
    var cfg = try Config.init(gpa, glmModel(), 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(Stream, prompts.len);
    defer gpa.free(streams);
    const looks = try gpa.alloc(SuffixLookup, prompts.len);
    defer gpa.free(looks);
    for (prompts, streams, looks) |p, *s, *look| {
        look.* = try SuffixLookup.init(gpa, .{ .ngram = 3, .min_match = 6 });
        s.* = try Stream.init(gpa, .{
            .id = "s",
            .prompt = p,
            .max_new = 24,
            .eos = &eos,
            .sampling = sampling,
            .drafts = drafts,
            .proposer = look.proposer(),
        });
    }
    defer for (streams, looks) |*s, *look| {
        s.deinit(gpa);
        look.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    const out = try gpa.alloc(Out, prompts.len);
    for (out, streams) |*o, *s| o.* = .{ .tokens = try gpa.dupe(u32, s.emitted()), .drafted = s.drafted };
    return out;
}

fn freeOut(rows: []Out) void {
    for (rows) |o| gpa.free(o.tokens);
    gpa.free(rows);
}

/// Two copies of a 16-token block, then the block's first 8, so a prompt copy has a continuation.
const repeat = blk: {
    var p: [40]u32 = undefined;
    for (&p, 0..) |*t, i| t.* = @as(u32, @intCast(i % 16));
    break :blk p;
};
const other = [_]u32{ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 3, 5, 7, 9, 11 };

test "drafted GLM copy windows match draft false, greedy and sampled" {
    var cfg = try Config.init(gpa, glmModel(), 16, 15);
    defer cfg.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 15), cfg.max_copy);
    try std.testing.expectEqual(@as(u32, 3), cfg.first_copy);
    try std.testing.expectEqual(@as(u32, 7), cfg.most_drafts);
    for ([_]?Sampling{ null, sampled }) |sampling| {
        const drafted = try run(&.{&repeat}, sampling, true);
        defer freeOut(drafted);
        const plain = try run(&.{&repeat}, sampling, false);
        defer freeOut(plain);
        try std.testing.expectEqualSlices(u32, plain[0].tokens, drafted[0].tokens);
        try std.testing.expectEqual(@as(u64, 0), plain[0].drafted);
        if (sampling == null) try std.testing.expect(drafted[0].drafted > 0);
    }
}

fn propose(extend: bool, alone: bool, width: ?u32, prompt: []const u32, extra: []const u32) ![]const u32 {
    var cfg = try Config.init(gpa, .{
        .exact_width = 16,
        .first_copy_rows = 4,
        .mtp = true,
        .speculate = true,
        .drafts = 7,
        .copy_match = 8,
        .reply_match = 16,
        .strong_match = if (extend) 16 else 0,
        .strong_copy = if (extend) 15 else 0,
    }, 16, 15);
    var target: fake.Fake = .{ .gpa = gpa };
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    engine.alone = alone;
    var look = try SuffixLookup.init(gpa, .{ .ngram = 3, .min_match = 6 });
    var s = try Stream.init(gpa, .{ .id = "p", .prompt = prompt, .max_new = 64, .proposer = look.proposer() });
    if (extra.len > 0) try s.context.appendSlice(gpa, extra);
    s.copy_width = width;
    const got = try @import("windows.zig").copyProposal(&engine, &s);
    const kept = try gpa.dupe(u32, got.tokens);
    s.deinit(gpa);
    look.deinit();
    engine.deinit();
    target.deinit();
    cfg.deinit(gpa);
    return kept;
}

test "reply copies need 16 tokens, and a shared ramp is the grown width" {
    var block: [24]u32 = undefined;
    for (&block, 0..) |*t, i| t.* = @intCast(i + 1);
    const prompt = [_]u32{ 90, 91, 92, 93 };
    var reply: [34]u32 = undefined;
    @memcpy(reply[0..24], &block);
    @memcpy(reply[24..], block[0..10]);
    const short = try propose(false, true, null, &prompt, &reply);
    defer gpa.free(short);
    try std.testing.expectEqual(@as(usize, 0), short.len);
    @memcpy(reply[24..], block[0..10]);
    var long_reply: [40]u32 = undefined;
    @memcpy(long_reply[0..24], &block);
    @memcpy(long_reply[24..], block[0..16]);
    const allowed = try propose(false, false, 7, &prompt, &long_reply);
    defer gpa.free(allowed);
    try std.testing.expect(allowed.len >= 2 and allowed.len <= 7);
    var prompt_hist: [40]u32 = undefined;
    @memcpy(prompt_hist[0..24], &block);
    @memcpy(prompt_hist[24..], block[0..16]);
    const capped = try propose(false, true, null, &prompt_hist, &.{});
    defer gpa.free(capped);
    try std.testing.expect(capped.len >= 2 and capped.len <= 3);
    const extended = try propose(true, true, null, &prompt_hist, &.{});
    defer gpa.free(extended);
    try std.testing.expect(extended.len > 3 and extended.len <= 15);
}

test "two streams' drafted tokens match each stream's draft false run" {
    for ([_]?Sampling{ null, sampled }) |sampling| {
        const drafted = try run(&.{ &repeat, &other }, sampling, true);
        defer freeOut(drafted);
        const a = try run(&.{&repeat}, sampling, false);
        defer freeOut(a);
        const b = try run(&.{&other}, sampling, false);
        defer freeOut(b);
        try std.testing.expectEqualSlices(u32, a[0].tokens, drafted[0].tokens);
        try std.testing.expectEqualSlices(u32, b[0].tokens, drafted[1].tokens);
    }
}
