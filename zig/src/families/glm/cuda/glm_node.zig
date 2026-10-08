//! One rank of the two-rank GLM engine and the commands that keep the ranks in step: rank 0 sends each command to
//! rank 1 over the TCP pair before running it, rank 1 runs the same commands in the same order. Every token the
//! ranks pick comes from the same all-gather, so they agree without sending results back.
const std = @import("std");
const cuda = @import("cuda");
const glm_config = @import("glm_config");
const wt = @import("glm_weights.zig");
const tri = @import("glm_triton.zig");
const kern = @import("glm_kernels.zig");
const fwd = @import("glm_forward.zig");
const dec = @import("glm_decode.zig");

const Engine = fwd.Engine;

pub const Kind = enum(u32) { prefill = 1, verify = 2, keep = 3, draft = 4, quit = 5 };

/// Where a draft's hidden rows come from: the prompt's last row, or the last verify's first rows.
pub const From = enum(u32) { prompt = 0, verify = 1 };

pub const Node = struct {
    gpa: std.mem.Allocator,
    rank: u32,
    stream: cuda.Stream,
    lib: cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    pair: cuda.Pair,
    w: wt.Weights,
    set: cuda.aot.Set,
    k: kern.Kernels,
    e: Engine,
    words: std.ArrayList(u32) = .empty,

    /// Both ranks: the TCP pair (rank 0 listens on `port`), NCCL, the rank's weights, kernels and engine state.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, context: usize, rank: u32, master: []const u8, port: u16) !*Node {
        const d = ctx.d;
        const n = try gpa.create(Node);
        errdefer gpa.destroy(n);
        n.gpa = gpa;
        n.rank = rank;
        n.words = .empty;
        n.stream = try cuda.Stream.init(d, true);
        errdefer n.stream.deinit();
        n.lib = try cuda.nccl.Library.open();
        errdefer n.lib.close();
        n.pair = if (rank == 0) try cuda.Pair.listen(port) else try cuda.Pair.connect(master, port, 1800);
        errdefer n.pair.close();
        // both ranks must have been started with the same window
        var mine = [2]u32{ @intCast(context), 0x474c4d31 };
        var theirs: [2]u32 = undefined;
        if (rank == 0) {
            try n.pair.send(std.mem.sliceAsBytes(&mine));
            try n.pair.recv(std.mem.sliceAsBytes(&theirs));
        } else {
            try n.pair.recv(std.mem.sliceAsBytes(&theirs));
            try n.pair.send(std.mem.sliceAsBytes(&mine));
        }
        if (!std.mem.eql(u32, &mine, &theirs)) {
            std.log.err("the two GLM ranks were started with different windows: {d} here, {d} on the other rank", .{ mine[0], theirs[0] });
            return error.RankSettingsDiffer;
        }
        var id: cuda.nccl.UniqueId = undefined;
        if (rank == 0) {
            try n.lib.check(n.lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try n.pair.send(&id.internal);
        } else try n.pair.recv(&id.internal);
        n.comm = null;
        try n.lib.check(n.lib.api.ncclCommInitRank(&n.comm, wt.world, id, @intCast(rank)), "ncclCommInitRank");
        errdefer _ = n.lib.api.ncclCommDestroy(n.comm);

        const cfg_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(cfg_path);
        const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 22));
        defer gpa.free(cfg_text);
        const cfg = try glm_config.parse(gpa, cfg_text);
        n.w = try wt.load(gpa, io, d, .{ .d = d, .s = n.stream }, dir, cfg, rank, true);
        errdefer n.w.deinit(gpa);
        n.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, kernels);
        errdefer n.set.deinit();
        n.k = try kern.Kernels.load(d);
        errdefer n.k.deinit();
        const long_context = context > fwd.denseLimit(&n.w);
        n.e = try Engine.init(&n.w, d, n.stream, .{ .set = &n.set, .s = n.stream }, &n.k, .{ .lib = &n.lib, .comm = n.comm }, context + Engine.max_rows, long_context);
        return n;
    }

    pub fn deinit(n: *Node) void {
        n.e.deinit();
        n.k.deinit();
        n.set.deinit();
        n.w.deinit(n.gpa);
        _ = n.lib.api.ncclCommDestroy(n.comm);
        n.pair.close();
        n.lib.close();
        n.stream.deinit();
        n.words.deinit(n.gpa);
        n.gpa.destroy(n);
    }

    /// Rank 0: the command's words to rank 1 (kind, count, then the words).
    fn send(n: *Node, kind: Kind, words: []const u32) !void {
        if (n.rank != 0) return;
        const head = [2]u32{ @intFromEnum(kind), @intCast(words.len) };
        try n.pair.send(std.mem.sliceAsBytes(&head));
        try n.pair.send(std.mem.sliceAsBytes(words));
    }

    // -- the commands (rank 0 sends, both run) ------------------------------------------------------------------
    /// The prompt's caches and its first token (the MTP head absorbs the prompt too when `mtp`).
    pub fn prefill(n: *Node, prompt: []const u32, mtp: bool) !u32 {
        n.words.clearRetainingCapacity();
        try n.words.append(n.gpa, @intFromBool(mtp));
        try n.words.appendSlice(n.gpa, prompt);
        try n.send(.prefill, n.words.items);
        return dec.prefill(&n.e, prompt, mtp);
    }

    /// A window's forward; each row's greedy token into `out`.
    pub fn verify(n: *Node, rows: []const u32, out: []u32) !void {
        try n.send(.verify, rows);
        try n.e.forward(rows);
        try n.e.greedy(&n.e.buf, n.e.buf.logits, rows.len, n.e.w.head.n, out);
    }

    /// Commit the last window's first `keep` of its `rows` rows.
    pub fn keep(n: *Node, rows: usize, kept: usize) !void {
        try n.send(.keep, &.{ @intCast(rows), @intCast(kept) });
        try n.e.commit(rows, kept);
    }

    /// Absorb (hidden row, following token) pairs and draft `out.len` tokens.
    pub fn draft(n: *Node, from: From, next: []const u32, out: []u32) !void {
        n.words.clearRetainingCapacity();
        try n.words.appendSlice(n.gpa, &.{ @intFromEnum(from), @intCast(out.len) });
        try n.words.appendSlice(n.gpa, next);
        try n.send(.draft, n.words.items);
        const hidden = switch (from) {
            .prompt => n.e.last_hidden,
            .verify => n.e.buf.fnormed,
        };
        try dec.draft(&n.e, hidden, next, out.len, out);
    }

    pub fn quit(n: *Node) void {
        n.send(.quit, &.{}) catch {};
    }

    /// Rank 1: run rank 0's commands until it quits or the link closes.
    pub fn follow(n: *Node) !void {
        var buf: std.ArrayList(u32) = .empty;
        defer buf.deinit(n.gpa);
        var scratch: [Engine.max_rows]u32 = undefined;
        while (true) {
            var head: [2]u32 = undefined;
            n.pair.recv(std.mem.sliceAsBytes(&head)) catch |e| switch (e) {
                error.PeerClosed => return,
                else => return e,
            };
            try buf.resize(n.gpa, head[1]);
            try n.pair.recv(std.mem.sliceAsBytes(buf.items));
            const w = buf.items;
            switch (@as(Kind, @enumFromInt(head[0]))) {
                .prefill => _ = try n.prefill(w[1..], w[0] != 0),
                .verify => try n.verify(w, scratch[0..w.len]),
                .keep => try n.keep(w[0], w[1]),
                .draft => try n.draft(@enumFromInt(w[0]), w[2..], scratch[0..w[1]]),
                .quit => return,
            }
        }
    }
};
