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

pub const Kind = enum(u32) { begin = 1, verify = 2, keep = 3, draft = 4, quit = 5, chunk = 6, first = 7, save = 8, drop = 9 };

/// `begin` without a kept state: the pass starts at 0.
pub const no_snapshot: u32 = 0xFFFF_FFFF;

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
    prompt: std.ArrayList(u32) = .empty, // the prompt `begin` started, which `chunk` reads
    mtp: bool = false,
    snaps: std.AutoHashMapUnmanaged(u32, cuda.DeviceBuffer) = .empty, // kept prompt states by id, this rank's half

    /// Both ranks: the TCP pair (rank 0 listens on `port`), NCCL, the rank's weights, kernels and engine state.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, context: usize, rank: u32, master: []const u8, port: u16) !*Node {
        const d = ctx.d;
        const n = try gpa.create(Node);
        errdefer gpa.destroy(n);
        n.gpa = gpa;
        n.rank = rank;
        n.words = .empty;
        n.prompt = .empty;
        n.mtp = false;
        n.snaps = .empty;
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
        var it = n.snaps.valueIterator();
        while (it.next()) |b| b.free();
        n.snaps.deinit(n.gpa);
        n.prompt.deinit(n.gpa);
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
        const head = [2]u32{ @backingInt(kind), @intCast(words.len) };
        try n.pair.send(std.mem.sliceAsBytes(&head));
        try n.pair.send(std.mem.sliceAsBytes(words));
    }

    // -- the commands (rank 0 sends, both run) ------------------------------------------------------------------
    /// The prompt's caches in chunks of up to Engine.prefill_rows from 0, then its first token (`first`).
    pub fn prefill(n: *Node, prompt: []const u32, mtp: bool, k: usize) !u32 {
        try n.begin(prompt, mtp, no_snapshot, 0);
        var at: usize = 0;
        while (at < prompt.len) {
            const rows = @min(prompt.len - at, Engine.prefill_rows);
            try n.chunk(at, rows);
            at += rows;
        }
        return n.first(k);
    }

    /// A prompt pass starts: from 0, or from kept state `snap` after `at` tokens (the attention caches' rows below
    /// `at` must still hold the prompt's tokens; the caller checks). The MTP head absorbs the prompt too when `mtp`.
    pub fn begin(n: *Node, prompt: []const u32, mtp: bool, snap: u32, at: usize) !void {
        n.words.clearRetainingCapacity();
        try n.words.appendSlice(n.gpa, &.{ @intFromBool(mtp), snap, @intCast(at) });
        try n.words.appendSlice(n.gpa, prompt);
        try n.send(.begin, n.words.items);
        n.prompt.clearRetainingCapacity();
        try n.prompt.appendSlice(n.gpa, prompt);
        n.mtp = mtp;
        if (snap == no_snapshot) return n.e.reset();
        const b = n.snaps.get(snap) orelse return error.NoSuchSnapshot;
        try n.e.restoreState(b.ptr, at);
    }

    /// The begun prompt's rows at .. at + rows, at the state's position.
    pub fn chunk(n: *Node, at: usize, rows: usize) !void {
        try n.send(.chunk, &.{ @intCast(at), @intCast(rows) });
        try dec.prefillChunk(&n.e, n.prompt.items, at, rows, n.mtp);
    }

    /// The first token after the prompt's last chunk. k 0: greedy, the token returned; else the last row's k
    /// candidates a rank gathered (Engine.cand_host), or both ranks' whole rows (whole_vocabulary).
    pub fn first(n: *Node, k: usize) !u32 {
        try n.send(.first, &.{@intCast(k)});
        return n.pick(n.e.last_logits, 1, k, null);
    }

    /// Keep the state between prompt chunks as `id` (each rank its half).
    pub fn save(n: *Node, id: u32) !void {
        try n.send(.save, &.{id});
        errdefer n.drop(id); // rank 0 failed: rank 1 frees its half too
        var b = try cuda.DeviceBuffer.alloc(n.e.d, n.e.snapshotBytes());
        errdefer b.free();
        try n.e.saveState(b.ptr);
        try n.stream.synchronize(); // the copy is whole before a later chunk changes the state
        try n.snaps.put(n.gpa, id, b);
    }

    pub fn drop(n: *Node, id: u32) void {
        n.send(.drop, &.{id}) catch {};
        if (n.snaps.fetchRemove(id)) |kv| {
            var b = kv.value;
            b.free();
        }
    }

    /// A window's forward: each row's greedy token into `out` (k 0), or each row's k candidates a rank gathered.
    pub fn verify(n: *Node, rows: []const u32, k: usize, out: []u32) !void {
        n.words.clearRetainingCapacity();
        try n.words.append(n.gpa, @intCast(k));
        try n.words.appendSlice(n.gpa, rows);
        try n.send(.verify, n.words.items);
        try n.e.forward(rows);
        _ = try n.pick(n.e.buf.logits, rows.len, k, out);
    }

    /// A `k` asking for every rank's whole logits rows (Engine.fullRows) instead of k candidates a rank.
    pub const whole_vocabulary: usize = Engine.max_k + 1;

    fn pick(n: *Node, logits: u64, rows: usize, k: usize, out: ?[]u32) !u32 {
        if (k == whole_vocabulary) {
            _ = try n.e.fullRows(logits, rows);
            return 0;
        }
        if (k > 0) {
            try n.e.candidates(&n.e.buf, logits, rows, k);
            return 0;
        }
        var one: [Engine.max_rows]u32 = undefined;
        const dst = out orelse one[0..rows];
        try n.e.greedy(&n.e.buf, logits, rows, n.e.w.head.n, dst);
        return dst[0];
    }

    /// Commit the last window's first `keep` of its `rows` rows.
    pub fn keep(n: *Node, rows: usize, kept: usize) !void {
        try n.send(.keep, &.{ @intCast(rows), @intCast(kept) });
        try n.e.commit(rows, kept);
    }

    /// Absorb (hidden row, following token) pairs and draft `out.len` tokens.
    pub fn draft(n: *Node, from: From, next: []const u32, out: []u32) !void {
        n.words.clearRetainingCapacity();
        try n.words.appendSlice(n.gpa, &.{ @backingInt(from), @intCast(out.len) });
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
            switch (@as(Kind, @fromBackingInt(@intCast(head[0])))) {
                .begin => try n.begin(w[3..], w[0] != 0, w[1], w[2]),
                .chunk => try n.chunk(w[0], w[1]),
                .first => _ = try n.first(w[0]),
                .save => try n.save(w[0]),
                .drop => n.drop(w[0]),
                .verify => try n.verify(w[1..], w[0], scratch[0 .. w.len - 1]),
                .keep => try n.keep(w[0], w[1]),
                .draft => try n.draft(@fromBackingInt(@intCast(w[0])), w[2..], scratch[0..w[1]]),
                .quit => return,
            }
        }
    }
};
