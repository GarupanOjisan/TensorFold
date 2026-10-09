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
const dflash = @import("glm_dflash.zig");
const roce_mod = @import("glm_roce.zig");

const Engine = fwd.Engine;

pub const Kind = enum(u32) { begin = 1, verify = 2, keep = 3, draft = 4, quit = 5, chunk = 6, first = 7, save = 8, drop = 9, dflash = 10, nucleus = 11, bind = 12, grow = 13, copy = 14, shared = 15, segment = 16, dflash_blocks = 17, keep_lanes = 18, chunk_lanes = 19, dflash_taps = 20, pick_rows = 21 };

/// TF_GLM_DRAFTER_DIR: the DFlash2 draft model's snapshot folder (v1.8's DRAFTER=dflash2); unset: the MTP head drafts.
pub fn drafterDir() ?[]const u8 {
    const v = std.c.getenv("TF_GLM_DRAFTER_DIR") orelse return null;
    const s = std.mem.span(v);
    return if (s.len == 0) null else s;
}

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
    prompts: [Engine.max_lanes]std.ArrayList(u32) = @splat(.empty), // each lane's begun prompt, which `chunk` and `chunkLanes` read
    mtp: bool = false,
    roce: ?*roce_mod.Roce = null,
    dflash: ?dflash.Drafter = null, // the bound lane's DFlash2 context (bind swaps it with `drafters`)
    drafters: [Engine.max_lanes]?dflash.Drafter = @splat(null), // every other lane's
    batch: ?dflash.Batch = null, // several lanes' block passes at once (dflashBlocks), with two lanes or more
    snaps: std.AutoHashMapUnmanaged(u32, cuda.DeviceBuffer) = .empty, // kept prompt states by id, this rank's half

    /// Both ranks: the TCP pair (rank 0 listens on `port`), NCCL, the rank's weights, kernels and engine state.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, context: usize, rank: u32, master: []const u8, port: u16) !*Node {
        const d = ctx.d;
        const n = try gpa.create(Node);
        errdefer gpa.destroy(n);
        n.gpa = gpa;
        n.rank = rank;
        n.words = .empty;
        n.prompts = @splat(.empty);
        n.mtp = false;
        n.snaps = .empty;
        n.drafters = @splat(null);
        n.batch = null;
        n.stream = try cuda.Stream.init(d, true);
        errdefer n.stream.deinit();
        n.lib = try cuda.nccl.Library.open();
        errdefer n.lib.close();
        n.pair = if (rank == 0) try cuda.Pair.listen(port) else try cuda.Pair.connect(master, port, 1800);
        errdefer n.pair.close();
        // both ranks must have been started with the same window and the same TF_GLM_DENSE (their all-gathered partials
        // would mix 4-bit and BF16 projections)
        // and the same row split (TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP: one rank's swaps would wait on the other's forever)
        var mine = [5]u32{ @intCast(context), 0x474c4d31, @intFromEnum(try wt.denseKind()), @intFromBool(drafterDir() != null), (try fwd.SplitSettings.fromEnv()).code() };
        var theirs: [5]u32 = undefined;
        if (rank == 0) {
            try n.pair.send(std.mem.sliceAsBytes(&mine));
            try n.pair.recv(std.mem.sliceAsBytes(&theirs));
        } else {
            try n.pair.recv(std.mem.sliceAsBytes(&theirs));
            try n.pair.send(std.mem.sliceAsBytes(&mine));
        }
        if (!std.mem.eql(u32, &mine, &theirs)) {
            std.log.err("the two GLM ranks were started with different settings: window {d} here, {d} on the other rank; TF_GLM_DENSE {t} here, kind {d} there; row split {x} here, {x} there", .{ mine[0], theirs[0], try wt.denseKind(), theirs[2], mine[4], theirs[4] });
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
        n.k = try kern.Kernels.load(d);
        errdefer n.k.deinit();
        n.roce = null;
        const comm_mode = try roce_mod.mode(.auto);
        if (comm_mode != .nccl) n.roce = roce_mod.Roce.init(gpa, d, &n.k, n.pair, rank, roce_mod.maxBytes()) catch |err| blk: {
            if (comm_mode == .roce) return err;
            std.log.warn("RoCE all-gather unavailable ({t}); NCCL carries every gather (TF_GLM_COMM=nccl silences this)", .{err});
            break :blk null;
        };
        errdefer if (n.roce) |r| r.deinit();
        if (n.roce != null) std.log.info("RoCE all-gathers up to {d} KiB a rank; larger ones over NCCL (TF_GLM_COMM, TF_ROCE_MAX_KB)", .{roce_mod.maxBytes() >> 10});
        const draft_dir = drafterDir();
        n.w = try wt.load(gpa, io, d, .{ .d = d, .s = n.stream }, &n.k, dir, cfg, rank, draft_dir == null); // DFlash2 and the MTP head are not loaded together (TF_GLM_MTP=auto)
        errdefer n.w.deinit(gpa);
        n.dflash = null;
        if (draft_dir) |dd| try wt.loadDraft(gpa, io, d, .{ .d = d, .s = n.stream }, &n.k, &n.w, dd, rank, rank == 0);
        n.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, kernels);
        errdefer n.set.deinit();
        const long_context = context > fwd.denseLimit(&n.w);
        n.e = try Engine.init(&n.w, d, n.stream, .{ .set = &n.set, .s = n.stream }, &n.k, .{ .lib = &n.lib, .comm = n.comm, .roce = n.roce }, context + Engine.max_rows, long_context, if (n.w.draft) |dr| dr.taps else null);
        errdefer n.e.deinit();
        if (n.w.draft != null) n.dflash = try dflash.Drafter.init(gpa, d, &n.w, context + Engine.max_rows);
        return n;
    }

    pub fn deinit(n: *Node) void {
        var it = n.snaps.valueIterator();
        while (it.next()) |b| b.free();
        n.snaps.deinit(n.gpa);
        for (&n.prompts) |*pr| pr.deinit(n.gpa);
        if (n.dflash) |*dr| dr.deinit();
        for (&n.drafters) |*slot| if (slot.*) |*dr| dr.deinit();
        if (n.batch) |*b| b.deinit();
        n.e.deinit();
        if (n.roce) |r| r.deinit();
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
        const mine = &n.prompts[n.e.lane];
        mine.clearRetainingCapacity();
        try mine.appendSlice(n.gpa, prompt);
        n.mtp = mtp;
        if (snap == no_snapshot) {
            try n.e.reset();
            if (n.dflash) |*dr| try dr.reset(&n.e);
            return;
        }
        const b = n.snaps.get(snap) orelse return error.NoSuchSnapshot;
        try n.e.restoreState(b.ptr, at);
        if (n.dflash) |*dr| try dr.restore(&n.e, b.ptr + n.e.snapshotBytes(), at);
    }

    /// The begun prompt's rows at .. at + rows, at the state's position.
    pub fn chunk(n: *Node, at: usize, rows: usize) !void {
        try n.send(.chunk, &.{ @intCast(at), @intCast(rows) });
        try dec.prefillChunk(&n.e, n.prompts[n.e.lane].items, at, rows, n.mtp, if (n.dflash) |*dr| dr else null);
    }

    /// One prompt chunk over several begun lanes' next rows (multi_prefill.prefill_pieces; rank 0 sends it, both run it):
    /// `items` (lane, at, rows), each piece at the next multiple of 64 rows of the prompt buffers; then per piece its taps
    /// into its lane's DFlash2 context and its lane's commit. Not with the MTP head (its cache is not grouped).
    pub fn chunkLanes(n: *Node, items: []const [3]usize) !void {
        n.words.clearRetainingCapacity();
        try n.words.append(n.gpa, @intCast(items.len));
        for (items) |it| try n.words.appendSlice(n.gpa, &.{ @intCast(it[0]), @intCast(it[1]), @intCast(it[2]) });
        try n.send(.chunk_lanes, n.words.items);
        if (n.mtp or items.len > Engine.max_lanes) return error.BadChunk;
        var pieces: [Engine.max_lanes]Engine.Piece = undefined;
        var ids: [Engine.prefill_rows]u32 = undefined;
        var at: usize = 0;
        for (items, 0..) |it, i| {
            const lane = it[0];
            if (lane >= n.e.n_lanes) return error.NoSuchLane;
            const prompt = n.prompts[lane].items;
            if (it[2] == 0 or it[1] + it[2] > prompt.len or n.e.laneState(lane).pos != it[1]) return error.ChunkOutOfStep;
            const lo = std.mem.alignForward(usize, at, 64);
            if (lo + it[2] > ids.len) return error.BadChunk;
            @memset(ids[at..lo], 0);
            @memcpy(ids[lo..][0..it[2]], prompt[it[1]..][0..it[2]]);
            pieces[i] = .{ .lane = lane, .rows = it[2], .lo = lo };
            at = lo + it[2];
        }
        try n.e.prefillPieces(pieces[0..items.len], ids[0..at]);
        for (pieces[0..items.len]) |pc| {
            try n.bind(pc.lane);
            if (n.dflash) |*dr| { // the piece's rows of the taps (the row0 offset dflashRun takes)
                var b = n.e.pbuf;
                for (&b.taps) |*t| if (t.* != 0) {
                    t.* += pc.lo * n.e.w.cfg.hidden * 2;
                };
                try dr.addTaps(&n.e, &b, pc.rows);
            }
            try n.e.commitChunk(pc.rows);
        }
    }

    /// The first token after the prompt's last chunk. k 0: greedy, the token returned; else the last row's k
    /// candidates a rank gathered (Engine.cand_host), or both ranks' whole rows (whole_vocabulary).
    pub fn first(n: *Node, k: usize) !u32 {
        try n.send(.first, &.{@intCast(k)});
        return n.pick(n.e.st.last_logits, 1, k, null);
    }

    /// Keep the state between prompt chunks as `id` (each rank its half).
    pub fn save(n: *Node, id: u32) !void {
        try n.send(.save, &.{id});
        errdefer n.drop(id); // rank 0 failed: rank 1 frees its half too
        var b = try cuda.DeviceBuffer.alloc(n.e.d, n.snapshotBytes());
        errdefer b.free();
        try n.e.saveState(b.ptr);
        if (n.dflash) |*dr| try dr.save(&n.e, b.ptr + n.e.snapshotBytes());
        try n.stream.synchronize(); // the copy is whole before a later chunk changes the state
        try n.snaps.put(n.gpa, id, b);
    }

    /// A kept prompt state's bytes on this rank: the engine's, and DFlash2's ring.
    pub fn snapshotBytes(n: *const Node) usize {
        return n.e.snapshotBytes() + if (n.dflash) |*dr| dr.stateBytes() else 0;
    }

    /// DFlash2's turn (rank 0 sends it, both run it): from .verify the first `keep` rows of the last window join its context
    /// (their taps); then, with `block`, the block pass for `pending`, whose merged candidates rank 0 reads back (`chainOf`).
    pub fn dflashTurn(n: *Node, from: From, kept: usize, pending: u32, block: bool) !void {
        try n.send(.dflash, &.{ @backingInt(from), @intCast(kept), pending, @intFromBool(block) });
        try n.dflashRun(from, kept, pending, block);
    }

    /// DFlash2's context updates of several lanes after a shared round (rank 0 sends it, both run it): lane items[i][0]'s
    /// first items[i][1] rows of the last window join its context, stacked in one pass (dflash.Batch.taps), else lane by lane.
    pub fn dflashTapsLanes(n: *Node, items: []const [2]usize) !void {
        n.words.clearRetainingCapacity();
        try n.words.append(n.gpa, @intCast(items.len));
        for (items) |it| try n.words.appendSlice(n.gpa, &.{ @intCast(it[0]), @intCast(it[1]) });
        try n.send(.dflash_taps, n.words.items);
        if (items.len > Engine.max_lanes) return error.BadDraftBatch;
        var ds: [Engine.max_lanes]*dflash.Drafter = undefined;
        var kept: [Engine.max_lanes]usize = undefined;
        var row0: [Engine.max_lanes]usize = undefined;
        for (items, 0..) |it, i| {
            ds[i] = try n.drafterOf(it[0]);
            kept[i] = it[1];
            row0[i] = n.e.laneState(it[0]).row0;
        }
        if (n.batch) |*b| if (try b.taps(&n.e, ds[0..items.len], kept[0..items.len], row0[0..items.len], &n.e.buf)) return;
        for (items) |it| {
            try n.bind(it[0]);
            try n.dflashRun(.verify, it[1], 0, false);
        }
    }

    /// Lane `lane`'s DFlash2 drafter (the bound lane's is `dflash`).
    pub fn drafterOf(n: *Node, lane: usize) !*dflash.Drafter {
        if (lane == n.e.lane) return &(n.dflash orelse return error.NoDraftModel);
        return &(n.drafters[lane] orelse return error.NoDraftModel);
    }

    /// The block passes of `lanes` (each at its context's end, `pend` its pending token) in one pass (dflash.Batch.blocks;
    /// rank 0 sends it, both run it, rank 0 reads the candidates back into each lane's drafter).
    pub fn dflashBlocks(n: *Node, lanes: []const usize, pend: []const u32) !void {
        n.words.clearRetainingCapacity();
        try n.words.append(n.gpa, @intCast(lanes.len));
        for (lanes) |l| try n.words.append(n.gpa, @intCast(l));
        try n.words.appendSlice(n.gpa, pend);
        try n.send(.dflash_blocks, n.words.items);
        var ds: [Engine.max_lanes]*dflash.Drafter = undefined;
        if (lanes.len > Engine.max_lanes) return error.BadDraftBatch;
        for (lanes, 0..) |l, i| ds[i] = try n.drafterOf(l);
        const b = &(n.batch orelse return error.NoDraftBatch);
        try b.blocks(&n.e, ds[0..lanes.len], pend, n.rank == 0);
    }

    fn dflashRun(n: *Node, from: From, kept: usize, pending: u32, block: bool) !void {
        const dr = &(n.dflash orelse return error.NoDraftModel);
        if (from == .verify) { // the lane's rows of the last window
            var b = n.e.buf;
            for (&b.taps) |*t| if (t.* != 0) {
                t.* += n.e.st.row0 * n.e.w.cfg.hidden * 2;
            };
            try dr.addTaps(&n.e, &b, kept);
        }
        if (block and dr.context_end > 0) try dr.block(&n.e, pending, n.rank == 0);
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

    /// A `k` asking for a draw over the vocabulary (sampling.nucleus_rows, or a top_k past the candidates a rank gives): the
    /// forward leaves its rows' logits where they are and the draw's `nucleus` commands follow.
    pub const whole_vocabulary: usize = Engine.max_k + 1;

    /// Which rows a `nucleus` command reads: the last window's, or the prompt's last row.
    pub const Rows = enum(u32) { window = 0, last = 1 };

    /// The draw's gather (rank 0 sends it, both run it): `count` > 0: Engine.nucleusGather's candidates; 0: every rank's whole rows
    /// (Engine.fullRows, into Engine.full_host). The words are valid until the next call.
    pub fn nucleus(n: *Node, which: Rows, rows: usize, temperature: f64, count: usize) ![]const u64 {
        const t: u64 = @bitCast(temperature);
        try n.send(.nucleus, &.{ @backingInt(which), @intCast(rows), @intCast(count), @truncate(t), @truncate(t >> 32) });
        return n.nucleusRun(which, rows, temperature, count);
    }

    fn nucleusRun(n: *Node, which: Rows, rows: usize, temperature: f64, count: usize) ![]const u64 {
        const logits = switch (which) {
            .window => n.e.buf.logits + n.e.st.row0 * n.e.w.head.n * 2, // the bound lane's rows of the window
            .last => n.e.st.last_logits,
        };
        if (count == 0) {
            _ = try n.e.fullRows(logits, rows);
            return &.{};
        }
        return n.e.nucleusGather(logits, rows, temperature, count);
    }

    fn pick(n: *Node, logits: u64, rows: usize, k: usize, out: ?[]u32) !u32 {
        if (k == whole_vocabulary) return 0; // the draw's own commands (`nucleus`) read the rows
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

    /// keep for several lanes of the last shared window at once (Engine.commitSegs).
    pub fn keepLanes(n: *Node, items: []const Engine.Keep) !void {
        n.words.clearRetainingCapacity();
        try n.words.append(n.gpa, @intCast(items.len));
        for (items) |it| try n.words.appendSlice(n.gpa, &.{ @intCast(it.lane), @intCast(it.rows), @intCast(it.keep) });
        try n.send(.keep_lanes, n.words.items);
        var segs: [Engine.max_lanes]Engine.Keep = undefined;
        var m: usize = 0;
        for (items) |it| {
            if (it.lane < n.e.n_lanes and n.e.laneState(it.lane).seg) {
                segs[m] = it;
                m += 1;
                continue;
            }
            try n.bind(it.lane); // a window it ran alone: its own commit
            try n.e.commit(it.rows, it.keep);
        }
        try n.e.commitSegs(segs[0..m]);
    }

    /// Absorb (hidden row, following token) pairs and draft `out.len` tokens.
    pub fn draft(n: *Node, from: From, next: []const u32, out: []u32) !void {
        n.words.clearRetainingCapacity();
        try n.words.appendSlice(n.gpa, &.{ @backingInt(from), @intCast(out.len) });
        try n.words.appendSlice(n.gpa, next);
        try n.send(.draft, n.words.items);
        const hidden = switch (from) {
            .prompt => n.e.st.last_hidden,
            .verify => n.e.buf.fnormed + n.e.st.row0 * n.e.w.cfg.hidden * 2, // the lane's rows of the window
        };
        try dec.draft(&n.e, hidden, next, out.len, out);
    }

    /// Several lanes' windows in one forward (Engine.forwardShared): `segs` in order, `ids` their rows back to back. Each
    /// segment's tokens are then picked with `pickSegment` (and drawn with `nucleus` after binding its lane), in order.
    pub fn verifyShared(n: *Node, segs: []const Engine.Seg, ids: []const u32) !void {
        n.words.clearRetainingCapacity();
        try n.words.append(n.gpa, @intCast(segs.len));
        for (segs) |g| try n.words.appendSlice(n.gpa, &.{ @intCast(g.lane), @intCast(g.rows) });
        try n.words.appendSlice(n.gpa, ids);
        try n.send(.shared, n.words.items);
        try n.e.forwardShared(segs, ids);
    }

    /// Segment `g`'s rows of the last shared window (rank 0 sends it, both run it): each row's greedy token into `out`
    /// (k 0), or its k candidates a rank gathered (Engine.cand_host), as `verify`.
    pub fn pickSegment(n: *Node, g: Engine.Seg, k: usize, out: []u32) !void {
        try n.send(.segment, &.{ @intCast(g.lane), @intCast(g.rows), @intCast(k) });
        const row0 = n.e.laneState(g.lane).row0;
        _ = try n.pick(n.e.buf.logits + row0 * n.e.w.head.n * 2, g.rows, k, out);
    }

    /// Every row of the last shared window at once (rank 0 sends it, both run it): `pick` over its first `rows` rows of
    /// buf.logits (the segments back to back from row 0), one greedy (or top-k candidates) pass, one gather, one copy back.
    pub fn pickRows(n: *Node, rows: usize, k: usize, out: []u32) !void {
        try n.send(.pick_rows, &.{ @intCast(rows), @intCast(k) });
        _ = try n.pick(n.e.buf.logits, rows, k, out);
    }

    /// Later commands act on lane `lane`'s state and DFlash2 context (sent only when the lane changes).
    pub fn bind(n: *Node, lane: usize) !void {
        if (lane == n.e.lane) return;
        try n.send(.bind, &.{@intCast(lane)});
        const from = n.e.lane;
        try n.e.bind(lane);
        n.drafters[from] = n.dflash;
        n.dflash = n.drafters[lane];
        n.drafters[lane] = null;
    }

    /// One more lane's device bytes: its caches, and its DFlash2 context's.
    pub fn laneBytes(n: *const Node) usize {
        return n.e.laneBytes() + if (n.dflash) |*dr| dr.mem.len else 0;
    }

    /// Lanes up to `total` on both ranks, each with its own DFlash2 context when drafting with it: how many both have
    /// (rank 1 answers with its count).
    pub fn grow(n: *Node, total: usize) !usize {
        try n.send(.grow, &.{@intCast(total)});
        const had = n.e.n_lanes;
        var mine = try n.e.addLanes(total);
        if (n.dflash) |*bound| for (had..mine) |i| {
            n.drafters[i] = dflash.Drafter.init(n.gpa, n.e.d, &n.w, bound.capacity) catch {
                mine = i; // the lanes past it serve no stream
                break;
            };
        };
        if (n.dflash != null and mine > 1) {
            n.batch = try dflash.Batch.init(n.gpa, n.e.d, &n.w, mine);
            var ds: [Engine.max_lanes]*dflash.Drafter = undefined;
            for (0..mine) |i| ds[i] = try n.drafterOf(i);
            try n.batch.?.adopt(&n.e, ds[0..mine]);
        }
        var word = [1]u32{@intCast(mine)};
        if (n.rank != 0) {
            try n.pair.send(std.mem.sliceAsBytes(&word));
            return mine;
        }
        try n.pair.recv(std.mem.sliceAsBytes(&word));
        return @min(mine, word[0]);
    }

    /// Lane `src`'s attention-cache rows below `rows` into lane `dst` (Engine.copyLaneRows). DFlash2's context comes
    /// whole with the kept state `begin` restores, so it is not copied.
    pub fn copy(n: *Node, src: usize, dst: usize, rows: usize) !void {
        try n.send(.copy, &.{ @intCast(src), @intCast(dst), @intCast(rows) });
        try n.e.copyLaneRows(src, dst, rows);
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
                .dflash => try n.dflashRun(@fromBackingInt(@intCast(w[0])), w[1], w[2], w[3] != 0),
                .nucleus => _ = try n.nucleusRun(@fromBackingInt(@intCast(w[0])), w[1], @bitCast(@as(u64, w[3]) | (@as(u64, w[4]) << 32)), w[2]),
                .shared => {
                    const m = w[0];
                    var segs: [Engine.max_lanes]Engine.Seg = undefined;
                    if (m > Engine.max_lanes) return error.TooManySegments;
                    for (0..m) |i| segs[i] = .{ .lane = w[1 + 2 * i], .rows = w[2 + 2 * i] };
                    try n.verifyShared(segs[0..m], w[1 + 2 * m ..]);
                },
                .segment => try n.pickSegment(.{ .lane = w[0], .rows = w[1] }, w[2], scratch[0..w[1]]),
                .pick_rows => {
                    var all: [Engine.shared_rows]u32 = undefined;
                    if (w[0] > all.len) return error.WindowTooWide;
                    try n.pickRows(w[0], w[1], all[0..w[0]]);
                },
                .dflash_blocks => {
                    const m = w[0];
                    if (m > Engine.max_lanes) return error.BadDraftBatch;
                    var ls: [Engine.max_lanes]usize = undefined;
                    for (0..m) |i| ls[i] = w[1 + i];
                    try n.dflashBlocks(ls[0..m], w[1 + m ..][0..m]);
                },
                .keep_lanes => {
                    const m = w[0];
                    var items: [Engine.max_lanes]Engine.Keep = undefined;
                    if (m > Engine.max_lanes) return error.TooManySegments;
                    for (0..m) |i| items[i] = .{ .lane = w[1 + 3 * i], .rows = w[2 + 3 * i], .keep = w[3 + 3 * i] };
                    try n.keepLanes(items[0..m]);
                },
                .chunk_lanes => {
                    const m = w[0];
                    var items: [Engine.max_lanes][3]usize = undefined;
                    if (m > Engine.max_lanes) return error.BadChunk;
                    for (0..m) |i| items[i] = .{ w[1 + 3 * i], w[2 + 3 * i], w[3 + 3 * i] };
                    try n.chunkLanes(items[0..m]);
                },
                .dflash_taps => {
                    const m = w[0];
                    var items: [Engine.max_lanes][2]usize = undefined;
                    if (m > Engine.max_lanes) return error.BadDraftBatch;
                    for (0..m) |i| items[i] = .{ w[1 + 2 * i], w[2 + 2 * i] };
                    try n.dflashTapsLanes(items[0..m]);
                },
                .bind => try n.bind(w[0]),
                .grow => _ = try n.grow(w[0]),
                .copy => try n.copy(w[0], w[1], w[2]),
                .quit => return,
            }
        }
    }
};
