//! `tf-glm`: GLM-5.3-Flash on two CUDA ranks without Python, and its check against the Python engine's replies.
//!   tf-glm check MODEL --rank R --master HOST [--port P] --prompts prompts.json --want tokens.json
//!          [--max-tokens N] [--context N] [--depth D] [--only NAME,...] [--kernels DIR]
//!   tf-glm qdump MODEL --rank R --out DIR [--only NAME,...]   (TF_GLM_DENSE=q4's quantization of a few tensors, as bytes)
//! Run once on each Spark (rank 0 on HOST). Each prompt is prefilled and decoded serially, then with MTP drafts;
//! rank 0 prints whether both equal the capture's serial reply.
const std = @import("std");
const cuda = @import("cuda");
const glm_config = @import("glm_config");
const wt = @import("glm_weights.zig");
const tri = @import("glm_triton.zig");
const kern = @import("glm_kernels.zig");
const fwd = @import("glm_forward.zig");
const dec = @import("glm_decode.zig");
const node = @import("glm_node.zig");

const Opts = struct {
    model: []const u8 = "",
    rank: u32 = 0,
    master: []const u8 = "",
    port: u16 = 29660,
    prompts: []const u8 = "",
    want: []const u8 = "",
    max_tokens: usize = 128,
    context: usize = 1048576,
    depth: usize = 3,
    only: []const u8 = "",
    kernels: ?[]const u8 = null,
    out: []const u8 = "",
};

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3 or !(std.mem.eql(u8, args[1], "check") or std.mem.eql(u8, args[1], "follow") or std.mem.eql(u8, args[1], "qdump"))) {
        std.debug.print("usage: tf-glm check MODEL --rank R --master HOST --prompts P --want W [options]\n" ++
            "       tf-glm follow MODEL --master HOST [--port P] [--context N] [--kernels DIR]   (rank 1 of a server)\n" ++
            "       tf-glm qdump MODEL --rank R --out DIR [--only NAME,...]\n", .{});
        return 2;
    }
    var o: Opts = .{ .model = args[2] };
    var i: usize = 3;
    while (i + 1 < args.len) : (i += 2) {
        const a = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, a, "--rank")) o.rank = try std.fmt.parseInt(u32, v, 10)
        else if (std.mem.eql(u8, a, "--master")) o.master = v
        else if (std.mem.eql(u8, a, "--port")) o.port = try std.fmt.parseInt(u16, v, 10)
        else if (std.mem.eql(u8, a, "--prompts")) o.prompts = v
        else if (std.mem.eql(u8, a, "--want")) o.want = v
        else if (std.mem.eql(u8, a, "--max-tokens")) o.max_tokens = try std.fmt.parseInt(usize, v, 10)
        else if (std.mem.eql(u8, a, "--context")) o.context = try std.fmt.parseInt(usize, v, 10)
        else if (std.mem.eql(u8, a, "--depth")) o.depth = try std.fmt.parseInt(usize, v, 10)
        else if (std.mem.eql(u8, a, "--only")) o.only = v
        else if (std.mem.eql(u8, a, "--kernels")) o.kernels = v
        else if (std.mem.eql(u8, a, "--out")) o.out = v
        else {
            std.debug.print("unknown option {s}\n", .{a});
            return 2;
        }
    }
    if (std.mem.eql(u8, args[1], "qdump")) return qdump(init, o);
    const kernels_dir = o.kernels orelse init.environ_map.get("TENSORFOLD_CUDA_KERNELS") orelse return error.NoKernelDir;
    if (std.mem.eql(u8, args[1], "follow")) return follow(init, o, kernels_dir);

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();

    // the two ranks: NCCL's id over TCP, then the communicator
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    var pair = if (o.rank == 0) try cuda.Pair.listen(o.port) else try cuda.Pair.connect(o.master, o.port, 600);
    defer pair.close();
    // both ranks must run the same TF_GLM_DENSE: their all-gathered partials would mix 4-bit and BF16 projections
    const dense_mine: u32 = @intFromEnum(try wt.denseKind());
    var dense_theirs: u32 = undefined;
    if (o.rank == 0) {
        try pair.send(std.mem.asBytes(&dense_mine));
        try pair.recv(std.mem.asBytes(&dense_theirs));
    } else {
        try pair.recv(std.mem.asBytes(&dense_theirs));
        try pair.send(std.mem.asBytes(&dense_mine));
    }
    if (dense_mine != dense_theirs) {
        std.log.err("the two ranks run different TF_GLM_DENSE (kind {d} here, {d} on the other rank)", .{ dense_mine, dense_theirs });
        return error.RankSettingsDiffer;
    }
    var id: cuda.nccl.UniqueId = undefined;
    if (o.rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        try pair.send(&id.internal);
    } else try pair.recv(&id.internal);
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, wt.world, id, @intCast(o.rank)), "ncclCommInitRank");
    defer _ = lib.api.ncclCommDestroy(comm);

    const cfg_path = try std.fs.path.join(arena, &.{ o.model, "config.json" });
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, arena, .limited(1 << 22));
    const cfg = try glm_config.parse(gpa, cfg_text);

    var k = try kern.Kernels.load(&driver);
    defer k.deinit();
    var t0 = now(io);
    var w = try wt.load(gpa, io, &driver, .{ .d = &driver, .s = stream }, &k, o.model, cfg, o.rank, true);
    defer w.deinit(gpa);
    std.debug.print("[tf-glm] rank {d}: weights {d:.1} GiB in {d:.1} s\n", .{ o.rank, @as(f64, @floatFromInt(w.bytes)) / (1 << 30), secs(io, t0) });

    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, kernels_dir);
    defer set.deinit();
    const long_context = o.context > fwd.denseLimit(&w);
    var e = try fwd.Engine.init(&w, &driver, stream, .{ .set = &set, .s = stream }, &k, .{ .lib = &lib, .comm = comm }, o.context + fwd.Engine.max_rows, long_context);
    defer e.deinit();

    const prompts_text = try std.Io.Dir.cwd().readFileAlloc(io, o.prompts, arena, .limited(1 << 26));
    const prompts = try std.json.parseFromSliceLeaky(std.json.Value, arena, prompts_text, .{});
    const want_text = try std.Io.Dir.cwd().readFileAlloc(io, o.want, arena, .limited(1 << 26));
    const want = try std.json.parseFromSliceLeaky(std.json.Value, arena, want_text, .{});

    var failed: usize = 0;
    var it = prompts.object.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (o.only.len > 0 and !listed(o.only, name)) continue;
        const ids = try arena.alloc(u32, entry.value_ptr.array.items.len);
        for (entry.value_ptr.array.items, ids) |v, *x| x.* = @intCast(v.integer);
        const ref = want.object.get(name).?.object.get("serial").?.array.items;
        const count = @min(o.max_tokens, ref.len);
        for ([_]bool{ false, true }) |drafted| {
            var out: std.ArrayList(u32) = .empty;
            defer out.deinit(gpa);
            t0 = now(io);
            const first = try dec.prefill(&e, ids, drafted);
            const pre_s = secs(io, t0);
            t0 = now(io);
            var rounds: usize = 0;
            if (drafted) {
                const r = try dec.mtpDecode(&e, first, count, o.depth, true, &out, gpa);
                rounds = r.rounds;
            } else try dec.serial(&e, first, count, true, &out, gpa);
            const dec_s = secs(io, t0);
            var same = out.items.len == count;
            var at: usize = 0;
            for (out.items, 0..) |tok, j| {
                if (j >= ref.len or tok != @as(u32, @intCast(ref[j].integer))) {
                    same = false;
                    at = j;
                    break;
                }
            }
            if (!same) failed += 1;
            if (o.rank == 0) std.debug.print("{s} {s}: {d} prompt tokens in {d:.2} s, {d} tokens in {d:.2} s ({d:.1} tok/s{s}), {s}{s}\n", .{
                if (same) "PASS" else "FAIL", name, ids.len, pre_s, out.items.len, dec_s,
                @as(f64, @floatFromInt(out.items.len)) / @max(dec_s, 1e-9), if (drafted) try std.fmt.allocPrint(arena, ", {d} rounds, MTP", .{rounds}) else "",
                if (same) "equal to Python" else try std.fmt.allocPrint(arena, "first difference at {d}", .{at}),
                if (same) "" else try std.fmt.allocPrint(arena, " (got {any})", .{out.items[0..@min(out.items.len, at + 4)]}),
            });
        }
    }
    return if (failed == 0) 0 else 1;
}

/// The quantization of a few tensors of one rank, on one GPU without NCCL (compared with v1.8's by qdump.py).
fn qdump(init: std.process.Init, o: Opts) !u8 {
    const gpa = init.gpa;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var k = try kern.Kernels.load(&driver);
    defer k.deinit();
    const cfg_path = try std.fs.path.join(gpa, &.{ o.model, "config.json" });
    defer gpa.free(cfg_path);
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(init.io, cfg_path, gpa, .limited(1 << 22));
    defer gpa.free(cfg_text);
    const cfg = try glm_config.parse(gpa, cfg_text);
    try wt.dumpQuant(gpa, init.io, &driver, .{ .d = &driver, .s = stream }, &k, o.model, cfg, o.rank, o.out, o.only);
    return 0;
}

/// Rank 1 of a served GLM: load this rank, then run rank 0's commands until it stops.
fn follow(init: std.process.Init, o: Opts, kernels_dir: []const u8) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const n = try node.Node.init(init.gpa, init.io, &ctx, o.model, kernels_dir, o.context, 1, o.master, o.port);
    defer n.deinit();
    std.debug.print("[tf-glm] rank 1 ready: following rank 0 at {s}:{d}\n", .{ o.master, o.port });
    try n.follow();
    std.debug.print("[tf-glm] rank 0 stopped\n", .{});
    return 0;
}

/// `name` is one of the comma-separated names in `list`.
fn listed(list: []const u8, name: []const u8) bool {
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn secs(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e9;
}
