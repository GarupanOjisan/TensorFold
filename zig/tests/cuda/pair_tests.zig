//! Two ranks over NCCL: the unique id over the TCP pair, then an all-gather of GLM's per-layer size inside a CUDA graph.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const expect = check.expect;

/// GLM's attention or FFN partial for an 8-row window: [rows, 4096] fp32 a rank.
const rows = 1;
/// A decode round's gathers: two a layer over 45 layers, plus the sampler's.
const per_graph = 92;
const hidden = 4096;

fn pattern(rank: usize, i: usize) f32 {
    return @floatFromInt((rank + 1) * 1_000_003 + i * 7);
}

/// `nccl-pair <rank> <rank 0's address> <port> [reps]`: run on both Sparks, rank 0 first or second.
pub fn ncclPair(gpu: Gpu, rank: usize, host: []const u8, port: u16, reps: usize) !void {
    const d = gpu.d;
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    var pair = if (rank == 0) try cuda.Pair.listen(port) else try cuda.Pair.connect(host, port, 120);
    defer pair.close();
    var id: cuda.nccl.UniqueId = undefined;
    if (rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        try pair.send(&id.internal);
    } else try pair.recv(&id.internal);
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, 2, id, @intCast(rank)), "ncclCommInitRank");
    defer _ = lib.api.ncclCommDestroy(comm);

    const n = rows * hidden;
    const mine = try gpu.gpa.alloc(f32, n);
    defer gpu.gpa.free(mine);
    for (mine, 0..) |*v, i| v.* = pattern(rank, i);
    var send = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(mine));
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(d, 2 * n * 4);
    defer recv.free();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();

    // eager, timed, then captured into a graph and replayed
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();
    try lib.check(lib.api.ncclAllGather(send.ptr, recv.ptr, n, .f32, comm, stream.handle), "ncclAllGather");
    try stream.synchronize();
    try e0.record(stream);
    for (0..reps) |_| try lib.check(lib.api.ncclAllGather(send.ptr, recv.ptr, n, .f32, comm, stream.handle), "ncclAllGather");
    try e1.record(stream);
    try stream.synchronize();
    const eager_ms = try cuda.Event.elapsedMs(e0, e1);
    try cuda.graph.beginCapture(stream, .global);
    for (0..per_graph) |_| try lib.check(lib.api.ncclAllGather(send.ptr, recv.ptr, n, .f32, comm, stream.handle), "ncclAllGather (capture)");
    var g = try cuda.graph.endCapture(stream);
    defer g.deinit();
    var exec = try g.instantiate();
    defer exec.deinit();
    try recv.fill8(0, null);
    var start = try cuda.Event.init(d, true);
    defer start.deinit();
    var end = try cuda.Event.init(d, true);
    defer end.deinit();
    try exec.launchOn(stream);
    try start.record(stream);
    for (0..reps) |_| try exec.launchOn(stream);
    try end.record(stream);
    try stream.synchronize();
    const ms = try cuda.Event.elapsedMs(start, end);

    const got_bytes = try check.download(gpu, recv);
    defer gpu.gpa.free(got_bytes);
    const got = std.mem.bytesAsSlice(f32, got_bytes);
    for (0..2) |r| for (0..n) |i| {
        try expect(@as(u32, @bitCast(got[r * n + i])) == @as(u32, @bitCast(pattern(r, i))), "rank {d} element {d}: {d} != {d}", .{ r, i, got[r * n + i], pattern(r, i) });
    };
    check.pass("NCCL {d}: two ranks, all-gather of {d}x{d} fp32 in a CUDA graph exact, {d:.1} us a gather in a graph of {d:.0}, {d:.1} us eager, over {d}", 
        .{ try lib.version(), rows, hidden, ms * 1000 / @as(f32, @floatFromInt(reps * per_graph)), per_graph, eager_ms * 1000 / @as(f32, @floatFromInt(reps)), reps });
}
