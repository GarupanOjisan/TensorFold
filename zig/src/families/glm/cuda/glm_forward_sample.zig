//! The forward engine's MTP step and sampling (greedy, candidates, nucleus draws).
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const kern = @import("glm_kernels.zig");
const fwd = @import("glm_forward.zig");
const buffers = @import("glm_forward_buffers.zig");

const Engine = fwd.Engine;
const bf = fwd.bf;
const Buffers = buffers.Buffers;
const max_rows = Engine.max_rows;
const shared_rows = Engine.shared_rows;
const max_k = Engine.max_k;
const max_nucleus = Engine.max_nucleus;

// -- MTP ----------------------------------------------------------------------------------------------------
/// mtp_stage + mtp_compute: the head on rows (next token, hidden), cache slots mtp_len.. onward; the last row's
/// draft-head logits in b.logits. The caller advances mtp_len.
pub fn mtp(e: *Engine, b: *Buffers, next: []const u32, hidden: u64) !void {
    const c = e.w.cfg;
    const d: usize = c.hidden;
    if (e.w.mtp == null) return error.NoMtpHead;
    const n = next.len;
    if (n > b.rows) return error.WindowTooWide;
    b.zero_first = e.st.mtp_len == 0;
    try e.checkRoom(n, e.st.mtp_len);
    try e.upload(b.ids, std.mem.sliceAsBytes(next));
    if (hidden != b.hin) try e.copyDev(b.hin, hidden, n * d * bf);
    try e.t.embed(b.ids, e.w.embed, b.me, n, d, 1);
    if (b.zero_first) try e.zero(b.me, d * bf);
    if (!b.prefill and !b.zero_first) {
        if (e.graphKey(.mtp, n, e.st.mtp_len)) |key| return e.replay(key, mtpBody, .{ e, b, n });
    }
    try e.mtpBody(b, n);
}

/// The MTP step after its inputs are staged and embedded (b.me, b.hin).
pub fn mtpBody(e: *Engine, b: *Buffers, n: usize) !void {
    const c = e.w.cfg;
    const d: usize = c.hidden;
    const m = e.w.mtp.?;
    try e.t.rmsnorm(b.me, d, m.enorm, b.mcat, 2 * d, null, n, d, c.eps);
    try e.t.rmsnorm(b.hin, d, m.hnorm, b.mcat + d * bf, 2 * d, null, n, d, c.eps);
    if (!b.prefill) try e.t.groupSums(b.mcat, 2 * d, b.mxs, n, 2 * d);
    try e.mm(b, b.mcat, 2 * d, b.mxs, m.eh, b.mx, n, false);
    const l = &m.layer;
    try e.t.rmsnorm(b.mx, d, l.in_norm, b.normed, d, b.xs, n, d, c.eps);
    const ix: ?[3]u64 = if (e.st.long_context) e.st.index[e.st.dsa_layers] else null;
    const g = try e.dsaBlock(l, b, e.st.mtp_kc, e.st.mtp_pos_dev, e.st.mtp_len, ix, n, null);
    try e.t.residualAdd(b.mx, b.mx, g, wt.world, n, d);
    try e.t.rmsnorm(b.mx, d, l.post_norm, b.normed, d, b.xs, n, d, c.eps);
    const g2 = try e.moeBlock(l.moe.?, b, n, l.index, null);
    try e.t.residualAdd(b.mx, b.mx, g2, wt.world, n, d);
    try e.t.rmsnorm(b.mx + (n - 1) * d * bf, d, m.head_norm, b.fnormed, d, b.fxs, 1, d, c.eps);
    try e.draftHead(b, b.fnormed, 1);
}

/// The MTP head's logits from the 4-bit draft head (mtp.py: `w.draft_head`, forward.mm): a prompt chunk's on the
/// prefill matmul, a decode step's on qmm_group with the rows' group sums (b.fxs, from the final norm).
pub fn draftHead(e: *Engine, b: *const Buffers, x: u64, rows: usize) !void {
    if (b.prefill) return kern.q4Prefill(e.k, e.s, x, e.draft_head.k, e.draft_head, b.logits, rows, false);
    try kern.q4Matmul(e.k, e.s, x, b.fxs, e.draft_head, b.logits, rows);
}

// -- sampling -----------------------------------------------------------------------------------------------
/// sample_rows' candidates for sampled rows: each rank's top `k` of its vocabulary slice (k = top_k + MARGIN),
/// gathered on every rank into cand_host, for the host chooser.
pub fn candidates(e: *Engine, b: *const Buffers, logits: u64, rows: usize, k: usize) !void {
    if (k == 0 or k > max_k or k > e.w.head.n or rows > shared_rows) return error.BadCandidateCount;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i32, @intCast(e.w.head.n)));
    a.add(@as(i32, @intCast(e.w.head.n)));
    a.add(@as(i32, @intCast(k)));
    a.add(@as(i32, @intCast(e.w.rank * e.w.head.n)));
    a.add(b.cand);
    try cuda.launch.launch(e.k.topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.s, &a);
    const n = rows * 2 * k;
    try e.comm.lib.check(e.comm.lib.api.ncclAllGather(b.cand, b.cand_all, n, .u32, e.comm.comm, e.s.handle), "ncclAllGather");
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(&e.cand_host, b.cand_all, wt.world * n * 4, e.s.handle), "cuMemcpyDtoHAsync");
    try e.s.synchronize();
}

/// sampling.nucleus_rows' `_shares` for `rows` rows of `logits` (this rank's half of the vocabulary, bf16): every rank's top
/// `count` (value, id, mass) per row and its shard's mass total and width, as 64-bit words [rank][row][count float64 values,
/// count ids, count masses, sum, width], gathered on every rank. The row maxima come from `greedy`'s gather.
pub fn nucleusGather(e: *Engine, logits: u64, rows: usize, temperature: f64, count: usize) ![]const u64 {
    if (rows == 0 or rows > max_rows or count == 0 or count > max_nucleus) return error.BadNucleus;
    const world = wt.world;
    const vn = e.w.head.n;
    const per_row = 3 * max_nucleus + 2;
    // [sums: max_rows words][candidate pairs: max_rows * 2 * max_nucleus u32][packed: max_rows * per_row words][gathered]
    const cand_off = max_rows * 8;
    const loc_off = cand_off + max_rows * 2 * max_nucleus * 4;
    const all_off = loc_off + max_rows * per_row * 8;
    if (e.nuc_dev == null) {
        e.nuc_dev = try cuda.DeviceBuffer.alloc(e.d, all_off + world * max_rows * per_row * 8);
        e.nuc_host = try std.heap.page_allocator.alloc(u64, world * max_rows * per_row);
    }
    const base = e.nuc_dev.?.ptr;
    const inv_t: f64 = 1.0 / @max(temperature, 1e-6);
    const b = &e.buf;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i32, @intCast(vn)));
    a.add(@as(i32, @intCast(vn)));
    a.add(@as(i32, @intCast(e.w.rank * vn)));
    a.add(b.greedy);
    try cuda.launch.launch(e.k.greedy, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.s, &a);
    try e.comm.lib.check(e.comm.lib.api.ncclAllGather(b.greedy, b.greedy_all, rows * 2, .u32, e.comm.comm, e.s.handle), "ncclAllGather");
    var st: cuda.Args = .{};
    st.add(logits);
    st.add(@as(i32, @intCast(vn)));
    st.add(@as(i32, @intCast(vn)));
    st.add(inv_t);
    st.add(b.greedy_all);
    st.add(@as(i32, world));
    st.add(@as(i32, @intCast(rows)));
    st.add(base);
    try cuda.launch.launch(e.k.nuc_stats, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.s, &st);
    var tk: cuda.Args = .{};
    tk.add(logits);
    tk.add(@as(i32, @intCast(vn)));
    tk.add(@as(i32, @intCast(vn)));
    tk.add(@as(i32, @intCast(count)));
    tk.add(@as(i32, @intCast(e.w.rank * vn)));
    tk.add(base + cand_off);
    try cuda.launch.launch(e.k.topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.s, &tk);
    const words = rows * (3 * count + 2);
    var pk: cuda.Args = .{};
    pk.add(base + cand_off);
    pk.add(@as(i32, @intCast(count)));
    pk.add(inv_t);
    pk.add(b.greedy_all);
    pk.add(@as(i32, world));
    pk.add(@as(i32, @intCast(rows)));
    pk.add(base);
    pk.add(@as(i32, @intCast(vn)));
    pk.add(base + loc_off);
    try cuda.launch.launch(e.k.nuc_pack, .{ .grid = .{ .x = @intCast((count + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, e.s, &pk);
    try e.comm.lib.check(e.comm.lib.api.ncclAllGather(base + loc_off, base + all_off, words * 2, .u32, e.comm.comm, e.s.handle), "ncclAllGather");
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(e.nuc_host.ptr, base + all_off, world * words * 8, e.s.handle), "cuMemcpyDtoHAsync");
    try e.s.synchronize();
    return e.nuc_host[0 .. world * words];
}

/// nucleus_rows' gather of whole shards: every rank's `rows` logits rows into full_host, [rank][row][head.n].
pub fn fullRows(e: *Engine, logits: u64, rows: usize) ![]const u16 {
    if (rows > max_rows) return error.WindowTooWide;
    const n = rows * e.w.head.n;
    try e.comm.lib.check(e.comm.lib.api.ncclAllGather(logits, e.full_dev.ptr, n, .bf16, e.comm.comm, e.s.handle), "ncclAllGather");
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(e.full_host.ptr, e.full_dev.ptr, wt.world * n * bf, e.s.handle), "cuMemcpyDtoHAsync");
    try e.s.synchronize();
    return e.full_host[0 .. wt.world * n];
}

/// sample_rows, greedy: each rank's row maximum (first column on ties), gathered; the larger value wins, the
/// smaller token on a tie (np.lexsort((tokens, -values))).
pub fn greedy(e: *Engine, b: *const Buffers, logits: u64, rows: usize, vocab: usize, out: []u32) !void {
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i32, @intCast(vocab)));
    a.add(@as(i32, @intCast(e.w.head.n)));
    a.add(@as(i32, @intCast(e.w.rank * e.w.head.n)));
    a.add(b.greedy);
    try cuda.launch.launch(e.k.greedy, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.s, &a);
    try e.comm.allGather(e.s, b.greedy, b.greedy_all, rows * 2);
    var host: [wt.world * shared_rows * 2]u32 = undefined;
    if (rows > shared_rows) return error.WindowTooWide;
    const n = wt.world * rows * 2;
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(&host, b.greedy_all, n * 4, e.s.handle), "cuMemcpyDtoHAsync");
    try e.s.synchronize();
    if (e.comm.roce) |r| try r.check();
    for (0..rows) |r| {
        var best_v: f32 = -std.math.inf(f32);
        var best_t: u32 = std.math.maxInt(u32);
        for (0..wt.world) |k| {
            const v: f32 = @bitCast(host[(k * rows + r) * 2]);
            const t = host[(k * rows + r) * 2 + 1];
            if (v > best_v or (v == best_v and t < best_t)) {
                best_v = v;
                best_t = t;
            }
        }
        out[r] = best_t;
    }
}
