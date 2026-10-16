//! GLM's KDA chain, segment and replay launches, the hyper-connection dots and the L2 prefetch.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("glm_kernels.zig");
const Kernels = kern.Kernels;
const go = kern.go;
const int = kern.int;
const u = kern.u;
const kda_chunk_smem = kern.kda_chunk_smem;

/// One KDA layer's chain (kda.chain): the conv, gates and delta rule over `rows` rows from `state_in`, the new state
/// into `state_out` (0: none) and each row's k, v, g, b saved for a later replay. Element strides.
pub const Chain = struct {
    heads: usize,
    p: u64,
    p_stride: usize,
    b_off: usize,
    a: u64,
    a_stride: usize,
    g: u64,
    g_stride: usize,
    conv_state: u64,
    conv_w: u64,
    state_in: u64,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    eps: f32,
    lower: f32,
    rows: usize,
    out: u64,
    state_out: u64,
    k_save: u64,
    v_save: u64,
    g_save: u64,
    b_save: u64,
    q_tmp: u64 = 0, // the wide path's scratch (rows >= 64)
    y_tmp: u64 = 0,
};

// -- several lanes' windows in one launch (v1.8 kda.py:116-178) ----------------------------------------------------
/// kda.SEG_COLS: a segment table's row, int32 (first row, rows, state slot, parity, conv slot, keep).
pub const seg_cols = 6;
pub const SegRow = extern struct { row0: i32, rows: i32, slot: i32, parity: i32, conv: i32, keep: i32 };

/// kda.segment_table: `segs` (rows, state slot, parity, conv slot, keep) back to back from row 0 into `out`, the rows past
/// them empty segments (0 rows), so a table sized for the most lanes serves fewer.
pub fn segmentTable(segs: []const SegRow, out: []SegRow) !void {
    if (segs.len > out.len) return error.TooManySegments;
    var row0: i32 = 0;
    for (segs, out[0..segs.len]) |sg, *o| {
        if (sg.rows < 0 or sg.parity < 0 or sg.parity > 1 or sg.keep < 0 or sg.keep > sg.rows) return error.BadSegment;
        o.* = sg;
        o.row0 = row0;
        row0 += sg.rows;
    }
    for (out[segs.len..]) |*o| o.* = .{ .row0 = row0, .rows = 0, .slot = 0, .parity = 0, .conv = 0, .keep = 0 };
}

test "segment tables lay the segments back to back and pad with empty ones" {
    var out: [4]SegRow = undefined;
    try segmentTable(&.{ .{ .row0 = 9, .rows = 3, .slot = 2, .parity = 1, .conv = 2, .keep = 3 }, .{ .row0 = 0, .rows = 5, .slot = 0, .parity = 0, .conv = 0, .keep = 2 } }, &out);
    try std.testing.expectEqual(@as(i32, 0), out[0].row0);
    try std.testing.expectEqual(@as(i32, 3), out[1].row0);
    try std.testing.expectEqual(@as(i32, 8), out[2].row0);
    try std.testing.expectEqual(@as(i32, 0), out[3].rows);
    try std.testing.expectError(error.BadSegment, segmentTable(&.{.{ .row0 = 0, .rows = 2, .slot = 0, .parity = 2, .conv = 0, .keep = 2 }}, &out));
    const one = SegRow{ .row0 = 0, .rows = 1, .slot = 0, .parity = 0, .conv = 0, .keep = 1 };
    const five: [5]SegRow = @splat(one);
    try std.testing.expectError(error.TooManySegments, segmentTable(&five, &out));
}

/// kda.chain_segments' arguments: Chain's, with the lanes' states stacked by slot (strides in elements, as kda.cpp passes
/// tensor strides): rec [slots, 2, layers, H, 128, 128] f32 at this layer (`rec` points at slot 0, parity 0, this layer),
/// conv [slots, layers, 3, 3 H 128] bf16 at this layer; `seg` the device table of `nseg` rows.
pub const ChainSegments = struct {
    heads: usize,
    seg: u64,
    nseg: usize,
    p: u64,
    p_stride: usize,
    b_off: usize,
    a: u64,
    a_stride: usize,
    g: u64,
    g_stride: usize,
    conv: u64,
    conv_slot_stride: usize,
    conv_w: u64,
    rec: u64,
    slot_stride: usize,
    parity_stride: usize,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    eps: f32,
    lower: f32,
    rows: usize,
    out: u64,
    k_save: u64,
    v_save: u64,
    g_save: u64,
    b_save: u64,
    q_tmp: u64,
    y_tmp: u64,
};

/// kda.chain_segments (kda_chain_wide_segments_cuda, v1.8 kda.cu:604-627): one layer's window cut into the table's
/// segments, each the bits of a solo wide chain of its rows; conv is read only (the commit shifts it).
pub fn chainSegments(k: *const Kernels, s: cuda.Stream, c: ChainSegments) !void {
    var a: cuda.Args = .{};
    a.add(int(c.heads));
    a.add(c.seg);
    a.add(int(c.nseg));
    a.add(c.p);
    a.add(int(c.p_stride));
    a.add(int(c.b_off));
    a.add(c.a);
    a.add(int(c.a_stride));
    a.add(c.conv);
    a.add(@as(u64, c.conv_slot_stride));
    a.add(c.conv_w);
    a.add(c.a_log);
    a.add(c.dt_bias);
    a.add(c.lower);
    a.add(c.q_tmp);
    a.add(c.k_save);
    a.add(c.v_save);
    a.add(c.g_save);
    a.add(c.b_save);
    try go(k.prep_seg, s, .{ c.rows, c.heads, 1 }, 512, 0, &a);
    var b: cuda.Args = .{};
    b.add(int(c.heads));
    b.add(c.seg);
    b.add(c.rec);
    b.add(@as(u64, c.slot_stride));
    b.add(@as(u64, c.parity_stride));
    b.add(c.q_tmp);
    b.add(c.k_save);
    b.add(c.v_save);
    b.add(c.g_save);
    b.add(c.b_save);
    b.add(c.y_tmp);
    try go(k.step_seg, s, .{ c.nseg * c.heads, 128 / 4 / 4, 1 }, 128, 0, &b);
    var o: cuda.Args = .{};
    o.add(int(c.heads));
    o.add(c.y_tmp);
    o.add(c.g);
    o.add(int(c.g_stride));
    o.add(c.norm_w);
    o.add(c.eps);
    o.add(c.out);
    try go(k.out, s, .{ c.rows, c.heads, 1 }, 128, 0, &o);
}

/// kda.replay_layers_segments (v1.8 kda.cu:629-634): each segment with keep < rows, every layer's state after its first
/// `keep` saved rows, rec[slot, parity] into rec[slot, 1 - parity]. rec at slot 0, parity 0, layer 0; strides in
/// elements; the saves [layers, scratch_rows, H, ...] of the window's rows.
pub fn replayLayersSegments(k: *const Kernels, s: cuda.Stream, seg: u64, nseg: usize, layers: usize, heads: usize, rec: u64, slot_stride: usize, parity_stride: usize, k_save: u64, v_save: u64, g_save: u64, b_save: u64, scratch_rows: usize) !void {
    var a: cuda.Args = .{};
    a.add(int(heads));
    a.add(int(layers));
    a.add(seg);
    a.add(rec);
    a.add(@as(u64, slot_stride));
    a.add(@as(u64, parity_stride));
    a.add(@as(u64, heads * 128 * 128));
    a.add(k_save);
    a.add(v_save);
    a.add(g_save);
    a.add(b_save);
    a.add(@as(u64, scratch_rows * heads * 128));
    a.add(@as(u64, scratch_rows * heads));
    try go(k.replay_layers_seg, s, .{ nseg * layers * heads, 1, 1 }, 1024, 0, &a);
}

/// hc.hc_partial (v1.8 hc.cu): decode rows' 24 mixing dots and sum of squares of x [rows, 16384] bf16 against fn [24, 16384]
/// into part [rows, 16, 32] fp32 (glue._hc_partial's bits, 48 blocks a row).
pub fn hcPartial(k: *const Kernels, s: cuda.Stream, x: u64, fn_w: u64, part: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(fn_w);
    a.add(part);
    a.add(int(rows));
    try go(k.hc_partial, s, .{ 16, 3, (rows + 3) / 4 }, 288, 0, &a);
}

/// l2pf.prefetch: `count` (address, bytes) pieces of the table at `table` + 16 `first` into L2 (mode 0: cp.async.bulk.prefetch.L2,
/// 1: prefetch.global.L2::evict_last per line, 2: loads), on the stream given.
pub fn l2Prefetch(k: *const Kernels, s: cuda.Stream, table: u64, first: usize, count: usize, mode: i32, blocks: usize, threads: usize, sink: u64) !void {
    var a: cuda.Args = .{};
    a.add(table + 16 * first);
    a.add(int(count));
    a.add(mode);
    a.add(sink);
    try go(k.l2pf, s, .{ blocks, 1, 1 }, threads, 0, &a);
}

/// kda_chunked.chain: a prompt chunk of `c.rows` rows whose first row sits at absolute position `pos`, in 32-row
/// sub-chunks at absolute positions (TF_GLM_KDA_CHUNKED=1). Writes `c.out` and `c.state_out` only; the conv state is read.
pub fn chainChunked(k: *const Kernels, s: cuda.Stream, c: Chain, pos: usize) !void {
    var a: cuda.Args = .{};
    a.add(int(c.heads));
    a.add(c.p);
    a.add(int(c.p_stride));
    a.add(int(c.b_off));
    a.add(c.a);
    a.add(int(c.a_stride));
    a.add(c.g);
    a.add(int(c.g_stride));
    a.add(c.conv_state);
    a.add(c.conv_w);
    a.add(c.state_in);
    a.add(c.a_log);
    a.add(c.dt_bias);
    a.add(c.norm_w);
    a.add(c.eps);
    a.add(c.lower);
    a.add(int(c.rows));
    a.add(int(pos % 32));
    a.add(c.out);
    a.add(c.state_out);
    try go(k.kda_chunk, s, .{ c.heads, 1, 1 }, 256, kda_chunk_smem, &a);
}

/// kda.replay_layers: every KDA layer's state after its first `rows` kept rows (commit with keep < R).
pub fn replayLayers(k: *const Kernels, s: cuda.Stream, layers: usize, heads: usize, state_in: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64, scratch_rows: usize, rows: usize, state_out: u64) !void {
    var a: cuda.Args = .{};
    a.add(int(heads));
    a.add(state_in);
    a.add(@as(u64, heads * 128 * 128));
    a.add(k_save);
    a.add(v_save);
    a.add(g_save);
    a.add(b_save);
    a.add(@as(u64, scratch_rows * heads * 128));
    a.add(@as(u64, scratch_rows * heads));
    a.add(int(rows));
    a.add(state_out);
    try go(k.replay_layers, s, .{ layers * heads, 1, 1 }, 1024, 0, &a);
}
