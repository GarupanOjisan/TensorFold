//! The device tables of a shared verify window's DSA rows (v1.10 segments.SegRows); notes/glm_segments.md#overview
const std = @import("std");

/// segments.MAX_SEGS: lanes a window holds at most.
pub const max_segs = 8;
/// fwd.Engine.shared_rows: a shared window's rows at most (v1.10's TF_GLM_MULTI_WINDOW default).
pub const max_rows = 32;
/// sparse.POOL: tokens a pooled key covers.
pub const pool = 4;
/// fwd.extent (segments.EXTENT): a lane's arena extent starts at a multiple of this many tokens.
pub const extent = 2048;
/// SegRows.NO_RING: the kernels' ring size without rings (positions stay below it).
pub const no_ring: i32 = std.math.maxInt(i32);

/// One lane's rows of a window: its extent's first arena row, its first row's position, and its row count.
pub const Span = struct { base: usize, pos: usize, rows: usize };

/// SegRows' one int32 table: five per-row columns, then four per-segment ones; notes/glm_segments.md#table
pub const SegRows = struct {
    pub const words = 5 * max_rows + 4 * max_segs;
    pub const pos_at = 0;
    pub const base_at = max_rows;
    pub const seg_at = 2 * max_rows;
    pub const sparse_at = 3 * max_rows;
    pub const ibase_at = 4 * max_rows;
    pub const start_at = 5 * max_rows;
    pub const rows_at = 5 * max_rows + max_segs;
    pub const pbase_at = 5 * max_rows + 2 * max_segs;
    pub const pools_at = 5 * max_rows + 3 * max_segs;

    /// SegRows.fill: the host table of `spans` into `t`, returning the window's rows; notes/glm_segments.md#fill
    pub fn fill(spans: []const Span, sparse_from: ?usize, t: *[words]i32) !usize {
        if (spans.len < 1 or spans.len > max_segs) return error.BadSegmentCount;
        var total: usize = 0;
        for (spans) |s| total += s.rows;
        if (total > max_rows) return error.WindowTooWide;
        @memset(t, 0);
        var r: usize = 0;
        for (spans, 0..) |s, i| {
            if (s.rows < 1 or s.base % extent != 0) return error.BadSegment;
            for (0..s.rows) |j| {
                const q = s.pos + j;
                t[pos_at + r + j] = @intCast(q);
                t[base_at + r + j] = @intCast(s.base);
                t[seg_at + r + j] = @intCast(i);
                t[sparse_at + r + j] = @intFromBool(if (sparse_from) |f| q >= f else false);
                t[ibase_at + r + j] = @intCast(s.base);
            }
            const last = s.pos + s.rows - 1;
            t[start_at + i] = @intCast(r);
            t[rows_at + i] = @intCast(s.rows);
            t[pbase_at + i] = @intCast(s.base / pool);
            t[pools_at + i] = if (sparse_from) |f| (if (last >= f) @intCast((last + 1) / pool) else 0) else 0;
            r += s.rows;
        }
        return total;
    }

    /// Whether any row of the table attends sparsely (SegRows.any_sparse).
    pub fn anySparse(t: *const [words]i32, rows: usize) bool {
        for (t[sparse_at..][0..rows]) |v| if (v != 0) return true;
        return false;
    }
};

test "SegRows lays two lanes back to back as segments.py does" {
    var t: [SegRows.words]i32 = undefined;
    const r = try SegRows.fill(&.{ .{ .base = 0, .pos = 2049, .rows = 3 }, .{ .base = 4096, .pos = 10, .rows = 2 } }, 2051, &t);
    try std.testing.expectEqual(@as(usize, 5), r);
    try std.testing.expectEqualSlices(i32, &.{ 2049, 2050, 2051, 10, 11 }, t[SegRows.pos_at..][0..5]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 4096, 4096 }, t[SegRows.base_at..][0..5]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1, 1 }, t[SegRows.seg_at..][0..5]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 1, 0, 0 }, t[SegRows.sparse_at..][0..5]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 3 }, t[SegRows.start_at..][0..2]);
    try std.testing.expectEqualSlices(i32, &.{ 3, 2, 0 }, t[SegRows.rows_at..][0..3]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1024 }, t[SegRows.pbase_at..][0..2]);
    try std.testing.expectEqualSlices(i32, &.{ 513, 0 }, t[SegRows.pools_at..][0..2]); // (2051 + 1) / 4; lane 1 dense
    try std.testing.expect(SegRows.anySparse(&t, r));
    try std.testing.expectError(error.BadSegment, SegRows.fill(&.{.{ .base = 100, .pos = 0, .rows = 1 }}, null, &t));
    try std.testing.expectError(error.WindowTooWide, SegRows.fill(&.{ .{ .base = 0, .pos = 0, .rows = 17 }, .{ .base = 2048, .pos = 0, .rows = 16 } }, null, &t));
}
