//! GLM's 4-bit and e4m3 dense matmul launches (qmm_group, qmm_prefill, v1.8 qmm.cu) and their packing.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("glm_kernels.zig");
const Kernels = kern.Kernels;
const go = kern.go;
const int = kern.int;
const u = kern.u;
const qmm32 = kern.qmm32;
const qmm_cfgs = kern.qmm_cfgs;
const q4_group_smem = kern.q4_group_smem;
const q4_prefill_smem = kern.q4_prefill_smem;

/// glm qmm.py DEC_TILES (TF_GLM_Q4_TILE unset): the config index of a few projection shapes, else 0.
fn decCfg(n: usize, kk: usize) usize {
    const table = [_][3]usize{ .{ 4096, 4096, 1 }, .{ 2048, 4096, 1 }, .{ 8192, 1536, 1 }, .{ 4096, 8192, 2 }, .{ 4096, 1024, 1 } };
    for (table) |t| if (t[0] == n and t[1] == kk) return t[2];
    return 0;
}

/// qmm_group.cu's Part and Parts, passed by value.
const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
const Parts = extern struct { p: [4]Part, count: c_int };

/// A tiled 4-bit projection (qmm.Q4): words [npad/64][k/64][8][32][2], (k/64, npad) bf16 scales and biases.
pub const Q4 = struct { w: u64, s: u64, b: u64, n: usize, k: usize, npad: usize };

pub fn q4Bytes(n: usize, k: usize) usize {
    const npad = (n + 127) / 128 * 128;
    return npad * k / 2 + 2 * (k / 64) * npad * 2 + 512;
}

/// Which ranges a group's 16 levels span: its min and max (qmm.quantize4, the draft head's), or the least-squares one
/// of 16 clippings (qmm.quantize4_mse, TF_GLM_DENSE=q4's projections).
pub const Quant = enum { minmax, mse };

/// qmm.quantize4 or quantize4_mse, then qmm.make_q4, of a bf16 (n, k) matrix into `at` (q4Bytes), with `tmp` for the
/// MLX arrays (n k / 2 + 4 n k / 64 bytes).
pub fn quantize4(k: *const Kernels, s: cuda.Stream, kind: Quant, w: u64, n: usize, kk: usize, at: u64, tmp: u64) !Q4 {
    const kg = kk / 64;
    const npad = (n + 127) / 128 * 128;
    const mw = tmp;
    const ms = tmp + n * kk / 2;
    const mb = ms + n * kg * 2;
    var a: cuda.Args = .{};
    a.add(w);
    a.add(int(n));
    a.add(int(kk));
    a.add(mw);
    a.add(ms);
    a.add(mb);
    const threads: usize = if (kind == .mse) 64 else 256; // the MSE search keeps a group's 64 values in registers
    try go(if (kind == .mse) k.quantize4_mse else k.quantize4, s, .{ (n * kg + threads - 1) / threads, 1, 1 }, threads, 0, &a);
    const q: Q4 = .{ .w = at, .s = at + npad * kk / 2, .b = at + npad * kk / 2 + kg * npad * 2, .n = n, .k = kk, .npad = npad };
    const total: u64 = @as(u64, npad / 64) * kg * 512;
    var p: cuda.Args = .{};
    p.add(mw);
    for ([_]usize{ n, kk / 8, kg }) |v| p.add(int(v));
    p.add(q.w);
    p.add(@as(c_longlong, @intCast(total)));
    try go(k.pack_dense, s, .{ @intCast((total + 255) / 256), 1, 1 }, 256, 0, &p);
    for ([_]u64{ ms, mb }, [_]u64{ q.s, q.b }) |src, out| {
        var t: cuda.Args = .{};
        t.add(src);
        for ([_]usize{ n, kg, npad }) |v| t.add(int(v));
        t.add(out);
        try go(k.transpose16, s, .{ (kg * npad + 255) / 256, 1, 1 }, 256, 0, &t);
    }
    return q;
}

/// An e4m3 matrix (qmm.F8): values [n, k] and an fp32 scale per row and 128 columns.
pub const F8 = struct { w: u64, s: u64, n: usize, k: usize };

/// qmm.make_f8 of a bf16 (n, k) matrix (k a multiple of 128) into `w` (n k bytes) and `s` (n k / 32 bytes).
pub fn makeF8(k: *const Kernels, s: cuda.Stream, src: u64, n: usize, kk: usize, w: u64, scale: u64) !F8 {
    if (kk % 128 != 0) return error.F8Width;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(int(n));
    a.add(int(kk));
    a.add(w);
    a.add(scale);
    try go(k.make_f8, s, .{ (n * kk / 128 + 63) / 64, 1, 1 }, 64, 0, &a);
    return .{ .w = w, .s = scale, .n = n, .k = kk };
}

/// glm qmm.split_k: the K slices of a 4-bit (n, k) matmul, from SHAPE_SK or else the generic rule (q4SplitK).
pub fn denseSplitK(n: usize, kk: usize) usize {
    const table = [_][3]usize{ .{ 12576, 4096, 4 }, .{ 4096, 4096, 2 }, .{ 2048, 4096, 4 }, .{ 8192, 1536, 4 }, .{ 8192, 512, 4 }, .{ 4096, 8192, 8 } };
    for (table) |t| if (t[0] == n and t[1] == kk) return t[2];
    return q4SplitK(n, kk);
}

/// qmm.matmul of 17 to 32 rows (a shared window): `qmm` on the 32-row tile, K slices as qmmDecode's (denseSplitK), so each
/// row's bits are those of its lane's own window on a 16-row tile.
pub fn qmmShared(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, xs: u64, q: Q4, out: u64, rows: usize, fp32: bool) !void {
    if (rows <= 16 or rows > 32) return error.WindowTooWide;
    if (x % 16 != 0 or x_stride % 8 != 0) return error.MisalignedInput;
    const sk = denseSplitK(q.n, q.k);
    const tiles = (q.n + qmm32.bn - 1) / qmm32.bn;
    var a: cuda.Args = .{};
    for ([_]u64{ x, xs, q.w, q.s, q.b, out, 0 }) |v| a.add(v);
    for ([_]usize{ rows, q.n, q.k, sk, q.npad, x_stride, 1 }) |v| a.add(int(v));
    try cuda.launch.launch(k.qmm32[@intFromBool(fp32)][@intFromBool(sk > 1)], .{
        .grid = .{ .x = u(tiles), .y = 1, .z = u(sk) },
        .block = .{ .x = qmm32.wn * 32 },
        .shared = qmm32.smem(),
        .cluster = if (sk > 1) .{ .x = 1, .y = 1, .z = u(sk) } else null,
    }, s, &a);
}

/// qmm.matmul of a 4-bit q for a decode window (shared.matmul -> `qmm` or, with dec_tile, `qmm_cfg`: v1.8's qmm_kernel):
/// x (rows <= 16, k) bf16, rows `x_stride` elements apart, with its group sums xs -> out (rows, n), fp32 with `f32`.
pub fn qmmDecode(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, xs: u64, q: Q4, out: u64, rows: usize, fp32: bool) !void {
    if (rows > 16) return error.WindowTooWide;
    if (x % 16 != 0 or (rows > 1 and x_stride % 8 != 0)) return error.MisalignedInput; // Python would clone x (cp.async reads 16 bytes)
    const sk = denseSplitK(q.n, q.k);
    const cfg = decCfg(q.n, q.k);
    const c = qmm_cfgs[cfg];
    const tiles = (q.n + c.bn - 1) / c.bn;
    var a: cuda.Args = .{};
    for ([_]u64{ x, xs, q.w, q.s, q.b, out, 0 }) |v| a.add(v); // part: unused, the K slices sum in a cluster
    // M, N, K, SK, npad (the scales' row stride), ldx, group (one row tile: one)
    for ([_]usize{ rows, q.n, q.k, sk, q.npad, if (rows == 1) q.k else x_stride, 1 }) |v| a.add(int(v));
    try cuda.launch.launch(k.qmm[cfg][@intFromBool(fp32)][@intFromBool(sk > 1)], .{
        .grid = .{ .x = u(tiles), .y = 1, .z = u(sk) },
        .block = .{ .x = c.wn * 32 },
        .shared = c.smem(),
        .cluster = if (sk > 1) .{ .x = 1, .y = 1, .z = u(sk) } else null,
    }, s, &a);
}

/// qmm.matmul of a 4-bit q on sm_12x (shared.matmul -> qmm_group, the kernel's own tile): x (rows, k) bf16 with
/// its group sums xs -> out (rows, n) bf16.
pub fn q4Matmul(k: *const Kernels, s: cuda.Stream, x: u64, xs: u64, q: Q4, out: u64, rows: usize) !void {
    if (rows > 16) return error.WindowTooWide;
    const sk = q4SplitK(q.n, q.k);
    const tiles = (q.n + 63) / 64;
    var parts: Parts = std.mem.zeroes(Parts);
    parts.count = 1;
    parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
    const rows_t = (rows + 15) / 16;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(xs);
    a.add(parts);
    for ([_]usize{ rows, q.k, q.k, rows_t, sk }) |v| a.add(int(v));
    try cuda.launch.launch(k.q4_group, .{
        .grid = .{ .x = u(rows_t * tiles * sk) },
        .block = .{ .x = 128 },
        .shared = q4_group_smem,
        .cluster = if (sk > 1) .{ .x = u(sk) } else null,
    }, s, &a);
}

/// shared.prefill_matmul (a prompt chunk's 4-bit projection): x rows `x_stride` elements apart, out bf16 or fp32.
pub fn q4Prefill(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, q: Q4, out: u64, rows: usize, fp32: bool) !void {
    if (x % 16 != 0 or (rows > 1 and x_stride % 8 != 0)) return error.MisalignedInput;
    const rows_t = (rows + 127) / 128;
    const band = (12 << 20) / (128 * q.k * 2);
    const group = @max(1, @min(rows_t, band));
    var a: cuda.Args = .{};
    for ([_]u64{ x, q.w, q.s, q.b, out }) |v| a.add(v);
    for ([_]usize{ rows, q.n, q.k, q.npad, if (rows == 1) q.k else x_stride, group }) |v| a.add(int(v));
    try go(if (fp32) k.q4_prefill_f32 else k.q4_prefill, s, .{ rows_t * ((q.n + 127) / 128), 1, 1 }, 256, q4_prefill_smem, &a);
}

/// glm qmm.split_k for shapes outside its SHAPE_SK table (the draft head's): K slices until column tiles times slices reach 192.
pub fn q4SplitK(n: usize, kk: usize) usize {
    const tiles = (n + 63) / 64;
    const groups = kk / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

test "the draft head's shape takes one K slice" {
    try std.testing.expectEqual(@as(usize, 1), q4SplitK(77440, 4096));
}

test "the dense 4-bit matmuls' K slices, tile configs and shared memory" {
    const t = std.testing;
    // every per-rank shape of TF_GLM_DENSE=q4: n, k, slices (SHAPE_SK, else the generic rule), config (DEC_TILES)
    const shapes = [_][4]usize{
        .{ 12576, 4096, 4, 0 }, .{ 4096, 128, 1, 0 },  .{ 4096, 4096, 2, 1 }, .{ 12288, 4096, 1, 0 }, .{ 4096, 6144, 4, 0 },
        .{ 2048, 4096, 4, 1 },  .{ 8192, 1536, 4, 1 }, .{ 4096, 8192, 8, 2 }, .{ 160, 4096, 8, 0 },   .{ 4096, 1536, 2, 0 },
        .{ 4096, 1024, 2, 1 },
    };
    for (shapes) |s| {
        try t.expectEqual(s[2], denseSplitK(s[0], s[1]));
        try t.expectEqual(s[3], decCfg(s[0], s[1]));
    }
    try t.expectEqual(@as(u32, 17664), qmm_cfgs[0].smem()); // Tile<64, 16, 64, 1, 4, 4>
    try t.expectEqual(@as(u32, 26112), qmm32.smem()); // Tile<64, 32, 64, 1, 4, 4>
    try t.expectEqual(@as(u32, 26880), qmm_cfgs[1].smem()); // Tile<64, 16, 128, 1, 8, 4>
    try t.expectEqual(@as(u32, 40320), qmm_cfgs[2].smem()); // Tile<64, 16, 128, 1, 4, 6>
}
