//! GLM's raw CUDA kernels (the glm_kda, glm_exl3 and glm_glue fatbins) launched with the Python host code's geometry
//! (glm_symbols.zig has the spec, with file:line references into v0.6.6).
const std = @import("std");
const cuda = @import("cuda");
const sym = @import("glm_symbols.zig");

fn u(x: usize) u32 {
    return @intCast(x);
}

fn int(x: usize) i32 {
    return @intCast(x);
}

pub const Kernels = struct {
    mods: [7]cuda.Module,
    chain: cuda.Function,
    prep: cuda.Function,
    step: cuda.Function,
    out: cuda.Function,
    replay_layers: cuda.Function,
    group: cuda.Function,
    rot_in: cuda.Function,
    grouped: cuda.Function,
    gateup_epilogue: cuda.Function,
    down_epilogue: cuda.Function,
    copy_rows: cuda.Function,
    tokens: cuda.Function,
    greedy: cuda.Function,
    topk: cuda.Function,
    quantize4: cuda.Function,
    pack_dense: cuda.Function,
    transpose16: cuda.Function,
    q4_group: cuda.Function,
    q4_prefill: cuda.Function,
    q4_prefill_f32: cuda.Function,
    quantize4_mse: cuda.Function,
    make_f8: cuda.Function,
    /// v1.8's decode matmul qmm_kernel by tile config (qmm_cfgs), fp32 out, K slices in a cluster.
    qmm: [qmm_cfgs.len][2][2]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.mods[0] = try cuda.Module.load(d, cuda.kernels.glm_kda);
        errdefer k.mods[0].unload();
        k.mods[1] = try cuda.Module.load(d, cuda.kernels.glm_exl3);
        errdefer k.mods[1].unload();
        k.mods[2] = try cuda.Module.load(d, cuda.kernels.glm_glue);
        errdefer k.mods[2].unload();
        // the 4-bit draft head: Nemotron's copies of qmm_group.cu and qmm_prefill.cu, and its MLX -> tiled packing (nemotron_ops)
        k.mods[3] = try cuda.Module.load(d, cuda.kernels.qmm_group);
        errdefer k.mods[3].unload();
        k.mods[4] = try cuda.Module.load(d, cuda.kernels.qmm_prefill);
        errdefer k.mods[4].unload();
        k.mods[5] = try cuda.Module.load(d, cuda.kernels.nemotron_ops);
        errdefer k.mods[5].unload();
        // TF_GLM_DENSE=q4's decode matmul: v1.8's qmm.cu (cuda/kernels/qmm.py `qmm`, `qmm_cfg`)
        k.mods[6] = try cuda.Module.load(d, cuda.kernels.qmm);
        errdefer k.mods[6].unload();
        k.chain = try k.mods[0].function(sym.kda_chain);
        k.prep = try k.mods[0].function(sym.kda_prep);
        k.step = try k.mods[0].function(sym.kda_step);
        k.out = try k.mods[0].function(sym.kda_out);
        k.replay_layers = try k.mods[0].function(sym.kda_replay_layers);
        k.group = try k.mods[1].function(sym.exl3_group);
        k.rot_in = try k.mods[1].function(sym.exl3_rot_in);
        k.grouped = try k.mods[1].function(sym.exl3_grouped);
        k.gateup_epilogue = try k.mods[1].function(sym.exl3_gateup_epilogue);
        k.down_epilogue = try k.mods[1].function(sym.exl3_down_epilogue);
        k.copy_rows = try k.mods[2].function("tf_glm_copy_rows");
        k.tokens = try k.mods[2].function("tf_glm_tokens");
        k.greedy = try k.mods[2].function("tf_glm_greedy");
        k.topk = try k.mods[2].function("tf_glm_topk");
        k.quantize4 = try k.mods[2].function("tf_glm_quantize4");
        k.pack_dense = try k.mods[5].function("tf_pack_dense");
        k.transpose16 = try k.mods[5].function("tf_transpose_pad16");
        k.q4_group = try k.mods[3].function(q4_sym.group);
        k.q4_prefill = try k.mods[4].function(q4_sym.prefill);
        k.q4_prefill_f32 = try k.mods[4].function(q4_sym.prefill_f32);
        k.quantize4_mse = try k.mods[2].function("tf_glm_quantize4_mse");
        k.make_f8 = try k.mods[2].function("tf_glm_make_f8");
        try k.q4_group.allowDynamicShared(q4_group_smem);
        try k.q4_prefill.allowDynamicShared(q4_prefill_smem);
        try k.q4_prefill_f32.allowDynamicShared(q4_prefill_smem);
        inline for (qmm_cfgs, 0..) |c, i| inline for (.{ false, true }, 0..) |fp32, j| inline for (.{ false, true }, 0..) |cluster, l| {
            k.qmm[i][j][l] = try k.mods[6].function(comptime c.symbol(fp32, cluster));
            try k.qmm[i][j][l].allowDynamicShared(c.smem());
        };
        // group_kernel's members list grows with the rows (a prompt chunk's 2048 rows * 9 slots * 4 B): experts.cu:292-305
        try k.group.allowDynamicShared(96 << 10);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (&k.mods) |*m| m.unload();
    }
};

/// qmm_group's and qmm_prefill's instantiations (nemotron/cuda_kernels.zig), and their shared memory.
const q4_sym = struct {
    const group = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    const prefill = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
    const prefill_f32 = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb1EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
};

/// v1.8 qmm.cu's qmm_kernel<64, 16, bn, 1, wn, stages> for a decode window (rows <= 16). Config 0 is `qmm` with the
/// 16-row tile; 1 and 2 are `qmm_cfg` 2 and 4. dec_tile (glm qmm.py DEC_TILES) picks one by the projection's shape.
const QmmCfg = struct {
    bn: u32,
    wn: u32,
    stages: u32,

    fn symbol(c: QmmCfg, comptime fp32: bool, comptime cluster: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN6tf_qmm10qmm_kernelILi64ELi16ELi{d}ELi1ELi{d}ELi{d}ELb{d}ELb{d}ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii", .{ c.bn, c.wn, c.stages, @intFromBool(fp32), @intFromBool(cluster) });
    }

    /// Tile<64, 16, bn, 1, wn, stages>::SMEM: the pipeline's stages, or the cluster sum's partials if larger.
    fn smem(c: QmmCfg) u32 {
        const stage = 16 * 128 + c.bn * 64 / 2 + 2 * (c.bn * 2) + 16 * 4;
        const partials = (c.bn / c.wn / 8) * 4 * (c.wn * 32) * 4;
        return @max(c.stages * stage, partials);
    }
};
const qmm_cfgs = [_]QmmCfg{ .{ .bn = 64, .wn = 4, .stages = 4 }, .{ .bn = 128, .wn = 8, .stages = 4 }, .{ .bn = 128, .wn = 4, .stages = 6 } };

/// glm qmm.py DEC_TILES (TF_GLM_Q4_TILE unset): the config index of a few projection shapes, else 0.
fn decCfg(n: usize, kk: usize) usize {
    const table = [_][3]usize{ .{ 4096, 4096, 1 }, .{ 2048, 4096, 1 }, .{ 8192, 1536, 1 }, .{ 4096, 8192, 2 }, .{ 4096, 1024, 1 } };
    for (table) |t| if (t[0] == n and t[1] == kk) return t[2];
    return 0;
}
const q4_group_smem: u32 = 35328; // LaneTile<64, 16, 64, 1, 4, 8>::SMEM
const q4_prefill_smem: u32 = 62976; // Tile<64, 128, 128, 2, 4, 3>::SMEM

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
    try t.expectEqual(@as(u32, 26880), qmm_cfgs[1].smem()); // Tile<64, 16, 128, 1, 8, 4>
    try t.expectEqual(@as(u32, 40320), qmm_cfgs[2].smem()); // Tile<64, 16, 128, 1, 4, 6>
}

fn go(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: usize, shared: usize, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = u(grid[0]), .y = u(grid[1]), .z = u(grid[2]) }, .block = .{ .x = u(block) }, .shared = u(shared) }, s, args);
}

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

/// kda.py:58: from this many rows a chain runs as prep, step and out.
pub const wide_rows = 64;

pub fn chain(k: *const Kernels, s: cuda.Stream, c: Chain) !void {
    var a: cuda.Args = .{};
    if (c.rows < wide_rows) {
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
        a.add(c.out);
        a.add(c.state_out);
        a.add(c.k_save);
        a.add(c.v_save);
        a.add(c.g_save);
        a.add(c.b_save);
        return go(k.chain, s, .{ c.heads, 1, 1 }, 1024, 0, &a);
    }
    if (c.q_tmp == 0 or c.y_tmp == 0) return error.WideChainScratch;
    a.add(int(c.heads));
    a.add(c.p);
    a.add(int(c.p_stride));
    a.add(int(c.b_off));
    a.add(c.a);
    a.add(int(c.a_stride));
    a.add(c.conv_state);
    a.add(c.conv_w);
    a.add(c.a_log);
    a.add(c.dt_bias);
    a.add(c.lower);
    a.add(c.q_tmp);
    a.add(c.k_save);
    a.add(c.v_save);
    a.add(c.g_save);
    a.add(c.b_save);
    try go(k.prep, s, .{ c.rows, c.heads, 1 }, 512, 0, &a);
    var b: cuda.Args = .{};
    b.add(int(c.heads));
    b.add(c.state_in);
    b.add(c.q_tmp);
    b.add(c.k_save);
    b.add(c.v_save);
    b.add(c.g_save);
    b.add(c.b_save);
    b.add(int(c.rows));
    b.add(c.y_tmp);
    b.add(c.state_out);
    try go(k.step, s, .{ c.heads, 128 / 4 / 4, 1 }, 128, 0, &b);
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

/// The EXL3 routed experts' scratch (exl3.experts.Scratch) for up to `rows` rows of `slots` slots.
pub const Exl3Scratch = struct {
    rows: usize,
    slots: usize,
    xg: u64, // [P, D] f16
    xu: u64,
    xd: u64, // [P, I] f16
    z: u64, // f32
    y: u64, // [P, D] f32
    ids: u64, // [min(P, E)] i32
    count: u64, // [1] i32
    members: u64, // [min(P, E) * rows] i32, -1 filled

    pub const gu_sk = 4; // GLM_GATEUP (8, 4, 4, 1)
    pub const d_sk = 1; // GLM_DOWN (8, 4, 1, 1)

    pub fn bytes(rows: usize, slots: usize, dims: usize, width: usize, experts: usize) usize {
        const p = rows * slots;
        const maxu = @min(p, experts);
        return p * dims * 2 * 2 + p * width * 2 + @max(2 * gu_sk * width, d_sk * dims) * p * 4 + p * dims * 4 + maxu * 4 + 4 + maxu * rows * 4;
    }
};

/// exl3.experts.routed with wts None (GLM's call, exl3_generic.py:9): Y per slot into `sc.y`, act_mode 0.
pub fn routed(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, pick: u64, ex: anytype, sc: Exl3Scratch, rows: usize, limit: f32) !void {
    const slots = sc.slots;
    const p = rows * slots;
    const e_n = ex.count;
    const d = ex.dims;
    const i_w = ex.width;
    const maxu = @min(p, e_n);
    {
        var a: cuda.Args = .{};
        a.add(pick);
        a.add(sc.ids);
        a.add(sc.count);
        a.add(sc.members);
        a.add(int(rows));
        a.add(int(slots));
        a.add(int(e_n));
        a.add(int(rows));
        try go(k.group, s, .{ 1, 1, 1 }, 1024, p * 4, &a);
    }
    {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(pick);
        a.add(ex.suh_g);
        a.add(ex.suh_u);
        a.add(sc.xg);
        a.add(sc.xu);
        a.add(int(d));
        a.add(int(slots));
        a.add(int(e_n));
        try go(k.rot_in, s, .{ p, d / 128, 2 }, 32, 0, &a);
    }
    const mt = (rows + 15) / 16;
    try grouped(k, s, sc.xg, sc.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, sc, 2, d, i_w, p, Exl3Scratch.gu_sk, rows, maxu, mt);
    {
        var a: cuda.Args = .{};
        a.add(sc.z);
        a.add(pick);
        a.add(ex.svh_g);
        a.add(ex.svh_u);
        a.add(ex.suh_d);
        a.add(sc.xd);
        a.add(int(p));
        a.add(int(i_w));
        a.add(int(Exl3Scratch.gu_sk));
        a.add(int(e_n));
        a.add(limit);
        a.add(@as(i32, 0));
        try go(k.gateup_epilogue, s, .{ p, i_w / 128, 1 }, 32, 0, &a);
    }
    try grouped(k, s, sc.xd, sc.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, sc, 1, i_w, d, p, Exl3Scratch.d_sk, rows, maxu, mt);
    {
        var a: cuda.Args = .{};
        a.add(sc.z);
        a.add(pick);
        a.add(ex.svh_d);
        a.add(sc.y);
        a.add(int(p));
        a.add(int(d));
        a.add(int(Exl3Scratch.d_sk));
        a.add(int(e_n));
        try go(k.down_epilogue, s, .{ p, d / 128, 1 }, 32, 0, &a);
    }
}

fn grouped(k: *const Kernels, s: cuda.Stream, x0: u64, x1: u64, tp0: u64, tp1: u64, k2_0: u64, k2_1: u64, sc: Exl3Scratch, mats: usize, kk: usize, n: usize, p: usize, sk: usize, maxm: usize, maxu: usize, mt: usize) !void {
    if (kk % (16 * sk * 4) != 0 or n % (16 * 8) != 0) return error.Exl3Tiles;
    var a: cuda.Args = .{};
    a.add(x0);
    a.add(x1);
    a.add(tp0);
    a.add(tp1);
    a.add(k2_0);
    a.add(k2_1);
    a.add(sc.ids);
    a.add(sc.count);
    a.add(sc.members);
    a.add(sc.z);
    a.add(int(kk));
    a.add(int(n));
    a.add(int(p));
    a.add(int(sk));
    a.add(int(maxm));
    a.add(int(sc.slots));
    try go(k.grouped, s, .{ maxu, n / 128, mats * sk * mt }, 128, 0, &a);
}
