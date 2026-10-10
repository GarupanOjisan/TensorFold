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
    mods: [13]cuda.Module,
    chain: cuda.Function,
    prep: cuda.Function,
    step: cuda.Function,
    out: cuda.Function,
    replay_layers: cuda.Function,
    prep_seg: cuda.Function,
    step_seg: cuda.Function,
    replay_layers_seg: cuda.Function,
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
    /// qmm_kernel on a 32-row tile (qmm32), fp32 out, K slices in a cluster.
    qmm32: [2][2]cuda.Function,
    /// v1.8's routed-expert kernels (glm_exl3d: glm5_next/cuda/exl3.cu) and the plan kernels of experts.cu.
    rot_rows: cuda.Function,
    rot_pairs: cuda.Function,
    dec_gu_rows: [2][3]cuda.Function, // by tile bits (0: 4-bit, 1: 3-bit), then launch order (TF_GLM_EXL3_DEC_ORDER, v1.10 patch 0090)
    dec_gu: [2][3]cuda.Function,
    dec_down: [2][3]cuda.Function,
    gu_epilogue18: cuda.Function,
    prompt_order: cuda.Function,
    prompt_gu_rows: [2]cuda.Function, // by tile bits (0: 4-bit, 1: 3-bit)
    prompt_gu: [2]cuda.Function,
    prompt_down: [2]cuda.Function,
    plan_small: cuda.Function,
    plan_rank: cuda.Function,
    plan_offsets: cuda.Function,
    plan_scatter: cuda.Function,
    kda_chunk: cuda.Function,
    hc_partial: cuda.Function,
    roce_gather: cuda.Function, // TF_GLM_COMM=roce (glm_roce.zig)
    l2pf: cuda.Function,
    nuc_stats: cuda.Function, // the top_p nucleus draw's device side (glm_glue.cu)
    nuc_pack: cuda.Function,
    df_scatter: cuda.Function, // DFlash2's glue (glm_glue.cu)
    df_scatter_seg: cuda.Function,
    df_scatter_rows: cuda.Function,
    df_rope_rows: cuda.Function,
    df_rope: cuda.Function,
    df_pos_add: cuda.Function,
    df_rowsum: cuda.Function,
    df_cat: cuda.Function,
    df_proj: cuda.Function,

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
        k.mods[7] = try cuda.Module.load(d, cuda.kernels.glm_exl3d);
        errdefer k.mods[7].unload();
        k.mods[8] = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer k.mods[8].unload();
        k.mods[9] = try cuda.Module.load(d, cuda.kernels.glm_kda_chunk);
        errdefer k.mods[9].unload();
        k.kda_chunk = try k.mods[9].function("_ZN12tf_kda_chunk12chunk_kernelEiPK13__nv_bfloat16iiS2_iS2_iS2_S2_PKfS4_S4_S2_ffiiPS0_Pf");
        try k.kda_chunk.allowDynamicShared(kda_chunk_smem);
        k.mods[10] = try cuda.Module.load(d, cuda.kernels.glm_hc);
        errdefer k.mods[10].unload();
        k.hc_partial = try k.mods[10].function("_ZN9tf_glm_hc17hc_partial_kernelEPK13__nv_bfloat16S2_Pfi");
        k.mods[12] = try cuda.Module.load(d, cuda.kernels.glm_roce);
        errdefer k.mods[12].unload();
        k.roce_gather = try k.mods[12].function("tf_glm_roce_gather");
        k.mods[11] = try cuda.Module.load(d, cuda.kernels.glm_l2pf);
        errdefer k.mods[11].unload();
        k.l2pf = try k.mods[11].function("_ZN7tf_l2pf11l2pf_kernelEPKxiiPj");
        k.chain = try k.mods[0].function(sym.kda_chain);
        k.prep = try k.mods[0].function(sym.kda_prep);
        k.step = try k.mods[0].function(sym.kda_step);
        k.out = try k.mods[0].function(sym.kda_out);
        k.replay_layers = try k.mods[0].function(sym.kda_replay_layers);
        k.prep_seg = try k.mods[0].function(sym.kda_prep_seg);
        k.step_seg = try k.mods[0].function(sym.kda_step_seg);
        k.replay_layers_seg = try k.mods[0].function(sym.kda_replay_layers_seg);
        k.group = try k.mods[1].function(sym.exl3_group);
        k.rot_in = try k.mods[1].function(sym.exl3_rot_in);
        k.grouped = try k.mods[1].function(sym.exl3_grouped);
        k.gateup_epilogue = try k.mods[1].function(sym.exl3_gateup_epilogue);
        k.down_epilogue = try k.mods[1].function(sym.exl3_down_epilogue);
        k.copy_rows = try k.mods[2].function("tf_glm_copy_rows");
        k.tokens = try k.mods[2].function("tf_glm_tokens");
        k.greedy = try k.mods[2].function("tf_glm_greedy");
        k.topk = try k.mods[2].function("tf_glm_topk");
        k.nuc_stats = try k.mods[2].function("tf_glm_nuc_stats");
        k.nuc_pack = try k.mods[2].function("tf_glm_nuc_pack");
        k.df_scatter = try k.mods[2].function("tf_glm_df_scatter");
        k.df_scatter_seg = try k.mods[2].function("tf_glm_df_scatter_seg");
        k.df_scatter_rows = try k.mods[2].function("tf_glm_df_scatter_rows");
        k.df_rope_rows = try k.mods[2].function("tf_glm_df_rope_rows");
        k.df_rope = try k.mods[2].function("tf_glm_df_rope");
        k.df_pos_add = try k.mods[2].function("tf_glm_df_pos_add");
        k.df_rowsum = try k.mods[2].function("tf_glm_df_rowsum");
        k.df_cat = try k.mods[2].function("tf_glm_df_cat");
        k.df_proj = try k.mods[2].function("tf_glm_df_proj");
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
            if (i == 0) {
                k.qmm32[j][l] = try k.mods[6].function(comptime qmm32.symbol(fp32, cluster));
                try k.qmm32[j][l].allowDynamicShared(qmm32.smem());
            }
        };
        k.rot_rows = try k.mods[7].function(e18.rot_rows);
        k.rot_pairs = try k.mods[7].function(e18.rot_pairs);
        inline for (.{ 4, 3 }, 0..) |bits, v| inline for (0..3) |o| {
            k.dec_gu_rows[v][o] = try k.mods[7].function(e18.dec(0, true, o, bits));
            k.dec_gu[v][o] = try k.mods[7].function(e18.dec(0, false, o, bits));
            k.dec_down[v][o] = try k.mods[7].function(e18.dec(1, false, o, bits));
        };
        k.gu_epilogue18 = try k.mods[7].function(e18.gu_epilogue);
        k.prompt_order = try k.mods[7].function(e18.prompt_order);
        inline for (.{ 4, 3 }, 0..) |bits, v| {
            k.prompt_gu_rows[v] = try k.mods[7].function(e18.promptGu(true, bits));
            k.prompt_gu[v] = try k.mods[7].function(e18.promptGu(false, bits));
            k.prompt_down[v] = try k.mods[7].function(e18.promptDown(bits));
            try k.prompt_gu_rows[v].allowDynamicShared(prompt_smem_rows);
            try k.prompt_gu[v].allowDynamicShared(prompt_smem);
            try k.prompt_down[v].allowDynamicShared(prompt_smem);
        }
        k.plan_small = try k.mods[8].function(e18.plan_small);
        k.plan_rank = try k.mods[8].function(e18.plan_rank);
        k.plan_offsets = try k.mods[8].function(e18.plan_offsets);
        k.plan_scatter = try k.mods[8].function(e18.plan_scatter);
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
    bm: u32 = 16,
    bn: u32,
    wn: u32,
    stages: u32,

    fn symbol(c: QmmCfg, comptime fp32: bool, comptime cluster: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN6tf_qmm10qmm_kernelILi64ELi{d}ELi{d}ELi1ELi{d}ELi{d}ELb{d}ELb{d}ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii", .{ c.bm, c.bn, c.wn, c.stages, @intFromBool(fp32), @intFromBool(cluster) });
    }

    /// Tile<64, bm, bn, 1, wn, stages>::SMEM: the pipeline's stages, or the cluster sum's partials if larger.
    fn smem(c: QmmCfg) u32 {
        const stage = c.bm * 128 + c.bn * 64 / 2 + 2 * (c.bn * 2) + c.bm * 4;
        const partials = (c.bm / 16) * (c.bn / c.wn / 8) * 4 * (c.wn * 32) * 4;
        return @max(c.stages * stage, partials);
    }
};
const qmm32: QmmCfg = .{ .bm = 32, .bn = 64, .wn = 4, .stages = 4 };
const qmm_cfgs = [_]QmmCfg{ .{ .bn = 64, .wn = 4, .stages = 4 }, .{ .bn = 128, .wn = 8, .stages = 4 }, .{ .bn = 128, .wn = 4, .stages = 6 } };

/// glm_exl3d.cu's and experts.cu's instantiations (cuobjdump -symbols), and the prompt kernels' dynamic shared memory
/// (exl3.cu prompt_smem<64, XM>: three stages of the members' rows and the trellis words, or an epilogue half).
const e18 = struct {
    const rot_rows = "_ZN12tf_glm_exl3d13rot_in_kernelILb1EEEvPK13__nv_bfloat16iPKiPK6__halfS8_PS6_S9_ii";
    const rot_pairs = "_ZN12tf_glm_exl3d13rot_in_kernelILb0EEEvPK13__nv_bfloat16iPKiPK6__halfS8_PS6_S9_ii";
    /// dec_kernel<8, 4, 16, fuse, xrow, 1, 1, swz, bits>'s mangled name.
    fn dec(comptime fuse: u8, comptime xrow: bool, comptime swz: u8, comptime bits: u8) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN12tf_glm_exl3d10dec_kernelILi8ELi4ELi16ELi{d}ELb{d}ELi1ELi1ELi{d}ELi{d}EEEvPK6__halfS3_PKjS5_PKiS7_S7_PfiiiiiiS3_S3_S3_PvfPi", .{ fuse, @intFromBool(xrow), swz, bits });
    }
    /// prompt_gateup_kernel<64, 8, shx, bits>'s mangled name.
    fn promptGu(comptime shx: bool, comptime bits: u8) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN12tf_glm_exl3d20prompt_gateup_kernelILi64ELi8ELb{d}ELi{d}EEEvPK6__halfS3_PKjS5_PKiS7_S7_S3_S3_S3_PS1_iiifiS7_", .{ @intFromBool(shx), bits });
    }
    /// prompt_down_kernel<64, 8, bits>'s mangled name.
    fn promptDown(comptime bits: u8) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN12tf_glm_exl3d18prompt_down_kernelILi64ELi8ELi{d}EEEvPK6__halfPKjPKiS7_S7_S3_PfiiiS7_", .{bits});
    }
    const gu_epilogue = "_ZN12tf_glm_exl3d22gateup_epilogue_kernelEPKfPKiPK6__halfS6_S6_PS4_iiiif";
    const prompt_order = "_ZN12tf_glm_exl3d19prompt_order_kernelEPKiS1_Piii";
    const prompt_gu_rows = "_ZN12tf_glm_exl3d20prompt_gateup_kernelILi64ELi8ELb1EEEvPK6__halfS3_PKjS5_PKiS7_S7_S3_S3_S3_PS1_iiifiS7_";
    const prompt_gu = "_ZN12tf_glm_exl3d20prompt_gateup_kernelILi64ELi8ELb0EEEvPK6__halfS3_PKjS5_PKiS7_S7_S3_S3_S3_PS1_iiifiS7_";
    const prompt_down = "_ZN12tf_glm_exl3d18prompt_down_kernelILi64ELi8EEEvPK6__halfPKjPKiS7_S7_S3_PfiiiS7_";
    const plan_small = "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_";
    const plan_rank = "_ZN10tf_experts9plan_rankEPKiiiPiS2_";
    const plan_offsets = "_ZN10tf_experts12plan_offsetsEiiiPiS0_S0_";
    const plan_scatter = "_ZN10tf_experts12plan_scatterEPKiiiS1_S1_Pi";
};
const prompt_smem: u32 = 43008; // prompt_smem<64>(): XM = 2 copies of the member rows (gate and up apart; the down kernel)
const prompt_smem_rows: u32 = 33792; // prompt_smem<64, 1>(): one copy (gate/up of a layer whose experts share one suh)

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

/// kda.py WIDE_ROWS (TF_GLM_KDA_WIDE_ROWS, v1.8's default 1; 64 was v0.6.6's): from this many rows a chain runs as prep, step
/// and out (the same bits as the one-block chain). The engine sets it from the environment before it sizes its buffers.
pub var wide_rows: usize = 1;

/// TF_GLM_EXL3_DEC_ORDER (v1.10 patch 0090): the decode expert kernel's block launch order, 0 as launched, 1 or 2 (default) a k slice's
/// blocks together; the same bits. The engine sets it from the environment.
pub var dec_order: usize = 2;

/// sizeof(Smem) of kda_chunk.cu's chunk_kernel (nvcc).
const kda_chunk_smem: u32 = 98560;

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
    plan: Plan, // routed18's plan (TF_GLM_EXL3_DEC)

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

/// experts.Plan's scratch (experts.py:113): pairs grouped by expert into items of at most `tile` pairs.
pub const Plan = struct {
    members: u64, // [pairs] i32
    items: u64, // [max_items(pairs, experts, 16), 3] i32: (expert, first member, count)
    counts: u64, // [2] i32: items, distinct experts
    rank: u64, // [pairs] i32 when the plan is wide (pairs > 1024), else one word
    hist: u64, // [ceil(pairs / 1024), experts] i32 when wide
    order: u64, // a prompt plan's launch order, [pairs / 16 + 1024] i32
};

/// experts.max_items: the items a plan of `pairs` can hold.
pub fn maxItems(pairs: usize, experts: usize, tile: usize) usize {
    return @min(pairs, experts) + pairs / tile;
}

/// experts.route (experts.cu:306): `picks` [pairs] expert ids -> the plan; `experts` counts the shared expert's id.
pub fn route(k: *const Kernels, s: cuda.Stream, picks: u64, pairs: usize, experts: usize, tile: usize, p: Plan) !void {
    var a: cuda.Args = .{};
    if (pairs <= 1024) {
        a.add(picks);
        for ([_]usize{ pairs, experts, tile }) |v| a.add(int(v));
        for ([_]u64{ p.members, p.items, p.counts }) |v| a.add(v);
        return go(k.plan_small, s, .{ 1, 1, 1 }, 1024, 0, &a);
    }
    const nblk = (pairs + 1023) / 1024;
    a.add(picks);
    a.add(int(pairs));
    a.add(int(experts));
    a.add(p.rank);
    a.add(p.hist);
    try go(k.plan_rank, s, .{ nblk, 1, 1 }, 1024, 0, &a);
    var b: cuda.Args = .{};
    for ([_]usize{ nblk, experts, tile }) |v| b.add(int(v));
    for ([_]u64{ p.hist, p.items, p.counts }) |v| b.add(v);
    try go(k.plan_offsets, s, .{ 1, 1, 1 }, 1024, 0, &b);
    var c: cuda.Args = .{};
    c.add(picks);
    c.add(int(pairs));
    c.add(int(experts));
    for ([_]u64{ p.rank, p.hist, p.members }) |v| c.add(v);
    try go(k.plan_scatter, s, .{ (pairs + 255) / 256, 1, 1 }, 256, 0, &c);
}

/// exl3_mm.routed for GLM at two ranks (D 4096, an expert width of 1024) on `plan` (route): Y per routed slot into
/// `sc.y`. A prompt chunk (`prompt`: plan items of 64 pairs) runs the Y^T kernels, a decode window (under 64 rows,
/// items of 16) the `dec` kernels with TF_GLM_EXL3_FUSE 1 (the down epilogue fused) and the 16-byte loads
/// (TF_GLM_EXL3_LOADS=nc). `ex.shared_suh` (0: none) is the one gate/up suh of a layer whose experts all share it.
pub fn routed18(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, pick: u64, ex: anytype, sc: Exl3Scratch, rows: usize, limit: f32, prompt: bool) !void {
    const bv: usize = switch (ex.k2) {
        8 => 0,
        6 => 1,
        else => return error.Exl3Bits, // 4- and 3-bit tiles only
    };
    const slots = sc.slots;
    const p = rows * slots;
    const e_n = ex.count;
    const d = ex.dims;
    const i_w = ex.width;
    const pl = sc.plan;
    if (d != 4096 or i_w != 1024 or d % 256 != 0) return error.Exl3Shape;
    if (!prompt and rows >= 64) return error.WindowTooWide; // PROMPT_ROWS: a window from 64 rows takes the prompt plan
    const tile: usize = if (prompt) 64 else 16;
    try route(k, s, pick, p, e_n + 1, tile, pl);
    const rows_in = ex.shared_suh != 0;
    if (rows_in) { // one rotated input a row, read by all its pairs' gate and up
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(@as(u64, 0));
        a.add(ex.shared_suh);
        a.add(ex.shared_suh);
        a.add(sc.xg);
        a.add(sc.xg);
        a.add(int(d));
        a.add(@as(i32, 1));
        try go(k.rot_rows, s, .{ rows, d / 128, 1 }, 32, 0, &a);
    } else {
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
        try go(k.rot_pairs, s, .{ p, d / 128, 2 }, 32, 0, &a);
    }
    const xu = if (rows_in) sc.xg else sc.xu;
    const items = maxItems(p, e_n + 1, tile);
    if (prompt) {
        var o: cuda.Args = .{};
        for ([_]u64{ pl.items, pl.counts, pl.order }) |v| o.add(v);
        o.add(int(items));
        o.add(int(e_n));
        try go(k.prompt_order, s, .{ 1, 1, 1 }, 1024, 0, &o);
        var g: cuda.Args = .{};
        for ([_]u64{ sc.xg, xu, ex.gate_base, ex.up_base, pl.items, pl.counts, pl.members, ex.svh_g, ex.svh_u, ex.suh_d, sc.xd }) |v| g.add(v);
        for ([_]usize{ d, i_w, e_n }) |v| g.add(int(v));
        g.add(limit);
        g.add(int(slots));
        g.add(pl.order);
        try go(if (rows_in) k.prompt_gu_rows[bv] else k.prompt_gu[bv], s, .{ i_w / 128, items, 1 }, 256, if (rows_in) prompt_smem_rows else prompt_smem, &g);
        var dn: cuda.Args = .{};
        for ([_]u64{ sc.xd, ex.down_base, pl.items, pl.counts, pl.members, ex.svh_d, sc.y }) |v| dn.add(v);
        for ([_]usize{ i_w, d, e_n }) |v| dn.add(int(v));
        dn.add(pl.order);
        return go(k.prompt_down[bv], s, .{ d / 256, items, 1 }, 256, prompt_smem, &dn);
    }
    const sk_gu = Exl3Scratch.gu_sk;
    var g: cuda.Args = .{};
    for ([_]u64{ sc.xg, xu, ex.gate_base, ex.up_base, pl.items, pl.counts, pl.members, sc.z }) |v| g.add(v);
    for ([_]usize{ d, i_w, p, sk_gu, e_n, slots }) |v| g.add(int(v));
    for ([_]u64{ ex.svh_g, ex.svh_u, ex.suh_d, sc.xd }) |v| g.add(v);
    g.add(limit);
    g.add(@as(u64, 0)); // done: only the gate/up epilogue fused into the kernel (TF_GLM_EXL3_FUSE 2) counts blocks
    try go(if (rows_in) k.dec_gu_rows[bv][dec_order] else k.dec_gu[bv][dec_order], s, .{ items, i_w / 128, 2 * sk_gu }, 128, 0, &g);
    var e: cuda.Args = .{};
    for ([_]u64{ sc.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, sc.xd }) |v| e.add(v);
    for ([_]usize{ p, i_w, sk_gu, slots }) |v| e.add(int(v));
    e.add(limit);
    try go(k.gu_epilogue18, s, .{ p, i_w / 128, 1 }, 32, 0, &e);
    var dn: cuda.Args = .{};
    for ([_]u64{ sc.xd, sc.xd, ex.down_base, ex.down_base, pl.items, pl.counts, pl.members, sc.z }) |v| dn.add(v);
    for ([_]usize{ i_w, d, p, 1, e_n, slots }) |v| dn.add(int(v));
    for ([_]u64{ ex.svh_d, ex.svh_d, ex.svh_d, sc.y }) |v| dn.add(v);
    dn.add(@as(f32, 0));
    dn.add(@as(u64, 0));
    try go(k.dec_down[bv][dec_order], s, .{ items, d / 128, 1 }, 128, 0, &dn);
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

test "routed-expert plan items and prompt kernels' shared memory" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 9), maxItems(9, 289, 16)); // one row: nine pairs, each its own expert
    try t.expectEqual(@as(usize, 289 + 18432 / 64), maxItems(18432, 289, 64)); // a 2,048-row chunk
    // exl3.cu prompt_smem<64, XM>(): the larger of the stages (XM member-row copies and 2 x PWORDS halves, 3 deep) and the epilogue halves
    const stages = struct {
        fn bytes(xm: usize) usize {
            return 3 * (xm * 64 * 40 + 2 * 1024) * 2;
        }
    };
    try t.expectEqual(@as(usize, prompt_smem), stages.bytes(2));
    try t.expectEqual(@as(usize, 2 * 32 * 132 * 4), prompt_smem_rows); // 33,792 B, above stages.bytes(1) = 27,648 B
    try t.expect(stages.bytes(1) < prompt_smem_rows);
}
