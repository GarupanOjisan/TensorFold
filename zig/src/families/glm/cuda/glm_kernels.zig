//! GLM's raw CUDA kernels (KDA, EXL3, glue fatbins) with the Python host geometry; spec in glm_symbols.zig.
const std = @import("std");
const cuda = @import("cuda");
const sym = @import("glm_symbols.zig");
const q4 = @import("glm_kernels_q4.zig");
const kda = @import("glm_kernels_kda.zig");
const exl3 = @import("glm_kernels_exl3.zig");

pub fn u(x: usize) u32 {
    return @intCast(x);
}

pub fn int(x: usize) i32 {
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
    dec_gu_fused: [2][2][3]cuda.Function, // FUSE 2 by tile bits, rows_in, launch order
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
    nuc_stats: cuda.Function, // the top_p nucleus draw's device side (glm_port_glue.cu)
    nuc_pack: cuda.Function,
    df_scatter: cuda.Function, // DFlash2's glue (glm_port_glue.cu)
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
        k.mods[0] = try cuda.Module.load(d, cuda.kernels.glm_port_kda);
        errdefer k.mods[0].unload();
        k.mods[1] = try cuda.Module.load(d, cuda.kernels.glm_exl3);
        errdefer k.mods[1].unload();
        k.mods[2] = try cuda.Module.load(d, cuda.kernels.glm_port_glue);
        errdefer k.mods[2].unload();
        // the 4-bit draft head: Nemotron's qmm_group.cu, qmm_prefill.cu and affine4_pack (MLX -> tiled packing)
        k.mods[3] = try cuda.Module.load(d, cuda.kernels.qmm_group);
        errdefer k.mods[3].unload();
        k.mods[4] = try cuda.Module.load(d, cuda.kernels.qmm_prefill);
        errdefer k.mods[4].unload();
        k.mods[5] = try cuda.Module.load(d, cuda.kernels.affine4_pack);
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
            k.dec_gu_fused[v][0][o] = try k.mods[7].function(e18.dec(2, false, o, bits));
            k.dec_gu_fused[v][1][o] = try k.mods[7].function(e18.dec(2, true, o, bits));
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
        // group_kernel's members list grows with the rows (2048 rows * 9 slots * 4 B; experts.cu:292-305)
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

/// v1.8 qmm.cu's qmm_kernel<64, 16, bn, 1, wn, stages> for decode windows; notes/glm_kernels.md#qmm-configs
const QmmCfg = struct {
    bm: u32 = 16,
    bn: u32,
    wn: u32,
    stages: u32,

    fn symbol(c: QmmCfg, comptime fp32: bool, comptime cluster: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN6tf_qmm10qmm_kernelILi64ELi{d}ELi{d}ELi1ELi{d}ELi{d}ELb{d}ELb{d}ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii", .{ c.bm, c.bn, c.wn, c.stages, @intFromBool(fp32), @intFromBool(cluster) });
    }

    /// Tile<64, bm, bn, 1, wn, stages>::SMEM: the pipeline's stages, or the cluster sum's partials if larger.
    pub fn smem(c: QmmCfg) u32 {
        const stage = c.bm * 128 + c.bn * 64 / 2 + 2 * (c.bn * 2) + c.bm * 4;
        const partials = (c.bm / 16) * (c.bn / c.wn / 8) * 4 * (c.wn * 32) * 4;
        return @max(c.stages * stage, partials);
    }
};
pub const qmm32: QmmCfg = .{ .bm = 32, .bn = 64, .wn = 4, .stages = 4 };
pub const qmm_cfgs = [_]QmmCfg{ .{ .bn = 64, .wn = 4, .stages = 4 }, .{ .bn = 128, .wn = 8, .stages = 4 }, .{ .bn = 128, .wn = 4, .stages = 6 } };

/// glm_exl3d.cu's and experts.cu's kernel names and prompt shared memory; notes/glm_kernels.md#e18
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
pub const prompt_smem: u32 = 43008; // prompt_smem<64>(): XM = 2 copies of the member rows (gate and up apart; the down kernel)
pub const prompt_smem_rows: u32 = 33792; // prompt_smem<64, 1>(): one copy (gate/up of a layer whose experts share one suh)

pub const q4_group_smem: u32 = 35328; // LaneTile<64, 16, 64, 1, 4, 8>::SMEM
pub const q4_prefill_smem: u32 = 62976; // Tile<64, 128, 128, 2, 4, 3>::SMEM

pub fn go(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: usize, shared: usize, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = u(grid[0]), .y = u(grid[1]), .z = u(grid[2]) }, .block = .{ .x = u(block) }, .shared = u(shared) }, s, args);
}

/// kda.py WIDE_ROWS: from this many rows a chain runs as prep, step and out; notes/glm_kernels.md#wide-rows
pub var wide_rows: usize = 1;

/// TF_GLM_EXL3_DEC_ORDER: the decode expert kernel's block launch order; notes/glm_kernels.md#dec-order
pub var dec_order: usize = 2;
/// TF_GLM_EXL3_FUSE2: the gate/up epilogue in the gate/up kernel's last block; notes/glm_kernels.md#fuse-gu
pub var fuse_gu: bool = false;

/// sizeof(Smem) of kda_chunk.cu's chunk_kernel (nvcc).
pub const kda_chunk_smem: u32 = 98560;

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

/// exl3_mm.routed for GLM at two ranks on `plan`: Y per routed slot into `sc.y`; notes/glm_kernels.md#routed18
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
    if (fuse_gu) { // the last block of each (item, n block) runs the epilogue
        g.add(pl.done);
        try go(k.dec_gu_fused[bv][@intFromBool(rows_in)][dec_order], s, .{ items, i_w / 128, 2 * sk_gu }, 128, 0, &g);
    } else {
        g.add(@as(u64, 0)); // done: only the gate/up epilogue fused into the kernel (TF_GLM_EXL3_FUSE 2) counts blocks
        try go(if (rows_in) k.dec_gu_rows[bv][dec_order] else k.dec_gu[bv][dec_order], s, .{ items, i_w / 128, 2 * sk_gu }, 128, 0, &g);
        var e: cuda.Args = .{};
        for ([_]u64{ sc.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, sc.xd }) |v| e.add(v);
        for ([_]usize{ p, i_w, sk_gu, slots }) |v| e.add(int(v));
        e.add(limit);
        try go(k.gu_epilogue18, s, .{ p, i_w / 128, 1 }, 32, 0, &e);
    }
    var dn: cuda.Args = .{};
    for ([_]u64{ sc.xd, sc.xd, ex.down_base, ex.down_base, pl.items, pl.counts, pl.members, sc.z }) |v| dn.add(v);
    for ([_]usize{ i_w, d, p, 1, e_n, slots }) |v| dn.add(int(v));
    for ([_]u64{ ex.svh_d, ex.svh_d, ex.svh_d, sc.y }) |v| dn.add(v);
    dn.add(@as(f32, 0));
    dn.add(@as(u64, 0));
    try go(k.dec_down[bv][dec_order], s, .{ items, d / 128, 1 }, 128, 0, &dn);
}

pub const Quant = q4.Quant;
pub const Q4 = q4.Q4;
pub const q4Bytes = q4.q4Bytes;
pub const quantize4 = q4.quantize4;
pub const F8 = q4.F8;
pub const makeF8 = q4.makeF8;
pub const denseSplitK = q4.denseSplitK;
pub const qmmShared = q4.qmmShared;
pub const qmmDecode = q4.qmmDecode;
pub const q4Matmul = q4.q4Matmul;
pub const q4Prefill = q4.q4Prefill;
pub const q4SplitK = q4.q4SplitK;
pub const Chain = kda.Chain;
pub const seg_cols = kda.seg_cols;
pub const SegRow = kda.SegRow;
pub const segmentTable = kda.segmentTable;
pub const ChainSegments = kda.ChainSegments;
pub const chainSegments = kda.chainSegments;
pub const replayLayersSegments = kda.replayLayersSegments;
pub const hcPartial = kda.hcPartial;
pub const l2Prefetch = kda.l2Prefetch;
pub const chainChunked = kda.chainChunked;
pub const replayLayers = kda.replayLayers;
pub const Exl3Scratch = exl3.Exl3Scratch;
pub const routed = exl3.routed;
pub const Plan = exl3.Plan;
pub const maxItems = exl3.maxItems;
pub const route = exl3.route;

test {
    _ = q4;
    _ = kda;
    _ = exl3;
}
