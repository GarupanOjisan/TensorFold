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
    mods: [3]cuda.Module,
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

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.mods[0] = try cuda.Module.load(d, cuda.kernels.glm_kda);
        errdefer k.mods[0].unload();
        k.mods[1] = try cuda.Module.load(d, cuda.kernels.glm_exl3);
        errdefer k.mods[1].unload();
        k.mods[2] = try cuda.Module.load(d, cuda.kernels.glm_glue);
        errdefer k.mods[2].unload();
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
        // group_kernel's members list grows with the rows (a prompt chunk's 2048 rows * 9 slots * 4 B): experts.cu:292-305
        try k.group.allowDynamicShared(96 << 10);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (&k.mods) |*m| m.unload();
    }
};

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
