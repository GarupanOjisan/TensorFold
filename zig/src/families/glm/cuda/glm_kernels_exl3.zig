//! GLM's EXL3 routed-expert launches (the experts.cu grouped path) and the routing plan.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("glm_kernels.zig");
const Kernels = kern.Kernels;
const go = kern.go;
const int = kern.int;
const u = kern.u;
const prompt_smem = kern.prompt_smem;
const prompt_smem_rows = kern.prompt_smem_rows;

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
    done: u64, // TF_GLM_EXL3_FUSE 2: [max_items(pairs, experts, 16), 8] i32 finished gate/up blocks, zero between launches
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
