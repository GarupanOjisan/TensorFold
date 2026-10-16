//! One rank's half of GLM-5.3-Flash's EXL3 checkpoint on the GPU, as weights.py lays it out; notes/glm_weights.md
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm_config = @import("glm_config");
const Source = @import("glm_source.zig").Source;
const kern = @import("glm_kernels.zig");
const loader = @import("glm_weights_loader.zig");
const draft = @import("glm_weights_draft.zig");

pub const Config = glm_config.Config;
pub const world = 2;
pub const prefix = "model.language_model.";

/// How a dense (non-expert) projection is held (TF_GLM_DENSE); notes/glm_weights.md#kind
pub const Kind = enum { b16, q4, f8 };

/// A projection [n, k]: BF16 rows, tiled 4-bit groups or e4m3 rows; notes/glm_weights.md#mat
pub const Mat = struct {
    kind: Kind = .b16,
    ptr: u64,
    n: usize,
    k: usize,
    s: u64 = 0,
    b: u64 = 0,
    npad: usize = 0,

    pub fn q4(m: Mat) kern.Q4 {
        std.debug.assert(m.kind == .q4);
        return .{ .w = m.ptr, .s = m.s, .b = m.b, .n = m.n, .k = m.k, .npad = m.npad };
    }
};

pub const Hc = struct { fn_: u64, base: u64, scale: u64 }; // [24, S*D] bf16, [24] f32, [3] f32

pub const Kda = struct {
    proj: Mat, // [q | k | v | f_a | g_a | b] of this rank
    fa_off: usize,
    ga_off: usize,
    b_off: usize,
    fb: Mat,
    gb: Mat,
    conv: u64, // [3 * LL * 128, taps] bf16
    a_log: u64, // [LL] f32
    dt_bias: u64, // [LL * 128] f32
    norm: u64, // [128] bf16
    o: Mat,
};

pub const Index = struct {
    kw: Mat, // [wk | weights_proj]
    qb: Mat,
    ln_w: u64,
    ln_b: u64,
    gate: u64, // index_kpool_compress_gate [128, D] bf16
    ape: u64, // index_kpool_compress_ape [kpool, 128] bf16
};

pub const Dsa = struct {
    proj: Mat, // [q_a | kv_a], replicated
    q_norm: u64,
    kv_norm: u64,
    q_b: Mat,
    o: Mat,
    index: Index,
    wk: u64, // AbsorbW: kv_b's key rows [HL, qk_dim, kv_lora] bf16
    wv: u64, // and its value rows [HL, v_dim, kv_lora]
};

pub const Mlp = struct { gu: Mat, down: Mat, width: usize };

/// One layer's routed experts (exl3.experts.Exl3RoutedExperts): trellis pointer tables (i64), widths (i32 half-bits).
pub const Exl3 = struct {
    gate_ptr: u64,
    up_ptr: u64,
    down_ptr: u64,
    gate_k2: u64,
    up_k2: u64,
    down_k2: u64,
    suh_g: u64, // [E, D] f16
    suh_u: u64,
    svh_g: u64, // [E, I] f16
    svh_u: u64,
    suh_d: u64, // [E, I]
    svh_d: u64, // [E, D]
    count: usize,
    dims: usize,
    width: usize, // I on this rank
    k2: u32, // every trellis here has the same width (checked)
    gate_base: u64, // the trellises of all experts as one block [E, K/16, N/16, 32] words (v1.8's kernels)
    up_base: u64,
    down_base: u64,
    shared_suh: u64, // [D] f16 if every expert's gate and up suh is this vector bit for bit (exl3_mm.shared_suh), else 0
};

pub const Moe = struct { router: u64, bias: u64, ex: Exl3, shared: Mlp };

pub const Layer = struct {
    index: u32,
    kind: glm_config.Kind,
    attn_hc: ?Hc,
    ffn_hc: ?Hc,
    in_norm: u64,
    post_norm: u64,
    kda: ?Kda = null,
    dsa: ?Dsa = null,
    mlp: ?Mlp = null,
    moe: ?Moe = null,
};

pub const Mtp = struct { enorm: u64, hnorm: u64, eh: Mat, head_norm: u64, layer: Layer };

/// One layer of the DFlash2 drafter (dflash2.DraftLayer), this rank's share.
pub const DraftLayer = struct {
    in_norm: u64,
    post_norm: u64,
    a_base: u64, // [2, 2, D] bf16
    a_kp: kern.Q4, // [2 * 2 * D / gs, D]
    m_base: u64,
    m_kp: kern.Q4,
    qkv: kern.Q4, // [q heads | k heads | v heads] of this rank
    kv: kern.Q4, // [k heads | v heads] (the context rows)
    q_norm: u64,
    k_norm: u64,
    o: kern.Q4, // [D, heads * head_dim]: a row-parallel partial
    gu: kern.Q4, // [gate | up] of this rank
    down: kern.Q4, // [D, MLP width]: a row-parallel partial
};

/// The DFlash2 draft model on this rank (dflash2.Drafter's constructor); notes/glm_weights.md#draft
pub const Draft = struct {
    d: usize, // hidden size
    hd: usize, // head dim
    heads: usize, // this rank's query heads
    kvh: usize, // and KV heads
    inter: usize, // and MLP width
    gs: usize, // conv group size
    block: usize,
    top_k: usize,
    window: usize, // sliding_window - 1
    causal: bool,
    eps: f32,
    theta: f64,
    mask_id: u32,
    sel_rank: usize, // the selector's projection width
    taps: [5]u32, // target layers whose stream means feed the drafter
    fc: kern.Q4, // [D, 5 D]
    hidden_norm: u64,
    norm: u64,
    hproj: u64, // [sel_rank, D] bf16
    inv_freq: u64, // [hd / 2] fp32
    layers: []DraftLayer,
    pred: []f32 = &.{}, // [vocab, sel_rank] host codebooks (rank 0 only)
    succ: []f32 = &.{},
    bytes: usize = 0,
};

pub const Weights = struct {
    cfg: Config,
    rank: u32,
    embed: u64, // [vocab, D] bf16, whole on both ranks
    layers: []Layer,
    norm: u64,
    head: Mat, // this rank's half of the vocabulary
    draft_head: kern.Q4, // the head quantized to 4 bits (min/max ranges) from its BF16 rows, read by draft steps only
    mtp: ?Mtp,
    draft: ?Draft = null,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(gpa);
        if (w.draft) |d| {
            gpa.free(d.layers);
            gpa.free(d.pred);
            gpa.free(d.succ);
        }
        gpa.free(w.layers);
        w.* = undefined;
    }
};

/// split.py's rule for a tensor name (EXL3_RULES for expert parts first).
pub const Split = enum { rep, row, col, dim1 };

pub fn rule(name: []const u8) Split {
    const has = std.mem.indexOf;
    if (has(u8, name, ".mlp.experts.") != null) {
        const down = has(u8, name, ".down_proj.") != null;
        if (std.mem.endsWith(u8, name, ".trellis")) return if (down) .row else .dim1;
        if (std.mem.endsWith(u8, name, ".suh")) return if (down) .row else .rep;
        if (std.mem.endsWith(u8, name, ".svh")) return if (down) .rep else .row;
        return .rep;
    }
    const cols = [_][]const u8{ ".mlp.shared_experts.down_proj.", ".mlp.down_proj.", ".self_attn.o_proj." };
    for (cols) |c| if (has(u8, name, c) != null) return .col;
    const rows = [_][]const u8{
        ".mlp.shared_experts.gate_proj.", ".mlp.shared_experts.up_proj.", ".mlp.gate_proj.",       ".mlp.up_proj.",
        ".self_attn.q_proj.",             ".self_attn.k_proj.",           ".self_attn.v_proj.",    ".self_attn.q_conv1d.",
        ".self_attn.k_conv1d.",           ".self_attn.v_conv1d.",         ".self_attn.f_b_proj.",  ".self_attn.g_b_proj.",
        ".self_attn.b_proj.",             ".self_attn.q_b_proj.",         ".self_attn.kv_b_proj.",
    };
    for (rows) |r| if (has(u8, name, r) != null) return .row;
    if (std.mem.endsWith(u8, name, ".self_attn.A_log") or std.mem.endsWith(u8, name, ".self_attn.dt_bias")) return .row;
    return .rep;
}

const Loader = loader.Loader;

/// TF_GLM_DENSE (qmm.dense_kind): `q4`, the default, or `b16`. The FP8 mode of Python's `fp8` is not ported.
pub fn denseKind() !Kind {
    const v = std.c.getenv("TF_GLM_DENSE") orelse return .q4;
    const s = std.mem.span(v);
    if (s.len == 0 or std.mem.eql(u8, s, "q4")) return .q4;
    if (std.mem.eql(u8, s, "b16") or std.mem.eql(u8, s, "bf16")) return .b16;
    std.log.err("TF_GLM_DENSE is q4 or b16, not {s}", .{s});
    return error.BadDenseKind;
}

/// Every weight of rank `rank` of the checkpoint in `dir`, uploaded on `ops`'s stream, dense ones as TF_GLM_DENSE says.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ops: @import("glm_source.zig").Ops, kernels: *const kern.Kernels, dir: []const u8, cfg: Config, rank: u32, mtp: bool) !Weights {
    // by prefix, so the vision tower's tensors (5-D convolutions past safetensors.max_rank) are never parsed
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
    defer ck.close();
    var ck_head = try core.Checkpoint.openModelPrefix(gpa, io, dir, "lm_head.");
    defer ck_head.close();
    var src = try Source.init(gpa, ops);
    defer src.deinit();
    try src.add(&ck);
    try src.add(&ck_head);
    var w: Weights = .{ .cfg = cfg, .rank = rank, .embed = 0, .layers = try gpa.alloc(Layer, cfg.layers), .norm = 0, .head = undefined, .draft_head = undefined, .mtp = null };
    errdefer w.deinit(gpa);
    var l: Loader = .{ .gpa = gpa, .ck = &ck, .src = &src, .d = d, .rank = rank, .w = &w, .kernels = kernels, .dense = try denseKind() };
    defer l.host.deinit(gpa);
    w.embed = try l.upOne(prefix ++ "embed_tokens.weight", .bf16);
    for (w.layers, 0..) |*lw, i| lw.* = try l.layer(i, false);
    w.norm = try l.upOne(prefix ++ "norm.weight", .bf16);
    // the head: this rank's half of the vocabulary rows in BF16, then quantized; notes/glm_weights.md#head
    const head = try ck_head.expect("lm_head.weight", .bf16, &.{ cfg.vocab, cfg.hidden });
    const vl: usize = cfg.vocab / world;
    const hk: usize = cfg.hidden;
    const half = head.bytes[rank * vl * hk * 2 ..][0 .. vl * hk * 2];
    var raw = try cuda.DeviceBuffer.alloc(d, half.len);
    defer raw.free();
    try src.upload(raw.ptr, half);
    try src.flush();
    w.draft_head = (try l.quantize4Draft(raw.ptr, vl, hk));
    if (l.dense == .b16) {
        w.head = .{ .ptr = try l.alloc(half.len), .n = vl, .k = hk };
        try ops.d.check(ops.d.api.cuMemcpyDtoDAsync_v2(w.head.ptr, raw.ptr, half.len, ops.s.handle), "cuMemcpyDtoDAsync");
        try ops.s.synchronize();
    } else w.head = try l.f8(raw.ptr, vl, hk);
    if (cfg.mtp > 0 and mtp) {
        const i = cfg.layers;
        const n = try l.layerNames(i, &.{ "enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight" });
        defer l.free(&n);
        w.mtp = .{ .enorm = try l.upOne(n[0], .bf16), .hnorm = try l.upOne(n[1], .bf16), .eh = try l.mat(n[2..3]), .head_norm = try l.upOne(n[3], .bf16), .layer = try l.layer(i, true) };
    }
    try src.flush();
    try ops.s.synchronize();
    return w;
}

/// The tensors `tf-glm qdump` quantizes: v1.8's TF_GLM_DENSE=q4 stacks (zig/tests/cuda/glm/qdump.py has the same list).
const dump_sets = [_]struct { name: []const u8, parts: []const []const u8 }{
    .{ .name = "l0_kda_proj", .parts = &.{ "layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj", "layers.0.self_attn.f_a_proj", "layers.0.self_attn.g_a_proj", "layers.0.self_attn.b_proj" } },
    .{ .name = "l0_kda_fb", .parts = &.{"layers.0.self_attn.f_b_proj"} },
    .{ .name = "l0_kda_o", .parts = &.{"layers.0.self_attn.o_proj"} },
    .{ .name = "l0_mlp_gu", .parts = &.{ "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj" } },
    .{ .name = "l0_mlp_down", .parts = &.{"layers.0.mlp.down_proj"} },
    .{ .name = "l3_dsa_proj", .parts = &.{ "layers.3.self_attn.q_a_proj", "layers.3.self_attn.kv_a_proj_with_mqa" } },
    .{ .name = "l3_dsa_qb", .parts = &.{"layers.3.self_attn.q_b_proj"} },
    .{ .name = "l3_dsa_o", .parts = &.{"layers.3.self_attn.o_proj"} },
    .{ .name = "l3_ix_kw", .parts = &.{ "layers.3.self_attn.indexer.wk", "layers.3.self_attn.indexer.weights_proj" } },
    .{ .name = "l3_ix_qb", .parts = &.{"layers.3.self_attn.indexer.wq_b"} },
    .{ .name = "l3_sh_gu", .parts = &.{ "layers.3.mlp.shared_experts.gate_proj", "layers.3.mlp.shared_experts.up_proj" } },
    .{ .name = "l3_sh_down", .parts = &.{"layers.3.mlp.shared_experts.down_proj"} },
    .{ .name = "mtp_eh", .parts = &.{"layers.45.eh_proj"} },
};

/// `tf-glm qdump`: the dense quantization of a few tensors of `rank` as raw bytes in `out`; notes/glm_weights.md#qdump
pub fn dumpQuant(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ops: @import("glm_source.zig").Ops, kernels: *const kern.Kernels, dir: []const u8, cfg: Config, rank: u32, out: []const u8, only: []const u8) !void {
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
    defer ck.close();
    var ck_head = try core.Checkpoint.openModelPrefix(gpa, io, dir, "lm_head.");
    defer ck_head.close();
    var src = try Source.init(gpa, ops);
    defer src.deinit();
    try src.add(&ck);
    try src.add(&ck_head);
    var w: Weights = .{ .cfg = cfg, .rank = rank, .embed = 0, .layers = &.{}, .norm = 0, .head = undefined, .draft_head = undefined, .mtp = null };
    defer for (w.buffers.items) |*b| b.free();
    defer w.buffers.deinit(gpa);
    var l: Loader = .{ .gpa = gpa, .ck = &ck, .src = &src, .d = d, .rank = rank, .w = &w, .kernels = kernels, .dense = .q4 };
    defer l.host.deinit(gpa);
    try std.Io.Dir.cwd().createDirPath(io, out);
    const want = struct {
        fn has(list: []const u8, name: []const u8) bool {
            if (list.len == 0) return true;
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |n| if (std.mem.eql(u8, n, name)) return true;
            return false;
        }
    };
    for (dump_sets) |set| {
        if (!want.has(only, set.name)) continue;
        var names: [6][]const u8 = undefined;
        for (set.parts, 0..) |p, i| names[i] = try std.fmt.allocPrint(gpa, prefix ++ "{s}.weight", .{p});
        defer for (names[0..set.parts.len]) |n| gpa.free(n);
        const m = try l.mat(names[0..set.parts.len]);
        const kg = m.k / 64;
        try dumpFile(gpa, io, d, ops, out, set.name, "w", m.ptr, m.npad * m.k / 2);
        try dumpFile(gpa, io, d, ops, out, set.name, "s", m.s, kg * m.npad * 2);
        try dumpFile(gpa, io, d, ops, out, set.name, "b", m.b, kg * m.npad * 2);
        std.debug.print("{s}: {d} x {d}\n", .{ set.name, m.n, m.k });
    }
    if (want.has(only, "head") or want.has(only, "draft")) {
        const head = try ck_head.expect("lm_head.weight", .bf16, &.{ cfg.vocab, cfg.hidden });
        const vl: usize = cfg.vocab / world;
        const hk: usize = cfg.hidden;
        const half = head.bytes[rank * vl * hk * 2 ..][0 .. vl * hk * 2];
        var raw = try cuda.DeviceBuffer.alloc(d, half.len);
        defer raw.free();
        try src.upload(raw.ptr, half);
        try src.flush();
        if (want.has(only, "head")) {
            const f = try l.f8(raw.ptr, vl, hk);
            try dumpFile(gpa, io, d, ops, out, "head", "q8", f.ptr, vl * hk);
            try dumpFile(gpa, io, d, ops, out, "head", "sc", f.s, vl * hk / 32);
        }
        if (want.has(only, "draft")) {
            const q = try l.quantize4Draft(raw.ptr, vl, hk);
            try dumpFile(gpa, io, d, ops, out, "draft", "w", q.w, q.npad * q.k / 2);
            try dumpFile(gpa, io, d, ops, out, "draft", "s", q.s, q.k / 64 * q.npad * 2);
            try dumpFile(gpa, io, d, ops, out, "draft", "b", q.b, q.k / 64 * q.npad * 2);
        }
    }
}

fn dumpFile(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ops: @import("glm_source.zig").Ops, dir: []const u8, name: []const u8, ext: []const u8, ptr: u64, bytes: usize) !void {
    const host = try gpa.alloc(u8, bytes);
    defer gpa.free(host);
    try d.check(d.api.cuMemcpyDtoHAsync_v2(host.ptr, ptr, bytes, ops.s.handle), "cuMemcpyDtoHAsync");
    try ops.s.synchronize();
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}.{s}", .{ dir, name, ext });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = host });
}

test "split rules match split.py on the tensors GLM reads" {
    const t = std.testing;
    try t.expectEqual(Split.dim1, rule("model.language_model.layers.3.mlp.experts.0.gate_proj.trellis"));
    try t.expectEqual(Split.dim1, rule("model.language_model.layers.3.mlp.experts.7.up_proj.trellis"));
    try t.expectEqual(Split.rep, rule("model.language_model.layers.3.mlp.experts.0.up_proj.suh"));
    try t.expectEqual(Split.row, rule("model.language_model.layers.3.mlp.experts.0.gate_proj.svh"));
    try t.expectEqual(Split.row, rule("model.language_model.layers.3.mlp.experts.0.down_proj.trellis"));
    try t.expectEqual(Split.row, rule("model.language_model.layers.3.mlp.experts.0.down_proj.suh"));
    try t.expectEqual(Split.rep, rule("model.language_model.layers.3.mlp.experts.0.down_proj.svh"));
    try t.expectEqual(Split.rep, rule("model.language_model.layers.3.mlp.experts.0.down_proj.mcg"));
    try t.expectEqual(Split.col, rule("model.language_model.layers.3.mlp.shared_experts.down_proj.weight"));
    try t.expectEqual(Split.row, rule("model.language_model.layers.3.mlp.shared_experts.up_proj.weight"));
    try t.expectEqual(Split.col, rule("model.language_model.layers.0.mlp.down_proj.weight"));
    try t.expectEqual(Split.row, rule("model.language_model.layers.0.mlp.gate_proj.weight"));
    try t.expectEqual(Split.col, rule("model.language_model.layers.0.self_attn.o_proj.weight"));
    try t.expectEqual(Split.row, rule("model.language_model.layers.0.self_attn.dt_bias"));
    try t.expectEqual(Split.row, rule("model.language_model.layers.0.self_attn.A_log"));
    try t.expectEqual(Split.rep, rule("model.language_model.layers.0.self_attn.f_a_proj.weight"));
    try t.expectEqual(Split.row, rule("model.language_model.layers.0.self_attn.f_b_proj.weight"));
    try t.expectEqual(Split.rep, rule("model.language_model.layers.0.self_attn.o_norm.weight"));
    try t.expectEqual(Split.row, rule("model.language_model.layers.3.self_attn.kv_b_proj.weight"));
    try t.expectEqual(Split.rep, rule("model.language_model.layers.3.self_attn.indexer.wq_b.weight"));
    try t.expectEqual(Split.rep, rule("model.language_model.layers.3.self_attn.q_a_proj.weight"));
    try t.expectEqual(Split.rep, rule("model.language_model.layers.45.eh_proj.weight"));
    try t.expectEqual(Split.rep, rule("model.language_model.layers.3.hc_attn_fn"));
    try t.expectEqual(Split.rep, rule("model.language_model.layers.3.mlp.gate.weight"));
}

pub const loadDraft = draft.loadDraft;
