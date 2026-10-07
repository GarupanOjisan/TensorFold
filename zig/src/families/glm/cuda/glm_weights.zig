//! One rank's half of GLM-5.3-Flash's EXL3 checkpoint on the GPU, laid out as the Python engine lays it out
//! (v0.6.6 families/glm5_next/cuda/weights.py `load`, split.py `rule`): BF16 projections, rows or columns split
//! between the two ranks, the routed experts' EXL3 trellises stacked per layer with pointer and width tables.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm_config = @import("glm_config");
const Source = @import("glm_source.zig").Source;

pub const Config = glm_config.Config;
pub const world = 2;
const prefix = "model.language_model.";

/// A BF16 matrix [n, k], row-major (qmm.B16).
pub const B16 = struct { ptr: u64, n: usize, k: usize };

pub const Hc = struct { fn_: u64, base: u64, scale: u64 }; // [24, S*D] bf16, [24] f32, [3] f32

pub const Kda = struct {
    proj: B16, // [q | k | v | f_a | g_a | b] of this rank
    fa_off: usize,
    ga_off: usize,
    b_off: usize,
    fb: B16,
    gb: B16,
    conv: u64, // [3 * LL * 128, taps] bf16
    a_log: u64, // [LL] f32
    dt_bias: u64, // [LL * 128] f32
    norm: u64, // [128] bf16
    o: B16,
};

pub const Index = struct {
    kw: B16, // [wk | weights_proj]
    qb: B16,
    ln_w: u64,
    ln_b: u64,
    gate: u64, // index_kpool_compress_gate [128, D] bf16
    ape: u64, // index_kpool_compress_ape [kpool, 128] bf16
};

pub const Dsa = struct {
    proj: B16, // [q_a | kv_a], replicated
    q_norm: u64,
    kv_norm: u64,
    q_b: B16,
    o: B16,
    index: Index,
    wk: u64, // AbsorbW: kv_b's key rows [HL, qk_dim, kv_lora] bf16
    wv: u64, // and its value rows [HL, v_dim, kv_lora]
};

pub const Mlp = struct { gu: B16, down: B16, width: usize };

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

pub const Mtp = struct { enorm: u64, hnorm: u64, eh: B16, head_norm: u64, layer: Layer };

pub const Weights = struct {
    cfg: Config,
    rank: u32,
    embed: u64, // [vocab, D] bf16, whole on both ranks
    layers: []Layer,
    norm: u64,
    head: B16, // this rank's half of the vocabulary
    mtp: ?Mtp,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(gpa);
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
        ".mlp.shared_experts.gate_proj.", ".mlp.shared_experts.up_proj.", ".mlp.gate_proj.", ".mlp.up_proj.",
        ".self_attn.q_proj.",             ".self_attn.k_proj.",           ".self_attn.v_proj.", ".self_attn.q_conv1d.",
        ".self_attn.k_conv1d.",           ".self_attn.v_conv1d.",         ".self_attn.f_b_proj.", ".self_attn.g_b_proj.",
        ".self_attn.b_proj.",             ".self_attn.q_b_proj.",         ".self_attn.kv_b_proj.",
    };
    for (rows) |r| if (has(u8, name, r) != null) return .row;
    if (std.mem.endsWith(u8, name, ".self_attn.A_log") or std.mem.endsWith(u8, name, ".self_attn.dt_bias")) return .row;
    return .rep;
}

const Loader = struct {
    gpa: std.mem.Allocator,
    ck: *core.Checkpoint,
    src: *Source,
    d: *const cuda.Driver,
    rank: u32,
    w: *Weights,
    host: std.ArrayList(u8) = .empty, // staging for strided splits and stacks

    fn name(l: *Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(l.gpa, fmt, args);
    }

    fn alloc(l: *Loader, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(l.d, @max(bytes, 256));
        errdefer b.free();
        try l.w.buffers.append(l.gpa, b);
        l.w.bytes += bytes;
        return b.ptr;
    }

    /// The rank's part of a tensor: its bytes on the host (`out` grows to fit), and its shape.
    fn part(l: *Loader, t: core.checkpoint.Tensor, kind: Split, out: *std.ArrayList(u8)) ![core.safetensors.max_rank]usize {
        var shape = t.shape;
        const item = t.dtype.size();
        switch (kind) {
            .rep => {
                try out.resize(l.gpa, t.bytes.len);
                try l.src.read(out.items, t.bytes);
            },
            .row => {
                if (t.shape[0] % 2 != 0) return error.OddSplit;
                const half = t.bytes.len / 2;
                try out.resize(l.gpa, half);
                try l.src.read(out.items, t.bytes[l.rank * half ..][0..half]);
                shape[0] /= 2;
            },
            .col, .dim1 => {
                if (t.rank < 2 or t.shape[1] % 2 != 0) return error.OddSplit;
                var inner: usize = item;
                for (t.shape[2..t.rank]) |s| inner *= s;
                const row = t.shape[1] * inner;
                const keep = row / 2;
                try l.host.resize(l.gpa, t.bytes.len);
                try l.src.read(l.host.items, t.bytes);
                try out.resize(l.gpa, t.bytes.len / 2);
                for (0..t.shape[0]) |r| @memcpy(out.items[r * keep ..][0..keep], l.host.items[r * row + l.rank * keep ..][0..keep]);
                shape[1] /= 2;
            },
        }
        return shape;
    }

    fn get(l: *Loader, comptime fmt: []const u8, args: anytype) !core.checkpoint.Tensor {
        const n = try l.name(fmt, args);
        defer l.gpa.free(n);
        return l.ck.get(n);
    }

    /// Tensors' rank parts concatenated (rows of several matrices with the same K, in order) and uploaded.
    fn stack(l: *Loader, names: []const []const u8, dtype: core.safetensors.DType) !struct { ptr: u64, rows: usize, cols: usize } {
        var all: std.ArrayList(u8) = .empty;
        defer all.deinit(l.gpa);
        var one: std.ArrayList(u8) = .empty;
        defer one.deinit(l.gpa);
        var rows: usize = 0;
        var cols: usize = 0;
        for (names) |n| {
            const t = try l.ck.get(n);
            if (t.dtype != dtype) {
                std.log.err("{s}: {t}, expected {t}", .{ n, t.dtype, dtype });
                return error.UnexpectedTensor;
            }
            const shape = try l.part(t, rule(n), &one);
            var c: usize = 1;
            for (shape[1..t.rank]) |s| c *= s;
            if (cols != 0 and c != cols) return error.StackWidth;
            cols = c;
            rows += shape[0];
            try all.appendSlice(l.gpa, one.items);
        }
        const ptr = try l.alloc(all.items.len);
        try l.src.ops.upload(ptr, all.items);
        return .{ .ptr = ptr, .rows = rows, .cols = cols };
    }

    fn upOne(l: *Loader, full: []const u8, dtype: core.safetensors.DType) !u64 {
        return (try l.stack(&.{full}, dtype)).ptr;
    }

    fn b16(l: *Loader, names: []const []const u8) !B16 {
        const s = try l.stack(names, .bf16);
        return .{ .ptr = s.ptr, .n = s.rows, .k = s.cols };
    }

    fn hc(l: *Loader, i: usize, site: []const u8) !Hc {
        const f = try l.name(prefix ++ "layers.{d}.hc_{s}_fn", .{ i, site });
        defer l.gpa.free(f);
        const b = try l.name(prefix ++ "layers.{d}.hc_{s}_base", .{ i, site });
        defer l.gpa.free(b);
        const s = try l.name(prefix ++ "layers.{d}.hc_{s}_scale", .{ i, site });
        defer l.gpa.free(s);
        return .{ .fn_ = try l.upOne(f, .bf16), .base = try l.upOne(b, .f32), .scale = try l.upOne(s, .f32) };
    }

    fn layerNames(l: *Loader, i: usize, comptime parts: []const []const u8) ![parts.len][]const u8 {
        var out: [parts.len][]const u8 = undefined;
        inline for (parts, 0..) |p, j| out[j] = try l.name(prefix ++ "layers.{d}." ++ p, .{i});
        return out;
    }

    fn free(l: *Loader, ns: []const []const u8) void {
        for (ns) |n| l.gpa.free(n);
    }

    fn kda(l: *Loader, i: usize) !Kda {
        const c = &l.w.cfg;
        const ll = c.kda_heads / world;
        const p = try l.layerNames(i, &.{ "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.f_a_proj.weight", "self_attn.g_a_proj.weight", "self_attn.b_proj.weight" });
        defer l.free(&p);
        const proj = try l.b16(&p);
        const cv = try l.layerNames(i, &.{ "self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight" });
        defer l.free(&cv);
        const conv = try l.stack(&cv, .bf16);
        const r = try l.layerNames(i, &.{ "self_attn.f_b_proj.weight", "self_attn.g_b_proj.weight", "self_attn.A_log", "self_attn.dt_bias", "self_attn.o_norm.weight", "self_attn.o_proj.weight" });
        defer l.free(&r);
        const width = 3 * ll * c.kda_dim;
        if (conv.rows != width or proj.n != width + 2 * 128 + ll) return error.KdaShape;
        return .{
            .proj = proj,
            .fa_off = width,
            .ga_off = width + 128,
            .b_off = width + 256,
            .fb = try l.b16(r[0..1]),
            .gb = try l.b16(r[1..2]),
            .conv = conv.ptr,
            .a_log = try l.upOne(r[2], .f32),
            .dt_bias = try l.upOne(r[3], .f32),
            .norm = try l.upOne(r[4], .bf16),
            .o = try l.b16(r[5..6]),
        };
    }

    fn dsa(l: *Loader, i: usize) !Dsa {
        const c = &l.w.cfg;
        const hl = c.mla_heads / world;
        const p = try l.layerNames(i, &.{ "self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight", "self_attn.q_a_layernorm.weight", "self_attn.kv_a_layernorm.weight", "self_attn.q_b_proj.weight", "self_attn.o_proj.weight", "self_attn.kv_b_proj.weight" });
        defer l.free(&p);
        const x = try l.layerNames(i, &.{ "self_attn.indexer.wk.weight", "self_attn.indexer.weights_proj.weight", "self_attn.indexer.wq_b.weight", "self_attn.indexer.k_norm.weight", "self_attn.indexer.k_norm.bias", "self_attn.indexer.index_kpool_compress_gate", "self_attn.indexer.index_kpool_compress_ape" });
        defer l.free(&x);
        // kv_b's rank rows, head by head [qk_dim key rows | v_dim value rows]: the key and value blocks apart
        var kvb: std.ArrayList(u8) = .empty;
        defer kvb.deinit(l.gpa);
        const t = try l.ck.get(p[6]);
        const shape = try l.part(t, .row, &kvb);
        const per = c.nope + c.v_dim;
        if (t.dtype != .bf16 or shape[0] != hl * per or shape[1] != c.kv_lora) return error.KvbShape;
        const row = c.kv_lora * 2;
        const wk = try l.gpa.alloc(u8, hl * c.nope * row);
        defer l.gpa.free(wk);
        const wv = try l.gpa.alloc(u8, hl * c.v_dim * row);
        defer l.gpa.free(wv);
        for (0..@as(usize, hl)) |h| {
            @memcpy(wk[h * c.nope * row ..][0 .. c.nope * row], kvb.items[h * per * row ..][0 .. c.nope * row]);
            @memcpy(wv[h * c.v_dim * row ..][0 .. c.v_dim * row], kvb.items[(h * per + c.nope) * row ..][0 .. c.v_dim * row]);
        }
        const wk_ptr = try l.alloc(wk.len);
        try l.src.ops.upload(wk_ptr, wk);
        const wv_ptr = try l.alloc(wv.len);
        try l.src.ops.upload(wv_ptr, wv);
        return .{
            .proj = try l.b16(p[0..2]),
            .q_norm = try l.upOne(p[2], .bf16),
            .kv_norm = try l.upOne(p[3], .bf16),
            .q_b = try l.b16(p[4..5]),
            .o = try l.b16(p[5..6]),
            .index = .{
                .kw = try l.b16(x[0..2]),
                .qb = try l.b16(x[2..3]),
                .ln_w = try l.upOne(x[3], .bf16),
                .ln_b = try l.upOne(x[4], .bf16),
                .gate = try l.upOne(x[5], .bf16),
                .ape = try l.upOne(x[6], .bf16),
            },
            .wk = wk_ptr,
            .wv = wv_ptr,
        };
    }

    fn mlp(l: *Loader, i: usize, comptime sub: []const u8) !Mlp {
        const p = try l.layerNames(i, &.{ sub ++ "gate_proj.weight", sub ++ "up_proj.weight", sub ++ "down_proj.weight" });
        defer l.free(&p);
        const gu = try l.b16(p[0..2]);
        return .{ .gu = gu, .down = try l.b16(p[2..3]), .width = gu.n / 2 };
    }

    /// The routed experts of layer `i` (experts.prepare): trellises stacked per projection, suh/svh stacked.
    fn exl3(l: *Loader, i: usize) !Exl3 {
        const c = &l.w.cfg;
        const e_n: usize = c.experts;
        const d: usize = c.hidden;
        const width: usize = c.moe_inter / world;
        var k2: u32 = 0;
        var ptrs: [3]u64 = undefined;
        var k2s: [3]u64 = undefined;
        var scales: [6]u64 = undefined; // suh_g, svh_g, suh_u, svh_u, suh_d, svh_d
        const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(l.gpa);
        var one_part: std.ArrayList(u8) = .empty;
        defer one_part.deinit(l.gpa);
        for (projs, 0..) |proj, j| {
            const kk: usize = if (j == 2) width else d;
            const nn: usize = if (j == 2) d else width;
            // trellises [E, K/16, N/16, 16 * k2 / 2]: one device block, an expert's at e * each
            var each: usize = 0;
            var block: u64 = 0;
            const table = try l.gpa.alloc(i64, e_n);
            defer l.gpa.free(table);
            const widths = try l.gpa.alloc(i32, e_n);
            defer l.gpa.free(widths);
            for (0..e_n) |e| {
                const n = try l.name(prefix ++ "layers.{d}.mlp.experts.{d}.{s}.trellis", .{ i, e, proj });
                defer l.gpa.free(n);
                const t = try l.ck.get(n);
                const shape = try l.part(t, rule(n), &one_part);
                if (t.dtype != .i16 or shape[0] * 16 != kk or shape[1] * 16 != nn or shape[2] % 8 != 0) return error.TrellisShape;
                const this_k2: u32 = @intCast(shape[2] / 8);
                if (k2 == 0) k2 = this_k2;
                if (this_k2 != k2) return error.MixedTrellisWidths;
                if (e == 0) {
                    each = one_part.items.len;
                    block = try l.alloc(each * e_n);
                }
                if (one_part.items.len != each) return error.TrellisShape;
                try l.src.ops.upload(block + e * each, one_part.items);
                table[e] = @intCast(block + e * each);
                widths[e] = @intCast(this_k2);
            }
            ptrs[j] = try l.alloc(e_n * 8);
            try l.src.ops.upload(ptrs[j], std.mem.sliceAsBytes(table));
            k2s[j] = try l.alloc(e_n * 4);
            try l.src.ops.upload(k2s[j], std.mem.sliceAsBytes(widths));
            // suh [E, K], svh [E, N] in f16
            for ([_][]const u8{ "suh", "svh" }, 0..) |part_name, s| {
                bytes.clearRetainingCapacity();
                for (0..e_n) |e| {
                    const n = try l.name(prefix ++ "layers.{d}.mlp.experts.{d}.{s}.{s}", .{ i, e, proj, part_name });
                    defer l.gpa.free(n);
                    const t = try l.ck.get(n);
                    if (t.dtype != .f16) return error.ScaleType;
                    _ = try l.part(t, rule(n), &one_part);
                    try bytes.appendSlice(l.gpa, one_part.items);
                }
                const want = e_n * (if (s == 0) kk else nn) * 2;
                if (bytes.items.len != want) return error.ScaleShape;
                scales[j * 2 + s] = try l.alloc(bytes.items.len);
                try l.src.ops.upload(scales[j * 2 + s], bytes.items);
            }
            // the codebook marker (I32 [1]) is replicated and unread, as in the Python loader
            for (0..e_n) |e| {
                const n = try l.name(prefix ++ "layers.{d}.mlp.experts.{d}.{s}.mcg", .{ i, e, proj });
                defer l.gpa.free(n);
                _ = l.ck.get(n) catch {};
            }
        }
        return .{
            .gate_ptr = ptrs[0],
            .up_ptr = ptrs[1],
            .down_ptr = ptrs[2],
            .gate_k2 = k2s[0],
            .up_k2 = k2s[1],
            .down_k2 = k2s[2],
            .suh_g = scales[0],
            .svh_g = scales[1],
            .suh_u = scales[2],
            .svh_u = scales[3],
            .suh_d = scales[4],
            .svh_d = scales[5],
            .count = e_n,
            .dims = d,
            .width = width,
            .k2 = k2,
        };
    }

    fn moe(l: *Loader, i: usize) !Moe {
        const p = try l.layerNames(i, &.{ "mlp.gate.weight", "mlp.gate.e_score_correction_bias" });
        defer l.free(&p);
        return .{ .router = try l.upOne(p[0], .bf16), .bias = try l.upOne(p[1], .f32), .ex = try l.exl3(i), .shared = try l.mlp(i, "mlp.shared_experts.") };
    }

    /// weights.load's `layer`; `plain`: the MTP layer (DSA and MoE, no hyper-connections).
    fn layer(l: *Loader, i: usize, plain: bool) !Layer {
        const c = &l.w.cfg;
        const n = try l.layerNames(i, &.{ "input_layernorm.weight", "post_attention_layernorm.weight" });
        defer l.free(&n);
        const kind: glm_config.Kind = if (plain) .mla else c.kind(@intCast(i));
        var lw: Layer = .{
            .index = @intCast(i),
            .kind = kind,
            .attn_hc = if (plain) null else try l.hc(i, "attn"),
            .ffn_hc = if (plain) null else try l.hc(i, "ffn"),
            .in_norm = try l.upOne(n[0], .bf16),
            .post_norm = try l.upOne(n[1], .bf16),
        };
        if (kind == .kda) lw.kda = try l.kda(i) else lw.dsa = try l.dsa(i);
        if (plain or c.isMoe(@intCast(i))) lw.moe = try l.moe(i) else lw.mlp = try l.mlp(i, "mlp.");
        return lw;
    }
};

/// Every weight of rank `rank` of the checkpoint in `dir`, uploaded on `ops`'s stream.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ops: @import("glm_source.zig").Ops, dir: []const u8, cfg: Config, rank: u32, mtp: bool) !Weights {
    // by prefix, so the vision tower's tensors (5-D convolutions past safetensors.max_rank) are never parsed
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
    defer ck.close();
    var ck_head = try core.Checkpoint.openModelPrefix(gpa, io, dir, "lm_head.");
    defer ck_head.close();
    var src = try Source.init(gpa, ops);
    defer src.deinit();
    try src.add(&ck);
    try src.add(&ck_head);
    var w: Weights = .{ .cfg = cfg, .rank = rank, .embed = 0, .layers = try gpa.alloc(Layer, cfg.layers), .norm = 0, .head = undefined, .mtp = null };
    errdefer w.deinit(gpa);
    var l: Loader = .{ .gpa = gpa, .ck = &ck, .src = &src, .d = d, .rank = rank, .w = &w };
    defer l.host.deinit(gpa);
    w.embed = try l.upOne(prefix ++ "embed_tokens.weight", .bf16);
    for (w.layers, 0..) |*lw, i| lw.* = try l.layer(i, false);
    w.norm = try l.upOne(prefix ++ "norm.weight", .bf16);
    // the head: this rank's half of the vocabulary rows
    const head = try ck_head.expect("lm_head.weight", .bf16, &.{ cfg.vocab, cfg.hidden });
    const vl = cfg.vocab / world;
    const half = head.bytes[rank * vl * cfg.hidden * 2 ..][0 .. vl * cfg.hidden * 2];
    w.head = .{ .ptr = try l.alloc(half.len), .n = vl, .k = cfg.hidden };
    try src.upload(w.head.ptr, half);
    if (cfg.mtp > 0 and mtp) {
        const i = cfg.layers;
        const n = try l.layerNames(i, &.{ "enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight" });
        defer l.free(&n);
        w.mtp = .{ .enorm = try l.upOne(n[0], .bf16), .hnorm = try l.upOne(n[1], .bf16), .eh = try l.b16(n[2..3]), .head_norm = try l.upOne(n[3], .bf16), .layer = try l.layer(i, true) };
    }
    try src.flush();
    try ops.s.synchronize();
    return w;
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
