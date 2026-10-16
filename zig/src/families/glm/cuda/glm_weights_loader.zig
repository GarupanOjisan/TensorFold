//! The loader of one rank's weights (glm_weights.zig): tensor parts, stacks, dense quantization and the layers.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm_config = @import("glm_config");
const Source = @import("glm_source.zig").Source;
const kern = @import("glm_kernels.zig");
const wt = @import("glm_weights.zig");

const world = wt.world;
const prefix = wt.prefix;
const Kind = wt.Kind;
const Split = wt.Split;
const rule = wt.rule;
const Mat = wt.Mat;
const Hc = wt.Hc;
const Kda = wt.Kda;
const Dsa = wt.Dsa;
const Mlp = wt.Mlp;
const Exl3 = wt.Exl3;
const Moe = wt.Moe;
const Layer = wt.Layer;
const Weights = wt.Weights;

pub const Loader = struct {
    gpa: std.mem.Allocator,
    ck: *core.Checkpoint,
    src: *Source,
    d: *const cuda.Driver,
    rank: u32,
    w: *Weights,
    kernels: *const kern.Kernels,
    dense: Kind,
    host: std.ArrayList(u8) = .empty, // staging for strided splits and stacks

    fn name(l: *Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(l.gpa, fmt, args);
    }

    pub fn alloc(l: *Loader, bytes: usize) !u64 {
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
        const shape = try l.stackHost(names, dtype, &all);
        const ptr = try l.alloc(all.items.len);
        try l.src.ops.upload(ptr, all.items);
        return .{ .ptr = ptr, .rows = shape.rows, .cols = shape.cols };
    }

    /// `stack`'s rank parts in `all`, not yet uploaded.
    fn stackHost(l: *Loader, names: []const []const u8, dtype: core.safetensors.DType, all: *std.ArrayList(u8)) !struct { rows: usize, cols: usize } {
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
        return .{ .rows = rows, .cols = cols };
    }

    pub fn upOne(l: *Loader, full: []const u8, dtype: core.safetensors.DType) !u64 {
        return (try l.stack(&.{full}, dtype)).ptr;
    }

    /// A dense projection (weights.py `q4`, `stack`) of the rank parts of `names`; notes/glm_weights.md#mat-loader
    pub fn mat(l: *Loader, names: []const []const u8) !Mat {
        var all: std.ArrayList(u8) = .empty;
        defer all.deinit(l.gpa);
        const shape = try l.stackHost(names, .bf16, &all);
        if (l.dense == .b16) {
            const ptr = try l.alloc(all.items.len);
            try l.src.ops.upload(ptr, all.items);
            return .{ .ptr = ptr, .n = shape.rows, .k = shape.cols };
        }
        var raw = try cuda.DeviceBuffer.alloc(l.d, all.items.len);
        defer raw.free();
        try l.src.ops.upload(raw.ptr, all.items);
        return l.quantize(raw.ptr, shape.rows, shape.cols, true);
    }

    /// Bf16 rows [n, k] on the device as a dense q4 projection (make_dense_q4's rule); notes/glm_weights.md#quantize
    fn quantize(l: *Loader, raw: u64, n: usize, k: usize, mse: bool) !Mat {
        const s = l.src.ops.s;
        if (k % 64 == 0) {
            const at = try l.alloc(kern.q4Bytes(n, k));
            var tmp = try cuda.DeviceBuffer.alloc(l.d, n * k / 2 + 4 * n * (k / 64));
            defer tmp.free();
            const q = try kern.quantize4(l.kernels, s, if (mse) .mse else .minmax, raw, n, k, at, tmp.ptr);
            try s.synchronize();
            return .{ .kind = .q4, .ptr = q.w, .n = n, .k = k, .s = q.s, .b = q.b, .npad = q.npad };
        }
        return l.f8(raw, n, k);
    }

    /// qmm.draft_quantize4 (TF_GLM_DRAFT_QUANT unset: min/max ranges) of bf16 rows [n, k] on the device.
    pub fn quantize4Draft(l: *Loader, raw: u64, n: usize, k: usize) !kern.Q4 {
        const m = try l.quantize(raw, n, k, false);
        return m.q4();
    }

    /// qmm.make_f8 of bf16 rows [n, k] on the device.
    pub fn f8(l: *Loader, raw: u64, n: usize, k: usize) !Mat {
        const q = try l.alloc(n * k);
        const sc = try l.alloc(n * (k / 128) * 4);
        const m = try kern.makeF8(l.kernels, l.src.ops.s, raw, n, k, q, sc);
        try l.src.ops.s.synchronize();
        return .{ .kind = .f8, .ptr = m.w, .n = n, .k = k, .s = m.s };
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

    pub fn layerNames(l: *Loader, i: usize, comptime parts: []const []const u8) ![parts.len][]const u8 {
        var out: [parts.len][]const u8 = undefined;
        inline for (parts, 0..) |p, j| out[j] = try l.name(prefix ++ "layers.{d}." ++ p, .{i});
        return out;
    }

    pub fn free(l: *Loader, ns: []const []const u8) void {
        for (ns) |n| l.gpa.free(n);
    }

    fn kda(l: *Loader, i: usize) !Kda {
        const c = &l.w.cfg;
        const ll = c.kda_heads / world;
        const p = try l.layerNames(i, &.{ "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.f_a_proj.weight", "self_attn.g_a_proj.weight", "self_attn.b_proj.weight" });
        defer l.free(&p);
        const proj = try l.mat(&p);
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
            .fb = try l.mat(r[0..1]),
            .gb = try l.mat(r[1..2]),
            .conv = conv.ptr,
            .a_log = try l.upOne(r[2], .f32),
            .dt_bias = try l.upOne(r[3], .f32),
            .norm = try l.upOne(r[4], .bf16),
            .o = try l.mat(r[5..6]),
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
            .proj = try l.mat(p[0..2]),
            .q_norm = try l.upOne(p[2], .bf16),
            .kv_norm = try l.upOne(p[3], .bf16),
            .q_b = try l.mat(p[4..5]),
            .o = try l.mat(p[5..6]),
            .index = .{
                .kw = try l.mat(x[0..2]),
                .qb = try l.mat(x[2..3]),
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
        const gu = try l.mat(p[0..2]);
        return .{ .gu = gu, .down = try l.mat(p[2..3]), .width = gu.n / 2 };
    }

    /// The routed experts of layer `i` (experts.prepare): trellises stacked per projection, suh/svh stacked.
    fn exl3(l: *Loader, i: usize) !Exl3 {
        const c = &l.w.cfg;
        const e_n: usize = c.experts;
        const d: usize = c.hidden;
        const width: usize = c.moe_inter / world;
        var k2: u32 = 0;
        var bases: [3]u64 = undefined;
        var gate_row: ?[]u8 = null; // expert 0's gate suh, while every expert's gate and up suh equal it
        defer if (gate_row) |g| l.gpa.free(g);
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
            bases[j] = block;
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
                if (s == 0 and j < 2) {
                    const row = kk * 2;
                    if (j == 0) gate_row = try l.gpa.dupe(u8, bytes.items[0..row]);
                    if (gate_row) |g| for (0..e_n) |e| {
                        if (!std.mem.eql(u8, g, bytes.items[e * row ..][0..row])) {
                            l.gpa.free(g);
                            gate_row = null;
                            break;
                        }
                    };
                }
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
            .gate_base = bases[0],
            .up_base = bases[1],
            .down_base = bases[2],
            .shared_suh = if (gate_row) |g| blk: {
                const ptr = try l.alloc(g.len);
                try l.src.ops.upload(ptr, g);
                break :blk ptr;
            } else 0,
        };
    }

    fn moe(l: *Loader, i: usize) !Moe {
        const p = try l.layerNames(i, &.{ "mlp.gate.weight", "mlp.gate.e_score_correction_bias" });
        defer l.free(&p);
        return .{ .router = try l.upOne(p[0], .bf16), .bias = try l.upOne(p[1], .f32), .ex = try l.exl3(i), .shared = try l.mlp(i, "mlp.shared_experts.") };
    }

    /// weights.load's `layer`; `plain`: the MTP layer (DSA and MoE, no hyper-connections).
    pub fn layer(l: *Loader, i: usize, plain: bool) !Layer {
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
