//! The DFlash2 draft model's weights for one rank (glm_weights.zig's loadDraft).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Source = @import("glm_source.zig").Source;
const kern = @import("glm_kernels.zig");
const wt = @import("glm_weights.zig");

const world = wt.world;
const Draft = wt.Draft;
const DraftLayer = wt.DraftLayer;
const Weights = wt.Weights;
const Loader = @import("glm_weights_loader.zig").Loader;

// -- the DFlash2 draft model ---------------------------------------------------------------------------------------
fn jint(v: std.json.Value) !usize {
    return switch (v) {
        .integer => |i| @intCast(i),
        else => error.BadDraftConfig,
    };
}

fn jfloat(v: std.json.Value) !f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadDraftConfig,
    };
}

fn jget(o: std.json.Value, key: []const u8) !std.json.Value {
    return (if (o == .object) o.object.get(key) else null) orelse error.BadDraftConfig;
}

/// dflash2.Drafter's weights of the model in `dir` for rank `rank`; notes/glm_weights.md#loaddraft
pub fn loadDraft(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ops: @import("glm_source.zig").Ops, kernels: *const kern.Kernels, w: *Weights, dir: []const u8, rank: u32, books: bool) !void {
    const cfg_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(cfg_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena_state.allocator(), text, .{});
    const dc = try jget(root, "dflash_config");
    if (try jint(try jget(dc, "conv_kernel_size")) != 2) return error.OnlyTwoTapConvolutions;
    const dm = try jint(try jget(root, "hidden_size"));
    const hd = try jint(try jget(root, "head_dim"));
    const heads = try jint(try jget(root, "num_attention_heads"));
    const kv_heads = try jint(try jget(root, "num_key_value_heads"));
    const inter = try jint(try jget(root, "intermediate_size"));
    const n_layers = try jint(try jget(root, "num_hidden_layers"));
    if (heads % kv_heads != 0 or kv_heads % world != 0 or inter % (128 * world) != 0) return error.DraftNotSplittable;
    const q_n = heads / world;
    const kv_n = kv_heads / world;
    const m_n = inter / world;
    var dr: Draft = .{
        .d = dm,
        .hd = hd,
        .heads = q_n,
        .kvh = kv_n,
        .inter = m_n,
        .gs = try jint(try jget(dc, "conv_group_size")),
        .block = try jint(try jget(dc, "block_size")),
        .top_k = try jint(try jget(dc, "selector_top_k")),
        .window = try jint(try jget(root, "sliding_window")) - 1,
        .causal = switch (try jget(root, "is_causal")) {
            .bool => |b| b,
            else => true,
        },
        .eps = @floatCast(try jfloat(try jget(root, "rms_norm_eps"))),
        .theta = try jfloat(try jget(try jget(root, "rope_parameters"), "rope_theta")),
        .mask_id = @intCast(try jint(try jget(dc, "mask_token_id"))),
        .sel_rank = try jint(try jget(dc, "selector_rank")),
        .taps = undefined,
        .fc = undefined,
        .hidden_norm = 0,
        .norm = 0,
        .hproj = 0,
        .inv_freq = 0,
        .layers = try gpa.alloc(DraftLayer, n_layers),
    };
    errdefer gpa.free(dr.layers);
    const ids = (try jget(dc, "target_layer_ids")).array.items;
    if (ids.len != 5) return error.DraftTapsNotFive;
    for (ids, &dr.taps) |v, *t| t.* = @intCast(try jint(v));

    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, null);
    defer ck.close();
    var src = try Source.init(gpa, ops);
    defer src.deinit();
    try src.add(&ck);
    var l: Loader = .{ .gpa = gpa, .ck = &ck, .src = &src, .d = d, .rank = rank, .w = w, .kernels = kernels, .dense = .q4 };
    defer l.host.deinit(gpa);
    const before = w.bytes;
    var all: std.ArrayList(u8) = .empty;
    defer all.deinit(gpa);

    dr.fc = try draftQ4(&l, &all, &.{.{ .name = "fc.weight" }});
    dr.hidden_norm = try draftVec(&l, "hidden_norm.weight");
    dr.norm = try draftVec(&l, "norm.weight");
    dr.hproj = try draftVec(&l, "candidate_selector.hidden_projection.weight");
    {
        // inv_freq = 1 / theta ** (arange(hd / 2) * 2 / hd) in fp32
        var inv: [64]f32 = undefined;
        if (hd / 2 > inv.len) return error.DraftHeadDim;
        for (inv[0 .. hd / 2], 0..) |*x, i| x.* = @floatCast(1.0 / std.math.pow(f64, dr.theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(hd))));
        const ptr = try l.alloc(hd / 2 * 4);
        try ops.upload(ptr, std.mem.sliceAsBytes(inv[0 .. hd / 2]));
        dr.inv_freq = ptr;
    }
    var names = std.heap.ArenaAllocator.init(gpa);
    defer names.deinit();
    for (dr.layers, 0..) |*ly, i| {
        _ = names.reset(.retain_capacity);
        const na = names.allocator();
        const pre = try std.fmt.allocPrint(na, "layers.{d}.", .{i});
        const q = try std.fmt.allocPrint(na, "{s}self_attn.q_proj.weight", .{pre});
        const k = try std.fmt.allocPrint(na, "{s}self_attn.k_proj.weight", .{pre});
        const v = try std.fmt.allocPrint(na, "{s}self_attn.v_proj.weight", .{pre});
        const o = try std.fmt.allocPrint(na, "{s}self_attn.o_proj.weight", .{pre});
        const g = try std.fmt.allocPrint(na, "{s}mlp.gate_proj.weight", .{pre});
        const u = try std.fmt.allocPrint(na, "{s}mlp.up_proj.weight", .{pre});
        const dn = try std.fmt.allocPrint(na, "{s}mlp.down_proj.weight", .{pre});
        const qr = rank * q_n * hd;
        const kr = rank * kv_n * hd;
        const mr = rank * m_n;
        ly.qkv = try draftQ4(&l, &all, &.{ .{ .name = q, .row0 = qr, .rows = q_n * hd }, .{ .name = k, .row0 = kr, .rows = kv_n * hd }, .{ .name = v, .row0 = kr, .rows = kv_n * hd } });
        ly.kv = try draftQ4(&l, &all, &.{ .{ .name = k, .row0 = kr, .rows = kv_n * hd }, .{ .name = v, .row0 = kr, .rows = kv_n * hd } });
        ly.o = try draftQ4(&l, &all, &.{.{ .name = o, .col0 = qr, .cols = q_n * hd }});
        ly.gu = try draftQ4(&l, &all, &.{ .{ .name = g, .row0 = mr, .rows = m_n }, .{ .name = u, .row0 = mr, .rows = m_n } });
        ly.down = try draftQ4(&l, &all, &.{.{ .name = dn, .col0 = mr, .cols = m_n }});
        ly.in_norm = try draftVec(&l, try std.fmt.allocPrint(na, "{s}input_layernorm.weight", .{pre}));
        ly.post_norm = try draftVec(&l, try std.fmt.allocPrint(na, "{s}post_attention_layernorm.weight", .{pre}));
        ly.q_norm = try draftVec(&l, try std.fmt.allocPrint(na, "{s}self_attn.q_norm.weight", .{pre}));
        ly.k_norm = try draftVec(&l, try std.fmt.allocPrint(na, "{s}self_attn.k_norm.weight", .{pre}));
        ly.a_base = try draftVec(&l, try std.fmt.allocPrint(na, "{s}attention_conv.base_kernel", .{pre}));
        ly.m_base = try draftVec(&l, try std.fmt.allocPrint(na, "{s}mlp_conv.base_kernel", .{pre}));
        ly.a_kp = try draftQ4(&l, &all, &.{.{ .name = try std.fmt.allocPrint(na, "{s}attention_conv.kernel_projection.weight", .{pre}) }});
        ly.m_kp = try draftQ4(&l, &all, &.{.{ .name = try std.fmt.allocPrint(na, "{s}mlp_conv.kernel_projection.weight", .{pre}) }});
    }
    if (books) {
        dr.pred = try draftBook(&l, "candidate_selector.predecessor_codebook", dr.sel_rank);
        errdefer gpa.free(dr.pred);
        dr.succ = try draftBook(&l, "candidate_selector.successor_codebook", dr.sel_rank);
    }
    try src.flush();
    try ops.s.synchronize();
    dr.bytes = w.bytes - before;
    w.draft = dr;
}

/// One slice of a draft tensor: rows row0 .. row0 + rows (all when rows is 0) or columns col0 .. col0 + cols.
const Piece = struct { name: []const u8, row0: usize = 0, rows: usize = 0, col0: usize = 0, cols: usize = 0 };

/// (members of Loader, defined here for the draft model)
fn draftBytes(l: *Loader, p: Piece, out: *std.ArrayList(u8)) !struct { rows: usize, cols: usize } {
    const t = try l.ck.get(p.name);
    if (t.dtype != .bf16 or t.rank != 2) {
        std.log.err("{s}: {t} rank {d}, expected a bf16 matrix", .{ p.name, t.dtype, t.rank });
        return error.UnexpectedTensor;
    }
    const n = t.shape[0];
    const k = t.shape[1];
    if (p.cols == 0) {
        const rows = if (p.rows == 0) n else p.rows;
        if (p.row0 + rows > n) return error.DraftSliceOutOfRange;
        const at = out.items.len;
        try out.resize(l.gpa, at + rows * k * 2);
        try l.src.read(out.items[at..], t.bytes[p.row0 * k * 2 ..][0 .. rows * k * 2]);
        return .{ .rows = rows, .cols = k };
    }
    if (p.col0 + p.cols > k) return error.DraftSliceOutOfRange;
    try l.host.resize(l.gpa, t.bytes.len);
    try l.src.read(l.host.items, t.bytes);
    const at = out.items.len;
    try out.resize(l.gpa, at + n * p.cols * 2);
    for (0..n) |r| @memcpy(out.items[at + r * p.cols * 2 ..][0 .. p.cols * 2], l.host.items[(r * k + p.col0) * 2 ..][0 .. p.cols * 2]);
    return .{ .rows = n, .cols = p.cols };
}

/// The pieces' rows stacked, uploaded and quantized to 4 bits (min/max ranges), as dflash2's quantize4(gpu(cat(...))).
fn draftQ4(l: *Loader, all: *std.ArrayList(u8), pieces: []const Piece) !kern.Q4 {
    all.clearRetainingCapacity();
    var rows: usize = 0;
    var cols: usize = 0;
    for (pieces) |p| {
        const sh = try draftBytes(l, p, all);
        if (cols != 0 and sh.cols != cols) return error.StackWidth;
        cols = sh.cols;
        rows += sh.rows;
    }
    var raw = try cuda.DeviceBuffer.alloc(l.d, all.items.len);
    defer raw.free();
    try l.src.ops.upload(raw.ptr, all.items);
    return l.quantize4Draft(raw.ptr, rows, cols);
}

/// A whole tensor (bf16 as stored) on the device.
fn draftVec(l: *Loader, name: []const u8) !u64 {
    const t = try l.ck.get(name);
    if (t.dtype != .bf16) return error.UnexpectedTensor;
    l.host.clearRetainingCapacity();
    try l.host.resize(l.gpa, t.bytes.len);
    try l.src.read(l.host.items, t.bytes);
    const ptr = try l.alloc(t.bytes.len);
    try l.src.ops.upload(ptr, l.host.items);
    return ptr;
}

/// A codebook [vocab, width] as host fp32.
fn draftBook(l: *Loader, name: []const u8, width: usize) ![]f32 {
    const t = try l.ck.get(name);
    if (t.rank != 2 or t.shape[1] != width) return error.UnexpectedTensor;
    const n = t.numel();
    const out = try l.gpa.alloc(f32, n);
    errdefer l.gpa.free(out);
    switch (t.dtype) {
        .f32 => try l.src.read(std.mem.sliceAsBytes(out), t.bytes),
        .bf16 => {
            const raw = try l.gpa.alloc(u16, n);
            defer l.gpa.free(raw);
            try l.src.read(std.mem.sliceAsBytes(raw), t.bytes);
            for (raw, out) |b, *x| x.* = @bitCast(@as(u32, b) << 16);
        },
        else => return error.UnexpectedTensor,
    }
    return out;
}
