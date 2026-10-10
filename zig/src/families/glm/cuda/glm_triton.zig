//! GLM-5.3-Flash's Triton kernels from the captured cubins (aot/c2): each launch has the Python 0.6.6 wrapper's grid,
//! argument names and types, and constexprs (py: src/tensorfold/families/glm5_next/cuda/). num_warps and num_stages
//! are baked into the cubins; `aot.Set.run` picks the specialization (ints equal to 1, 16-divisibility) itself.

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

const p = aot.ptr;

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn flag(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// triton.next_power_of_2.
fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, n) catch unreachable;
}

const bf16 = 2; // bytes of a bf16 element, for row offsets of sliced tensors

/// glue.HC_BLOCKS (glue.py:12): the fixed K split of the hyper-connection mixing dots.
pub const hc_blocks = 16;
/// glue.ROUTER_KS (glue.py:330).
pub const router_ks = 8;
/// qmm.SPLIT_TARGET (qmm.py:76) and qmm.B16_BN, B16_BK (qmm.py:186).
pub const split_target = 192;
pub const b16_bn = 64;
pub const b16_bk = 64;
/// qmm.F8_BN_DECODE (v1.8 qmm.py): `_fmm`'s columns a program for windows of up to 16 rows.
pub const f8_bn_decode = 32;
/// latent.CHUNK, KT, HB, HB_WIDE (latent.py:12-15).
pub const chunk = 512;
pub const kt = 32;
pub const hb = 16;
pub const hb_wide = 32;
/// sparse.POOL, TOPK_POOLS (sparse.py:11-12) and the token list width select_tokens builds (sparse.py:206): 2051.
pub const pool = 4;
pub const topk_pools = 512;
pub const select_width = topk_pools * pool + pool - 1;
/// geometry.MLA_SELECT_ROWS (cuda/geometry.py:14): a prompt chunk's rows scored at once.
pub const select_rows = 512;
/// geometry.MLA_PROMPT_ATT_ROWS (cuda/geometry.py:13): a prompt chunk's rows a dense latent call takes (LatentScratch.part_rows in prefill).
pub const prompt_att_rows = 512;
/// latent.MMA_TILE (64, 128, 64): the prompt chunks' absorb and expand tiles.
pub const mma_bm = 64;
pub const mma_bn = 128;
pub const mma_bk = 64;

/// qmm.b16_split_k (qmm.py:234): K slices of a BF16 matmul, fixed by the shape.
pub fn b16SplitK(n: usize, k: usize) usize {
    const tiles = cdiv(n, b16_bn);
    const steps = k / b16_bk;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < split_target and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return sk;
}

/// qmm.bucket (qmm.py:98): rows per BF16 matmul program.
pub fn bucket(m: usize) !usize {
    for ([_]usize{ 16, 32, 64, 128 }) |b| if (m <= b) return b;
    return error.TooManyRows;
}

/// glue.router's BM (glue.py:336): unlike `bucket`, m past 128 runs as several 128-row programs.
pub fn routerBm(m: usize) usize {
    return if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 64) 64 else 128;
}

/// latent.head_block (latent.py:18).
pub fn headBlock(rows: usize) usize {
    return if (rows >= 64) hb_wide else hb;
}

/// latent.chunks_for (latent.py:351); forward.chunks_for (forward.py:489) is chunksFor(pos + R) (attention.CHUNK is 512 too).
pub fn chunksFor(length: usize) usize {
    return cdiv(length, chunk);
}

/// sparse.py: decode rows' `_scores_rows` programs a row (ROW_SCORE_G), prompt rows a `_scores` program takes (TF_GLM_SCORE_RB,
/// default 4), and where the decode selection splits a row's scores across programs (SPLIT_FROM scores, under SPLIT_ROWS rows).
pub const row_score_g = 128;
pub const score_rb = 4;
pub const split_from = 8192;
pub const split_rows = 64;
/// sparse.SPLIT_BLK (v1.10 patch 0085): step 4's earlier chunks an iteration. And sparse.SCORE_LOOP (patch 0086): the pool blocks a
/// `_scores` program takes (one row a program; speed only).
pub const split_blk = 16;
pub const score_loop = 32;

/// sparse.split_chunk / split_chunks: scores a `_select_split` chunk program takes, and the chunk programs a row of np scores takes.
fn splitChunk(np: usize) usize {
    return if (np <= 16384) 2048 else 4096;
}

fn splitChunks(np: usize) usize {
    return pow2(cdiv(np, splitChunk(np)));
}

/// sparse.pool_count: pools a prompt chunk scores, the visible ones rounded up to a multiple of 1,024 (at most np_max).
pub fn poolCount(pos: usize, rows: usize, np_max: usize) usize {
    const visible = (pos + rows) / pool + 1;
    return @min(np_max, cdiv(visible, 1024) * 1024);
}

/// select_tokens' scale and wscale (1 / 5.656854249492381 exactly for 32 heads).
fn indexScale(dim: usize) f32 {
    return @floatCast(std.math.pow(f64, @floatFromInt(dim), -0.5));
}

fn indexWscale(heads: usize) f32 {
    return @floatCast(if (heads == 32) 1.0 / 5.656854249492381 else std.math.pow(f64, @floatFromInt(heads), -0.5));
}

/// glue.swiglu's block (glue.py:273): gcd(512, w), at least 64.
pub fn swigluBlock(w: usize) !usize {
    const b = std.math.gcd(512, w);
    if (b < 64) return error.WidthNotMultipleOf64;
    return b;
}

/// sparse.pool_bucket (sparse.py:174): pools scored for rows pos .. pos + R - 1, a power of two of at least 1024, at most np_max.
pub fn poolBucket(pos: usize, rows: usize, np_max: usize) usize {
    const visible = (pos + rows) / pool + 1;
    return @min(np_max, @max(1024, pow2(visible)));
}

pub const Tri = struct {
    set: *const aot.Set,
    s: cuda.Stream,
    /// TF_GLM_KV=fp8 (kv8.py): the latent and pooled-key caches are e4m3 rows with a power-of-two scale (width + 16 bytes).
    fp8: bool = false,

    /// A cache's values / scales as the kernels take them (kv8.parts): the codes and the fp32 view of the same rows, or
    /// the bf16 rows twice.
    fn kvv(t: Tri, name: []const u8, addr: u64) aot.Arg {
        return p(name, if (t.fp8) "*fp8e4nv" else "*bf16", addr);
    }

    fn kvs(t: Tri, name: []const u8, addr: u64) aot.Arg {
        return p(name, if (t.fp8) "*fp32" else "*bf16", addr);
    }

    /// kv8.row_bytes / element size: elements a row of `width` values takes.
    pub fn kvRow(t: Tri, width: usize) usize {
        return if (t.fp8) width + 16 else width;
    }

    fn run(t: Tri, name: []const u8, grid: [3]usize, args: []const aot.Arg, consts: []const aot.Const) !void {
        try t.set.run(t.s, name, .{ u(grid[0]), u(grid[1]), u(grid[2]) }, args, consts);
    }

    /// glue.embed (glue.py:46), the BF16 table: _embed_b16 into `copies` streams of each row.
    pub fn embed(t: Tri, ids: u64, table: u64, out: u64, rows: usize, dims: usize, copies: usize) !void {
        try t.run("_embed_b16", .{ rows, dims / 64, 1 }, &.{ p("IDS", "*i32", ids), p("W", "*bf16", table), p("OUT", "*bf16", out) }, &.{ ci("D", dims), ci("COPIES", copies) });
    }

    /// glue.rmsnorm (glue.py:75). Without `xs` Python passes OUT as XS (bf16) and SUMS=False. Rows of `out` must be contiguous.
    pub fn rmsnorm(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, xs: ?u64, rows: usize, d: usize, eps: f32) !void {
        const sums = if (xs) |a| p("XS", "*fp32", a) else p("XS", "*bf16", out);
        try t.run("_rmsnorm", .{ rows, 1, 1 }, &.{ p("X", "*bf16", x), int("x_stride", x_stride), p("W", "*bf16", w), p("OUT", "*bf16", out), int("o_stride", o_stride), sums, aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", pow2(d)), flag("SUMS", xs != null) });
    }

    /// glue.hc_pre (glue.py:162): x [R, S*D] streams -> out [R, D] normed and xs, post [R, S], comb [R, S, S]; part is [R, 16, 32] fp32.
    /// `scale` is the layer's 3 fp32 mixing scales (a pointer, the kernel's SCALE parameter).
    pub fn hcPartial(t: Tri, x: u64, fn_w: u64, part: u64, rows: usize, d: usize, streams: usize) !void {
        try t.run("_hc_partial", .{ rows, hc_blocks, 1 }, &.{ p("X", "*bf16", x), p("FN", "*bf16", fn_w), p("PART", "*fp32", part) }, &.{ ci("WIDE", streams * d), ci("NB", hc_blocks), ci("SUB", 128) });
    }

    /// glue._hc_partial_mma (v1.8 glue.py:155; TF_GLM_HC_MMA): a prompt chunk's mixing dots on the tensor cores, 64 rows a program.
    pub fn hcPartialMma(t: Tri, x: u64, fn_w: u64, part: u64, rows: usize, d: usize, streams: usize) !void {
        try t.run("_hc_partial_mma", .{ cdiv(rows, 64), hc_blocks, 1 }, &.{ p("X", "*bf16", x), p("FN", "*bf16", fn_w), p("PART", "*fp32", part), int("R", rows) }, &.{ ci("WIDE", streams * d), ci("NB", hc_blocks), ci("BM", 64), ci("BK", 64) });
    }

    /// hc_pre's second half: `part`'s blocks summed in order, the collapsed row and its RMSNorm, post and comb.
    pub fn hcFinish(t: Tri, x: u64, base: u64, scale: u64, norm_w: u64, out: u64, xs: u64, post: u64, comb: u64, part: u64, rows: usize, d: usize, streams: usize, eps: f32, hc_eps: f32, iters: usize) !void {
        try t.run("_hc_finish", .{ rows, 1, 1 }, &.{ p("X", "*bf16", x), p("PART", "*fp32", part), p("BASE", "*fp32", base), p("SCALE", "*fp32", scale), p("NW", "*bf16", norm_w), p("OUT", "*bf16", out), p("XS", "*fp32", xs), p("POST", "*fp32", post), p("COMB", "*fp32", comb), aot.float("eps_norm", eps), aot.float("hc_eps", hc_eps) }, &.{ ci("D", d), ci("S", streams), ci("NB", hc_blocks), ci("ITERS", iters), ci("BLOCK", d) });
    }

    /// glue.hc_post (glue.py:202): gathered [world, R, D] fp32 rank partials (contiguous); x and xout may be the same.
    pub fn hcPost(t: Tri, x: u64, xout: u64, gathered: u64, post: u64, comb: u64, world: usize, rows: usize, d: usize, streams: usize) !void {
        const block = @min(1024, d);
        try t.run("_hc_post", .{ rows, d / block, 1 }, &.{ p("X", "*bf16", x), p("XOUT", "*bf16", xout), p("G", "*fp32", gathered), p("POST", "*fp32", post), p("COMB", "*fp32", comb), int("RS", rows * d) }, &.{ ci("D", d), ci("S", streams), ci("WORLD", world), ci("BLOCK", block) });
    }

    /// glue.hc_post_pair (v1.10 glue.py:265, TF_GLM_HC_SPLIT): hc_post of two ranks' fp32 partials [R, D] held apart, rank 0's
    /// at `g0` and rank 1's `rs` elements from it (may be negative): the same kernel, sum order and bits.
    pub fn hcPostPair(t: Tri, x: u64, xout: u64, g0: u64, post: u64, comb: u64, rs: i64, rows: usize, d: usize, streams: usize) !void {
        const block = @min(1024, d);
        try t.run("_hc_post", .{ rows, d / block, 1 }, &.{ p("X", "*bf16", x), p("XOUT", "*bf16", xout), p("G", "*fp32", g0), p("POST", "*fp32", post), p("COMB", "*fp32", comb), aot.int("RS", @intCast(rs)) }, &.{ ci("D", d), ci("S", streams), ci("WORLD", 2), ci("BLOCK", block) });
    }

    /// glue.residual_add (glue.py:228): the MTP block's plain residual over [world, R, D] partials.
    pub fn residualAdd(t: Tri, x: u64, xout: u64, gathered: u64, world: usize, rows: usize, d: usize) !void {
        const block = @min(1024, d);
        try t.run("_residual_add", .{ rows, d / block, 1 }, &.{ p("X", "*bf16", x), p("XOUT", "*bf16", xout), p("G", "*fp32", gathered), int("RS", rows * d) }, &.{ ci("D", d), ci("WORLD", world), ci("BLOCK", block) });
    }

    /// glue.stream_mean (glue.py:247): out [R, D] = the mean of x's `streams` streams.
    pub fn streamMean(t: Tri, x: u64, out: u64, rows: usize, d: usize, streams: usize) !void {
        const block = @min(1024, d);
        try t.run("_stream_mean", .{ rows, d / block, 1 }, &.{ p("X", "*bf16", x), p("OUT", "*bf16", out) }, &.{ ci("D", d), ci("S", streams), ci("BLOCK", block) });
    }

    /// glue.swiglu (glue.py:271): gu [R, 2W] -> out [R, W] and xs [R, W/64].
    pub fn swiglu(t: Tri, gu: u64, out: u64, xs: u64, rows: usize, w: usize, limit: f32) !void {
        const block = try swigluBlock(w);
        try t.run("_swiglu", .{ rows, w / block, 1 }, &.{ p("GU", "*bf16", gu), p("OUT", "*bf16", out), p("XS", "*fp32", xs), aot.float("LIMIT", limit) }, &.{ ci("W", w), ci("BLOCK", block) });
    }

    /// glue.router (glue.py:333), the sliced path: `part` holds router_ks * m * ne fp32 (Python's torch.empty), `out` [m, ne] fp32 contiguous.
    /// Python's single-pass `_router` (d not a multiple of 512, or a strided out) was never captured, so it is an error here.
    pub fn router(t: Tri, x: u64, x_stride: usize, w: u64, part: u64, out: u64, m: usize, d: usize, ne: usize) !void {
        if (d % (router_ks * 64) != 0) return error.RouterSinglePassNotCaptured;
        const bm = routerBm(m);
        try t.run("_router_part", .{ cdiv(m, bm), cdiv(ne, 32), router_ks }, &.{ p("X", "*bf16", x), p("W", "*bf16", w), p("PART", "*fp32", part), int("M", m), int("x_stride", x_stride) }, &.{ ci("D", d), ci("NE", ne), ci("BM", bm), ci("BLOCK_E", 32), ci("BK", 64), ci("KS", router_ks) });
        const total = m * ne;
        try t.run("_router_sum", .{ cdiv(total, 1024), 1, 1 }, &.{ p("PART", "*fp32", part), p("OUT", "*fp32", out), int("total", total) }, &.{ ci("KS", router_ks), ci("BLOCK", 1024) });
    }

    /// glue.select (glue.py:383): each row's top_k experts and weights, then the shared expert (id `experts`, weight 1).
    pub fn select(t: Tri, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, top_k: usize, experts: usize, scale: f32, norm: bool) !void {
        try t.run("_topk", .{ rows, 1, 1 }, &.{ p("L", "*fp32", logits), p("BIAS", "*fp32", bias), p("PICK", "*i32", pick), p("WTS", "*fp32", wts), aot.float("scale", scale) }, &.{ ci("NE", experts), ci("TOPK", top_k), ci("SLOTS", top_k + 1), ci("BLOCK", pow2(experts + 1)), ci("SLOTP", pow2(top_k + 1)), flag("NORM", norm) });
    }

    /// glue.combine (glue.py:406): y [R, slots, D] fp32 (the EXL3 scratch; only fp32 was captured) -> out [R, D] fp32.
    /// The caller first copies the shared expert's sy [R, D] fp32 into slot top_k of y (forward.py:407,
    /// `b.ey[:R, c.top_k].copy_(b.sy[:R])`, a strided copy: row r's D values to y + ((r * slots + top_k) * D) * 4).
    pub fn combine(t: Tri, y: u64, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
        const block = @min(1024, d);
        try t.run("_combine", .{ rows, d / block, 1 }, &.{ p("Y", "*fp32", y), p("WTS", "*fp32", wts), p("OUT", "*fp32", out) }, &.{ ci("D", d), ci("SLOTS", slots), ci("BLOCK", block) });
    }

    /// qmm._matmul_b16 (qmm.py:245): out [m, n] (bf16, or fp32 with `wide`, Python's f32=True, contiguous) = x [m, k] (rows strided) . w [n, k].T.
    /// When b16SplitK(n, k) > 1, `part` holds sk * m * n fp32 (Python's b.sk); otherwise it is unused (Python passes OUT as PART).
    pub fn matmulB16(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) !void {
        if (k % b16_bk != 0) return error.KNotMultipleOf64;
        const bm = try bucket(@min(m, 128));
        const sk = b16SplitK(n, k);
        const ot = if (wide) "*fp32" else "*bf16";
        const partial = if (sk > 1) p("PART", "*fp32", part) else p("PART", ot, out);
        try t.run("_bmm", .{ cdiv(m, bm), cdiv(n, b16_bn), sk }, &.{ p("X", "*bf16", x), p("W", "*bf16", w), p("OUT", ot, out), partial, int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", bm), ci("BLOCK_N", b16_bn), ci("BK", b16_bk), flag("F32", wide) });
        if (sk == 1) return;
        const total = m * n;
        try t.run("_reduce", .{ cdiv(total, 1024), 1, 1 }, &.{ p("PART", "*fp32", part), p("OUT", ot, out), int("total", total) }, &.{ ci("SK", sk), ci("BLOCK", 1024), flag("F32", wide) });
    }

    /// qmm._matmul_f8 (v1.8 qmm.py:417) for a window of up to 16 rows on one K slice (the head's shape): `_fmm`, BM 16 and
    /// BLOCK_N F8_BN_DECODE columns a program, over e4m3 `w` [n, k] with its fp32 block scales `sc` [n, k / 128].
    /// `part` is unused (Python passes OUT as PART when sk is 1).
    pub fn matmulF8(t: Tri, x: u64, x_stride: usize, w: u64, sc: u64, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) !void {
        _ = part;
        if (k % 128 != 0) return error.KNotMultipleOf128;
        if (m > 32 or (m > 16 and wide) or b16SplitK(n, k) != 1) return error.F8ShapeNotCaptured;
        const ot = if (wide) "*fp32" else "*bf16";
        if (m > 16) // a shared window of 17 to 32 rows: the 32-row specialization, each weight read once
            return t.run("_fmm", .{ 1, cdiv(n, 64), 1 }, &.{ p("X", "*bf16", x), p("W", "*fp8e4nv", w), p("S", "*fp32", sc), p("OUT", ot, out), p("PART", ot, out), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", 1), ci("BM", 32), ci("BLOCK_N", 64), ci("BK", b16_bk), flag("F32", wide), flag("SEQ", false) });
        try t.run("_fmm", .{ 1, cdiv(n, f8_bn_decode), 1 }, &.{ p("X", "*bf16", x), p("W", "*fp8e4nv", w), p("S", "*fp32", sc), p("OUT", ot, out), p("PART", ot, out), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", 1), ci("BM", 16), ci("BLOCK_N", f8_bn_decode), ci("BK", b16_bk), flag("F32", wide), flag("SEQ", false) });
    }

    /// glue.combine_shared (v1.8 glue.py:514): `combine` with the shared expert's row read from sy [R, D] fp32 instead of
    /// y's last slot; the same terms in the same slot order.
    pub fn combineSy(t: Tri, y: u64, sy: u64, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
        const block = @min(1024, d);
        try t.run("_combine_sy", .{ rows, d / block, 1 }, &.{ p("Y", "*fp32", y), p("SY", "*fp32", sy), p("WTS", "*fp32", wts), p("OUT", "*fp32", out) }, &.{ ci("D", d), ci("SLOTS", slots), ci("BLOCK", block) });
    }

    /// qmm.group_sums (qmm.py:117): x [m, k] bf16 (rows strided) -> xs [m, k/64] fp32.
    /// dflash2._dconv: the two-tap grouped dynamic convolution of x [rows, d] over rows r and r - 1 (`branch` 0 the first
    /// mix, 1 the second), plus the residual `res` when given.
    pub fn dconv(t: Tri, x: u64, dyn: u64, base: u64, res: ?u64, out: u64, rows: usize, d: usize, gs: usize, branch: usize) !void {
        const block = 1024;
        try t.run("_dconv_kernel", .{ rows, cdiv(d, block), 1 }, &.{ p("X", "*bf16", x), p("DYN", "*bf16", dyn), p("BASE", "*bf16", base), p("RES", "*bf16", res orelse x), p("OUT", "*bf16", out) }, &.{ ci("D", d), ci("G", d / gs), ci("GS", gs), ci("BRANCH", branch), flag("HAS_RES", res != null), ci("BLOCK", block), ci("SEG", 0) });
    }

    /// dflash2._prep: q and k normalized and rotated, v copied, [heads, rows, head_dim] each (`qo` unused without query heads).
    pub fn draftPrep(t: Tri, qkv: u64, qn: u64, kn: u64, cos: u64, sin: u64, qo: u64, ko: u64, vo: u64, rows: usize, stride: usize, eps: f32, heads: usize, kvh: usize, hd: usize) !void {
        try t.run("_prep_kernel", .{ rows, heads + 2 * kvh, 1 }, &.{ p("QKV", "*bf16", qkv), p("QN", "*bf16", qn), p("KN", "*bf16", kn), p("COS", "*fp32", cos), p("SIN", "*fp32", sin), p("QO", "*bf16", qo), p("KO", "*bf16", ko), p("VO", "*bf16", vo), int("L", rows), int("stride", stride), aot.float("eps", eps) }, &.{ ci("H", heads), ci("HKV", kvh), ci("HALF", hd / 2) });
    }

    /// dflash2's `_dattn_kernel`: each KV head attends its queries to the window's tiles and the block's keys. `pos` is the
    /// context's end on the device.
    pub fn draftAttention(t: Tri, q: u64, kc: u64, vc: u64, out: u64, pos: u64, window: usize, rows: usize, heads: usize, kvh: usize, hd: usize, cap: usize, causal: bool, ring: bool) !void {
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(hd), -0.5));
        try t.run("_dattn_kernel", .{ kvh, 1, 1 }, &.{ p("Q", "*bf16", q), p("K", "*bf16", kc), p("V", "*bf16", vc), p("OUT", "*bf16", out), p("POS", "*i64", pos), int("window", window), aot.float("scale", scale) }, &.{ ci("N", rows), ci("G", heads / kvh), ci("NH", heads), ci("HD", hd), ci("CAP", cap), ci("BK", 64), flag("CAUSAL", causal), flag("RING", ring), flag("SKIP", true) });
    }

    /// dflash2_multi's `_dattn_seg_kernel`: `_dattn_kernel` for blocks of `n` rows side by side (q [heads, rows, hd], out
    /// [rows, heads hd]), block s reading its slot slot[s] of the pooled caches (kc, vc: slot 0; a slot counts rows of cap, the
    /// kv heads `slot_rows` apart) at position pos[s].
    pub fn draftAttentionSeg(t: Tri, q: u64, kc: u64, vc: u64, out: u64, pos: u64, slot: u64, window: usize, rows: usize, n: usize, slot_rows: usize, heads: usize, kvh: usize, hd: usize, cap: usize, causal: bool, ring: bool) !void {
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(hd), -0.5));
        try t.run("_dattn_seg_kernel", .{ kvh, rows / n, 1 }, &.{ p("Q", "*bf16", q), p("K", "*bf16", kc), p("V", "*bf16", vc), p("OUT", "*bf16", out), p("POS", "*i64", pos), p("SLOT", "*i64", slot), int("window", window), aot.float("scale", scale), int("L", rows), int("SLOT_ROWS", slot_rows) }, &.{ ci("N", n), ci("G", heads / kvh), ci("NH", heads), ci("HD", hd), ci("CAP", cap), ci("BK", 64), flag("CAUSAL", causal), flag("RING", ring) });
    }

    pub fn groupSums(t: Tri, x: u64, x_stride: usize, xs: u64, m: usize, k: usize) !void {
        try t.run("_group_sums", .{ m, cdiv(k / 64, 16), 1 }, &.{ p("X", "*bf16", x), p("XS", "*fp32", xs), int("x_stride", x_stride) }, &.{ ci("K", k), ci("GB", 16) });
    }

    /// latent.absorb_q (latent.py:162), the BF16 AbsorbW: q [R, H, D] -> out [R, H, LW] through wk [H, D, LW].
    pub fn absorbQ(t: Tri, q: u64, wk: u64, out: u64, rows: usize, heads: usize, d: usize, lw: usize) !void {
        try t.run("_absorb_q", .{ heads, lw / 32, 1 }, &.{ p("Q", "*bf16", q), p("WK", "*bf16", wk), p("QA", "*bf16", out), int("R", rows) }, &.{ ci("H", heads), ci("D", d), ci("LW", lw), ci("BN", 32) });
    }

    /// latent.expand_v (latent.py:175), the BF16 AbsorbW: o_lat [R, H, LW] -> out [R, H, DV] through wv [H, DV, LW].
    pub fn expandV(t: Tri, ol: u64, wv: u64, out: u64, rows: usize, heads: usize, dv: usize, lw: usize) !void {
        try t.run("_expand_v", .{ heads, dv / 16, 1 }, &.{ p("OL", "*bf16", ol), p("WV", "*bf16", wv), p("OUT", "*bf16", out), int("R", rows) }, &.{ ci("H", heads), ci("DV", dv), ci("LW", lw), ci("BN", 16) });
    }

    /// latent.absorb_q for a prompt chunk's rows (TF_GLM_LATENT_MMA, v1.8 latent.py:334): `_absorb_mma`, 64-row tiles on the
    /// tensor cores (rows contiguous).
    pub fn absorbQMma(t: Tri, q: u64, wk: u64, out: u64, rows: usize, heads: usize, d: usize, lw: usize) !void {
        try t.run("_absorb_mma", .{ heads, cdiv(rows, mma_bm), lw / mma_bn }, &.{ p("Q", "*bf16", q), p("WK", "*bf16", wk), p("QA", "*bf16", out), int("R", rows) }, &.{ ci("H", heads), ci("D", d), ci("LW", lw), ci("BM", mma_bm), ci("BN", mma_bn), ci("BK", mma_bk) });
    }

    /// latent.expand_v for a prompt chunk (latent.py:363): `_expand_mma`.
    pub fn expandVMma(t: Tri, ol: u64, wv: u64, out: u64, rows: usize, heads: usize, dv: usize, lw: usize) !void {
        try t.run("_expand_mma", .{ heads, cdiv(rows, mma_bm), dv / mma_bn }, &.{ p("OL", "*bf16", ol), p("WV", "*bf16", wv), p("OUT", "*bf16", out), int("R", rows) }, &.{ ci("H", heads), ci("DV", dv), ci("LW", lw), ci("BM", mma_bm), ci("BN", mma_bn), ci("BK", mma_bk) });
    }

    /// latent.sparse_onepass (v1.8 latent.py:709): a prompt chunk's rows with counts > 0 attend their selected tokens in one
    /// online softmax, a program a row (all the rank's heads, ONEPASS_TILE (32, 32, 1), launch (8 warps, 1 stage, prefetch)).
    pub fn sparseOnepass(t: Tri, qa: u64, lc: u64, tokens: u64, counts: u64, out: u64, rows: usize, heads: usize, lw: usize, width: usize, scale: f32) !void {
        const hbt = @max(16, @min(32, pow2(heads)));
        try t.run("_sparse_onepass", .{ rows, cdiv(heads, hbt), 1 }, &.{ p("QA", "*bf16", qa), t.kvv("LC", lc), t.kvs("LS", lc), p("TOK", "*i32", tokens), p("CNT", "*i32", counts), p("OUT", "*bf16", out) }, &.{ ci("W", width), ci("H", heads), ci("LW", lw), aot.cf("SCALE", scale), ci("HBT", hbt), ci("KTT", 32), ci("SPLIT", 1), ci("STAGES", 1), flag("PREFETCH", true), ci("RS", t.kvRow(lw)), flag("FP8", t.fp8) });
    }

    /// latent.latent_write (latent.py:198): lat [R, LW] rows (strided) into cache [capacity, LW] at slots pos..pos + R - 1 (pos on the device).
    pub fn latentWrite(t: Tri, lat: u64, lat_stride: usize, cache: u64, pos: u64, rows: usize, lw: usize) !void {
        try t.run("_lat_write", .{ rows, 1, 1 }, &.{ p("LAT", "*bf16", lat), int("lat_stride", lat_stride), t.kvv("LC", cache), t.kvs("LS", cache), p("POS", "*i32", pos) }, &.{ ci("LW", lw), ci("RS", t.kvRow(lw)), flag("FP8", t.fp8) });
    }

    /// latent.LatentScratch's device buffers (latent.py:304): fp32 chunk partials for `part_rows` rows and an i32 zero word.
    pub const Scratch = struct { po: u64, pm: u64, pl: u64, dummy: u64, part_rows: usize };

    /// latent.attention (latent.py:318): one dense call of `rows` rows at `pos`, `hb_` heads a program.
    fn latentAttention(t: Tri, qa: u64, lc: u64, pos: u64, s: Scratch, out: u64, rows: usize, heads: usize, lw: usize, nch: usize, hb_: usize, scale: f32) !void {
        if (rows > s.part_rows) return error.RowsPastScratch;
        try t.run("_dense_chunks", .{ rows, cdiv(heads, hb_), nch }, &.{ p("QA", "*bf16", qa), t.kvv("LC", lc), t.kvs("LS", lc), p("POS", "*i32", pos), p("PO", "*fp32", s.po), p("PM", "*fp32", s.pm), p("PL", "*fp32", s.pl), int("R", rows) }, &.{ ci("H", heads), ci("LW", lw), ci("CH", chunk), aot.cf("SCALE", scale), ci("HBT", hb_), ci("KTT", kt), ci("RS", t.kvRow(lw)), flag("FP8", t.fp8) });
        try t.run("_merge", .{ rows, heads, 1 }, &.{ p("PO", "*fp32", s.po), p("PM", "*fp32", s.pm), p("PL", "*fp32", s.pl), p("OUT", "*bf16", out), p("CNT", "*i32", s.dummy), int("R", rows) }, &.{ ci("H", heads), ci("LW", lw), ci("NCH", nch), flag("SPARSE", false) });
    }

    /// forward.dense_attention (forward.py:331) over latent.attention: qa [R, H, LW] causal through pos + R - 1 in `nch` 512-key
    /// chunks, in blocks of s.part_rows rows, every block with head_block(R) of the whole window.
    /// `pos_at[j]` is the device i32 position of block j's first row: pos_at[0] is pos itself.
    // TORCH-GLUE: forward.py:341 `at = pos_dev + r0` for block j >= 1 (r0 = j * part_rows): a new one-element int32 device
    //   tensor holding pos + r0 (int32 add, no clamp). The Zig caller fills pos_at[1..] with that value before this call;
    //   each address must be 16-byte aligned (the captured _dense_chunks variants have POS divisible by 16).
    // The caller picks nch as _dsa_latent does (forward.py:372): min(chunks_for(pos + R) or s.nch, s.nch).
    pub fn denseAttention(t: Tri, qa: u64, lc: u64, pos_at: []const u64, s: Scratch, out: u64, rows: usize, window: usize, heads: usize, lw: usize, nch: usize, scale: f32) !void {
        const h = headBlock(window); // the window's head block, though only `rows` of its rows (the ones within the dense limit) run
        const step = s.part_rows;
        if (pos_at.len < cdiv(rows, step)) return error.MissingBlockPositions;
        var r0: usize = 0;
        while (r0 < rows) : (r0 += step) {
            const off: u64 = r0 * heads * lw * bf16;
            try t.latentAttention(qa + off, lc, pos_at[r0 / step], s, out + off, @min(step, rows - r0), heads, lw, nch, h, scale);
        }
    }

    /// latent.sparse_attention (latent.py:335): rows with counts > 0 attend their selected tokens [R, W] (ascending, -1 padded)
    /// into out [R, H, LW]; other rows are left alone. po, pm, pl are Python's torch.empty scratch: nch * R * H (* LW) fp32,
    /// nch = cdiv(W, 512).
    pub fn sparseAttention(t: Tri, qa: u64, lc: u64, tokens: u64, counts: u64, out: u64, po: u64, pm: u64, pl: u64, rows: usize, heads: usize, lw: usize, width: usize, scale: f32) !void {
        const nch = chunksFor(width);
        const h = headBlock(rows);
        try t.run("_sparse_chunks", .{ rows, cdiv(heads, h), nch }, &.{ p("QA", "*bf16", qa), t.kvv("LC", lc), t.kvs("LS", lc), p("TOK", "*i32", tokens), p("CNT", "*i32", counts), p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows) }, &.{ ci("W", width), ci("H", heads), ci("LW", lw), ci("CH", chunk), aot.cf("SCALE", scale), ci("HBT", h), ci("KTT", kt), ci("RS", t.kvRow(lw)), flag("FP8", t.fp8) });
        try t.run("_merge", .{ rows, heads, 1 }, &.{ p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), p("OUT", "*bf16", out), p("CNT", "*i32", counts), int("R", rows) }, &.{ ci("H", heads), ci("LW", lw), ci("NCH", nch), flag("SPARSE", true) });
    }

    /// sparse.index_update (v1.8 sparse.py:76): k_raw [R, 128] (rows strided, the first 128 columns of ikr) and gate [R, 128] fp32
    /// into the index rings ik, ig at pos.. (token t at row t % ring; `ring` is at least the context here: no row wraps), then
    /// every pool the window completes into pk (bf16, or kv8 rows under TF_GLM_KV=fp8).
    pub fn indexUpdate(t: Tri, k_raw: u64, k_stride: usize, gate: u64, ln_w: u64, ln_b: u64, ape: u64, ik: u64, ig: u64, pk: u64, pos: u64, rows: usize, ring: usize) !void {
        try t.run("_index_write", .{ rows, 1, 1 }, &.{ p("KR", "*bf16", k_raw), int("k_stride", k_stride), p("GR", "*fp32", gate), p("LNW", "*bf16", ln_w), p("LNB", "*bf16", ln_b), p("IK", "*bf16", ik), p("IG", "*bf16", ig), p("POS", "*i32", pos), int("RING", ring), aot.float("eps", 1e-6) }, &.{ci("D", 128)});
        try t.run("_pool_keys", .{ rows / 4 + 2, 1, 1 }, &.{ p("IK", "*bf16", ik), p("IG", "*bf16", ig), p("APE", "*bf16", ape), t.kvv("PK", pk), t.kvs("PKS", pk), p("POS", "*i32", pos), int("R", rows), int("RING", ring) }, &.{ ci("D", 128), ci("RS", t.kvRow(128)), flag("FP8", t.fp8) });
    }

    /// The split selection's int32 scratch (sparse.split_scratch) for `rows` rows of up to `np` scores: totals, histograms, counts.
    pub fn splitInts(rows: usize, np: usize) usize {
        return rows * (1024 + splitChunks(np) * 257);
    }

    /// sparse.select_tokens for decode rows (v1.8 sparse.py:323; also a prompt chunk of under 64 rows): `_scores_rows` over the
    /// `np` pools of the graph's bucket into `scores` [rows, np], then the K best pools and each row's tokens and count:
    /// `_select_split` (five launches, np >= 8192; `split` is its scratch) writes them itself, else `_select_rows` (visible
    /// pools only) into `pools` and `_tokens`. qi [rows, H * D] contiguous; w points at ikr's gate columns (row stride w_stride).
    pub fn selectDecode(t: Tri, qi: u64, w: u64, w_stride: usize, pk: u64, pos: u64, scores: u64, pools: u64, split: u64, tokens: u64, counts: u64, rows: usize, np: usize, heads: usize, dim: usize) !void {
        if (np < topk_pools) return error.TopPoolsFallbackNotCaptured;
        const g = @min(row_score_g, cdiv(np, 64));
        const wscale = indexWscale(heads);
        try t.run("_scores_rows", .{ rows, g, 1 }, &.{ p("QI", "*bf16", qi), p("W", "*bf16", w), int("w_stride", w_stride), t.kvv("PK", pk), t.kvs("PKS", pk), p("OUT", "*fp32", scores), int("o_stride", np), p("POS", "*i32", pos), p("SPR", "*i32", pos), p("BASE", "*i32", pos), int("R", rows), int("NP", np), aot.float("scale", indexScale(dim)), aot.float("wscale", wscale) }, &.{ ci("H", heads), ci("HP", @max(16, pow2(heads))), ci("D", dim), ci("BP", 64), ci("G", g), flag("SEG", false), ci("RS", t.kvRow(128)), flag("FP8", t.fp8), ci("K", topk_pools) });
        if (rows < split_rows and np >= split_from) {
            try t.selectSplit(scores, np, pos, split, 0, tokens, counts, rows, true);
            return;
        }
        const big = rows < split_rows and np > 1024;
        try t.run("_select_rows", .{ rows, 1, 1 }, &.{ p("S", "*fp32", scores), p("OUT", "*i64", pools), int("NPS", np), p("POS", "*i32", pos) }, &.{ ci("K", topk_pools), ci("BLOCK", if (big) 4096 else 1024), flag("VIS", true) });
        try t.run("_tokens", .{ rows, 1, 1 }, &.{ p("POOLS", "*i64", pools), p("POS", "*i32", pos), p("TOK", "*i32", tokens), p("CNT", "*i32", counts) }, &.{ ci("W", select_width), ci("K", topk_pools), ci("PL", pool), ci("BLOCK", 1024) });
    }

    // -- segmented windows (v1.10 segments.py, latent.seg_*, sparse.seg_*): several lanes' rows back to back, each row's
    // position and arena base in the SegRows tables (glm_segments.zig), so one CUDA graph a row count serves every mix.

    /// latent.seg_latent_write: lat [R, LW] rows into the arena at each row's base + pos.
    pub fn segLatWrite(t: Tri, lat: u64, lat_stride: usize, cache: u64, pos: u64, base: u64, rows: usize, lw: usize) !void {
        try t.run("_seg_lat_write", .{ rows, 1, 1 }, &.{ p("LAT", "*bf16", lat), int("lat_stride", lat_stride), t.kvv("LC", cache), t.kvs("LS", cache), p("POS", "*i32", pos), p("BASE", "*i32", base) }, &.{ ci("LW", lw), ci("RS", t.kvRow(lw)), flag("FP8", t.fp8) });
    }

    /// sparse.seg_index_update: each row's index key and gate at ibase + pos % ring, then every pool a row completes.
    pub fn segIndexUpdate(t: Tri, k_raw: u64, k_stride: usize, gate: u64, ln_w: u64, ln_b: u64, ape: u64, ik: u64, ig: u64, pk: u64, pos: u64, base: u64, ibase: u64, ring: i32, rows: usize) !void {
        try t.run("_seg_index_write", .{ rows, 1, 1 }, &.{ p("KR", "*bf16", k_raw), int("k_stride", k_stride), p("GR", "*fp32", gate), p("LNW", "*bf16", ln_w), p("LNB", "*bf16", ln_b), p("IK", "*bf16", ik), p("IG", "*bf16", ig), p("POS", "*i32", pos), p("IBASE", "*i32", ibase), aot.int("RING", ring), aot.float("eps", 1e-6) }, &.{ci("D", 128)});
        try t.run("_seg_pool_keys", .{ rows, 1, 1 }, &.{ p("IK", "*bf16", ik), p("IG", "*bf16", ig), p("APE", "*bf16", ape), t.kvv("PK", pk), t.kvs("PKS", pk), p("POS", "*i32", pos), p("BASE", "*i32", base), p("IBASE", "*i32", ibase), aot.int("RING", ring) }, &.{ ci("D", 128), ci("RS", t.kvRow(128)), flag("FP8", t.fp8) });
    }

    /// Scores a segmented selection's rows hold (segments.SelectScratch over a pool of up to 4,194,304 tokens: the
    /// split selection's CHS 4096 / CP 256, the shapes v1.10's captured graphs launch).
    pub const seg_pools = 1 << 20;

    /// sparse.seg_select_tokens: every segment's pools up to its device-side count scored by `_seg_scores` (`segs` programs,
    /// one a table segment), then the split selection over each sparse row's visible pools into tokens / counts.
    pub fn segSelect(t: Tri, qi: u64, w: u64, w_stride: usize, pk: u64, scores: u64, split: u64, tokens: u64, counts: u64, pos: u64, sparse: u64, start: u64, nrows: u64, pbase: u64, pools: u64, rows: usize, segs: usize, heads: usize, dim: usize) !void {
        const np = seg_pools;
        const g = @min(256, cdiv(np, 64));
        try t.run("_seg_scores", .{ segs, g, 1 }, &.{ p("QI", "*bf16", qi), p("W", "*bf16", w), int("w_stride", w_stride), t.kvv("PK", pk), t.kvs("PKS", pk), p("OUT", "*fp32", scores), int("o_stride", np), p("POS", "*i32", pos), p("SPR", "*i32", sparse), p("SEG_START", "*i32", start), p("SEG_ROWS", "*i32", nrows), p("SEG_PBASE", "*i32", pbase), p("SEG_POOLS", "*i32", pools), aot.float("scale", indexScale(dim)), aot.float("wscale", indexWscale(heads)) }, &.{ ci("H", heads), ci("HP", @max(16, pow2(heads))), ci("D", dim), ci("BP", 64), ci("G", g), ci("RS", t.kvRow(128)), flag("FP8", t.fp8) });
        const cp = splitChunks(np);
        const tot = split;
        const hist = split + rows * 1024 * 4;
        const gt = split + rows * (1024 + cp * 256) * 4;
        try t.s.d.check(t.s.d.api.cuMemsetD8Async(tot, 0, rows * 1024 * 4, t.s.handle), "cuMemsetD8Async");
        for (0..5) |step| {
            try t.run("_select_split", .{ rows, cp, 1 }, &.{ p("S", "*fp32", scores), int("s_stride", np), int("NPC", np), p("POS", "*i32", pos), p("SPR", "*i32", sparse), p("TOT", "*i32", tot), p("HIST", "*i32", hist), p("GT", "*i32", gt), p("OUT", "*i32", split), p("TOK", "*i32", tokens), p("CNT", "*i32", counts) }, &.{ ci("W", select_width), ci("K", topk_pools), ci("PL", pool), ci("CHS", splitChunk(np)), ci("CP", cp), ci("STEP", step), flag("SEG", true), flag("VIS", false), flag("TOKS", false), ci("BLK", split_blk) });
        }
    }

    /// latent.seg_attention: dense rows over their stream's keys 0 .. pos, sparse rows over their selected tokens, each in its
    /// own arena extent; seg_chunks() = 5 chunk programs a row, `_merge` over them (a row's bits are its lone window's).
    pub fn segAttention(t: Tri, qa: u64, lc: u64, pos: u64, base: u64, sparse: u64, tokens: ?u64, counts: ?u64, s: Scratch, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
        const nch = 5; // latent.seg_chunks: cdiv(max(SPARSE_FROM, TOKENS), CHUNK)
        if (rows > s.part_rows) return error.RowsPastScratch;
        const hb_: usize = if (rows >= 8) hb_wide else hb; // latent.seg_head_block (SEG_WIDE_ROWS 8)
        const r_arg: aot.Arg = int("R", rows);
        try t.run("_seg_chunks", .{ rows, cdiv(heads, hb_), nch }, &.{ p("QA", "*bf16", qa), t.kvv("LC", lc), t.kvs("LS", lc), p("POS", "*i32", pos), p("BASE", "*i32", base), p("SPR", "*i32", sparse), p("TOK", "*i32", tokens orelse s.dummy), p("CNT", "*i32", counts orelse s.dummy), p("PO", "*fp32", s.po), p("PM", "*fp32", s.pm), p("PL", "*fp32", s.pl), r_arg }, &.{ ci("W", if (tokens != null) select_width else 1), ci("H", heads), ci("LW", lw), ci("CH", chunk), aot.cf("SCALE", scale), ci("HBT", hb_), ci("KTT", kt), ci("RS", t.kvRow(lw)), flag("FP8", t.fp8) });
        try t.run("_merge", .{ rows, heads, 1 }, &.{ p("PO", "*fp32", s.po), p("PM", "*fp32", s.pm), p("PL", "*fp32", s.pl), p("OUT", "*bf16", out), p("CNT", "*i32", s.dummy), int("R", rows) }, &.{ ci("H", heads), ci("LW", lw), ci("NCH", nch), flag("SPARSE", false) });
    }

    /// forward.conv_shift_segments: each table segment's conv slot takes rows keep .. keep + 2 of [conv; its projection rows].
    /// Strides in elements: conv slot and layer, projection layer and row.
    pub fn convShiftSeg(t: Tri, conv: u64, proj: u64, seg: u64, nseg: usize, layers: usize, c: usize, conv_slot: usize, conv_layer: usize, proj_layer: usize, proj_row: usize) !void {
        try t.run("_conv_shift_seg", .{ layers, cdiv(c, 1024), nseg }, &.{ p("CONV", "*bf16", conv), p("PROJ", "*bf16", proj), p("SEG", "*i32", seg), int("conv_slot", conv_slot), int("conv_layer", conv_layer), int("proj_layer", proj_layer), int("proj_row", proj_row) }, &.{ ci("C", c), ci("TAPS", 3), ci("COLS", 6), ci("BLOCK", 1024) });
    }

    /// sparse.select_split: the K best pools of `rows` rows of `np` scores across chunk programs, five launches over `split`
    /// (zeroed totals first). `toks`: rows at pos, pos + 1, .. through `_tokens`' formulas into tokens / counts (VIS), else the
    /// pool ids into `pools` [rows, K] int64.
    fn selectSplit(t: Tri, scores: u64, np: usize, pos: u64, split: u64, pools: u64, tokens: u64, counts: u64, rows: usize, toks: bool) !void {
        const cp = splitChunks(np);
        const tot = split;
        const hist = split + rows * 1024 * 4;
        const gt = split + rows * (1024 + cp * 256) * 4;
        try t.s.d.check(t.s.d.api.cuMemsetD8Async(tot, 0, rows * 1024 * 4, t.s.handle), "cuMemsetD8Async");
        for (0..5) |step| {
            try t.run("_select_split", .{ rows, cp, 1 }, &.{ p("S", "*fp32", scores), int("s_stride", np), int("NPC", np), p("POS", "*i32", pos), p("SPR", "*i32", split), p("TOT", "*i32", tot), p("HIST", "*i32", hist), p("GT", "*i32", gt), if (toks) p("OUT", "*i32", split) else p("OUT", "*i64", pools), p("TOK", "*i32", if (toks) tokens else split), p("CNT", "*i32", if (toks) counts else split) }, &.{ ci("W", select_width), ci("K", topk_pools), ci("PL", pool), ci("CHS", splitChunk(np)), ci("CP", cp), ci("STEP", step), flag("SEG", false), flag("VIS", toks), flag("TOKS", toks), ci("BLK", split_blk) });
        }
    }

    /// sparse._select_prompt (v1.8 sparse.py:358) for a prompt chunk of 64 rows or more at host position `host_pos`: its rows in
    /// blocks of select_rows, each scored over `pool_count` pools by `_scores` (score_rows() = 4 rows a program) into `scores`,
    /// then the best pools (`_select_split` for a block of under 64 rows at 8192 pools or more, else `_select_rows`) and
    /// `_tokens`. `pos_at[j]` is the device i32 position of block j's first row (pos_at[0] is pos itself).
    pub fn selectPrompt(t: Tri, qi: u64, w: u64, w_stride: usize, pk: u64, pos_at: []const u64, scores: u64, pools: u64, split: u64, tokens: u64, counts: u64, rows: usize, host_pos: usize, np_max: usize, heads: usize, dim: usize) !void {
        const b = @min(rows, select_rows);
        if (pos_at.len < cdiv(rows, b)) return error.MissingBlockPositions;
        const np = poolCount(host_pos, rows, np_max);
        if (np < topk_pools) return error.TopPoolsFallbackNotCaptured;
        const wscale = indexWscale(heads);
        var a: usize = 0;
        while (a < rows) : (a += b) {
            const n = @min(b, rows - a);
            const at = pos_at[a / b];
            try t.run("_scores", .{ n, cdiv(cdiv(np, 64), score_loop), 1 }, &.{ p("QI", "*bf16", qi + a * heads * dim * bf16), p("W", "*bf16", w + a * w_stride * bf16), int("w_stride", w_stride), t.kvv("PK", pk), t.kvs("PKS", pk), p("OUT", "*fp32", scores), p("POS", "*i32", at), int("R", n), int("NP", np), aot.float("scale", indexScale(dim)), aot.float("wscale", wscale) }, &.{ ci("H", heads), ci("HP", @max(16, pow2(heads))), ci("D", dim), ci("BP", 64), ci("L", score_loop), ci("RS", t.kvRow(128)), flag("FP8", t.fp8) });
            if (n < split_rows and np >= split_from) {
                try t.selectSplit(scores, np, at, split, pools, 0, 0, n, false);
            } else {
                const big = n < split_rows and np > 1024;
                try t.run("_select_rows", .{ n, 1, 1 }, &.{ p("S", "*fp32", scores), p("OUT", "*i64", pools), int("NPS", np), p("POS", "*fp32", scores) }, &.{ ci("K", topk_pools), ci("BLOCK", if (big) 4096 else 1024), flag("VIS", false) });
            }
            try t.run("_tokens", .{ n, 1, 1 }, &.{ p("POOLS", "*i64", pools), p("POS", "*i32", at), p("TOK", "*i32", tokens + a * select_width * 4), p("CNT", "*i32", counts + a * 4) }, &.{ ci("W", select_width), ci("K", topk_pools), ci("PL", pool), ci("BLOCK", 1024) });
        }
    }

    /// forward._shift_conv (forward.py:525): conv [L, 3, C] (in place) takes rows keep .. keep + 2 of [conv; proj rows]
    /// (proj [L, R, >= C]); strides in elements as Python's conv.stride(0), proj.stride(0), proj.stride(1).
    pub fn convShift(t: Tri, conv: u64, proj: u64, keep: usize, layers: usize, c: usize, conv_layer: usize, proj_layer: usize, proj_row: usize) !void {
        try t.run("_conv_shift", .{ layers, cdiv(c, 1024), 1 }, &.{ p("CONV", "*bf16", conv), p("PROJ", "*bf16", proj), int("keep", keep), int("conv_layer", conv_layer), int("proj_layer", proj_layer), int("proj_row", proj_row) }, &.{ ci("C", c), ci("TAPS", 3), ci("BLOCK", 1024) });
    }
};

// Expected values below come from the capture's launches.json (cap/c2): each kernel's constexprs and grids.
test "b16 split-K matches every captured _bmm shape" {
    const cases = [_][3]usize{
        .{ 12576, 4096, 1 }, .{ 4096, 128, 1 },  .{ 4096, 4096, 4 }, .{ 12288, 4096, 1 },
        .{ 4096, 6144, 4 },  .{ 2048, 4096, 8 }, .{ 8192, 1536, 2 }, .{ 160, 4096, 8 },
        .{ 4096, 8192, 4 },  .{ 4096, 1024, 4 }, .{ 77440, 4096, 1 }, .{ 4096, 1536, 4 },
    };
    for (cases) |c| try std.testing.expectEqual(c[2], b16SplitK(c[0], c[1]));
    // grids: (1, 1210, 1) for the 77440-column head, (1, 197, 1), (1, 3, 8)
    try std.testing.expectEqual(@as(usize, 1210), cdiv(77440, b16_bn));
    try std.testing.expectEqual(@as(usize, 197), cdiv(12576, b16_bn));
    try std.testing.expectEqual(@as(usize, 3), cdiv(160, b16_bn));
    // _reduce grid for one row of 4096 columns: (4,)
    try std.testing.expectEqual(@as(usize, 4), cdiv(1 * 4096, 1024));
}

test "row buckets" {
    try std.testing.expectEqual(@as(usize, 16), try bucket(1));
    try std.testing.expectEqual(@as(usize, 32), try bucket(17));
    try std.testing.expectEqual(@as(usize, 64), try bucket(64));
    try std.testing.expectEqual(@as(usize, 128), try bucket(@min(2048, 128)));
    try std.testing.expectError(error.TooManyRows, bucket(129));
    // _bmm BM=128 grids (16, ., .), (3, ., .), (7, ., .): 2048, 257..384 and 769..896 rows
    try std.testing.expectEqual(@as(usize, 16), cdiv(2048, 128));
    // router: BM 128 with m > 128 gives (16, 9, 8) for 2048 rows and 288 experts, (1, 4, 8) for one row and 128 gates
    try std.testing.expectEqual(@as(usize, 128), routerBm(2048));
    try std.testing.expectEqual(@as(usize, 16), cdiv(2048, routerBm(2048)));
    try std.testing.expectEqual(@as(usize, 9), cdiv(288, 32));
    try std.testing.expectEqual(@as(usize, 16), routerBm(1));
    try std.testing.expectEqual(@as(usize, 32), routerBm(32));
    try std.testing.expectEqual(@as(usize, 64), routerBm(33));
}

test "attention heads, chunks and blocks" {
    try std.testing.expectEqual(@as(usize, 16), headBlock(63)); // 63 rows: HBT 16 (_merge grid (63, 32))
    try std.testing.expectEqual(@as(usize, 32), headBlock(64)); // grid (64, 1, 1)
    try std.testing.expectEqual(@as(usize, 2), cdiv(32, headBlock(7))); // grid (7, 2, 1)
    try std.testing.expectEqual(@as(usize, 6), chunksFor(2560 + 1)); // decode grid (1, 2, 6), _merge NCH 6
    try std.testing.expectEqual(@as(usize, 5), chunksFor(select_width)); // _sparse_chunks W 2051, NCH 5
    try std.testing.expectEqual(@as(usize, 2051), select_width);
    // a prompt chunk of 869 rows: blocks of 512 then 357 rows, both with HBT 32 (grids (512, 1, 6), (357, 1, 6))
    try std.testing.expectEqual(@as(usize, 2), cdiv(869, prompt_att_rows));
    try std.testing.expectEqual(@as(usize, 357), 869 - prompt_att_rows);
    try std.testing.expectEqual(@as(usize, 32), headBlock(869));
}

test "index pools and buckets" {
    // _pool_keys grids (2,), (219,), (514,): R // 4 + 2 for 1, 869, 2048 rows
    try std.testing.expectEqual(@as(usize, 2), 1 / 4 + 2);
    try std.testing.expectEqual(@as(usize, 219), 869 / 4 + 2);
    try std.testing.expectEqual(@as(usize, 514), 2048 / 4 + 2);
    // _scores grids (., 16) .. (., 256) and (., 257): buckets 1024 .. 16384 and the 16448-pool capacity
    try std.testing.expectEqual(@as(usize, 1024), poolBucket(2600, 1, 16448));
    try std.testing.expectEqual(@as(usize, 16), cdiv(poolBucket(2600, 1, 16448), 64));
    try std.testing.expectEqual(@as(usize, 2048), poolBucket(4096, 1, 16448));
    try std.testing.expectEqual(@as(usize, 16384), poolBucket(60000, 1, 16448));
    try std.testing.expectEqual(@as(usize, 16448), poolBucket(65540, 1, 16448));
    try std.testing.expectEqual(@as(usize, 257), cdiv(poolBucket(65540, 1, 16448), 64));
}

test "elementwise blocks" {
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(6144)); // _swiglu W 6144: grid (., 12)
    try std.testing.expectEqual(@as(usize, 12), 6144 / try swigluBlock(6144));
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(1024)); // W 1024: grid (., 2)
    try std.testing.expectError(error.WidthNotMultipleOf64, swigluBlock(96));
    try std.testing.expectEqual(@as(usize, 2048), pow2(1536)); // _rmsnorm D 1536 BLOCK 2048
    try std.testing.expectEqual(@as(usize, 512), pow2(288 + 1)); // _topk BLOCK 512
    try std.testing.expectEqual(@as(usize, 16), pow2(8 + 1)); // _topk SLOTP 16
    try std.testing.expectEqual(@as(usize, 12), cdiv(12288, 1024)); // _conv_shift grid (., 12)
    try std.testing.expectEqual(@as(usize, 4), cdiv(4096 / 64, 16)); // _group_sums K 4096: grid (., 4)
    try std.testing.expectEqual(@as(f32, 0.0625), @as(f32, @floatCast(std.math.pow(f64, 256.0, -0.5)))); // SCALE 0.0625
}

test "every launch method type-checks" {
    std.testing.refAllDecls(Tri);
}

test "selection sizes: pool counts, split chunks, scale" {
    const t = std.testing;
    // v1.8 captures (cap/v1): _select_split CHS 2048 with CP 4 at 8,192 scores, CP 8 at 16,384; CHS 4096 with CP 8 at 32,768
    try t.expectEqual(@as(usize, 4), splitChunks(8192));
    try t.expectEqual(@as(usize, 8), splitChunks(16384));
    try t.expectEqual(@as(usize, 4096), splitChunk(32768));
    try t.expectEqual(@as(usize, 8), splitChunks(32768));
    try t.expectEqual(@as(usize, 128), splitChunks(262146)); // 65 chunks of 4,096, a power of two
    // sparse.pool_count: the visible pools (pos + rows) / 4 + 1 rounded up to 1,024, at most the cache's
    try t.expectEqual(@as(usize, 1024), poolCount(0, 2048, 16448));
    try t.expectEqual(@as(usize, 2048), poolCount(2048, 2048, 16448));
    try t.expectEqual(@as(usize, 16448), poolCount(65536, 2048, 16448));
    try t.expectApproxEqAbs(@as(f32, 0.17677669), indexWscale(32), 1e-7);
}
