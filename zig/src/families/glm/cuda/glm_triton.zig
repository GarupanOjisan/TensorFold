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
    pub fn hcPre(t: Tri, x: u64, fn_w: u64, base: u64, scale: u64, norm_w: u64, out: u64, xs: u64, post: u64, comb: u64, part: u64, rows: usize, d: usize, streams: usize, eps: f32, hc_eps: f32, iters: usize) !void {
        try t.run("_hc_partial", .{ rows, hc_blocks, 1 }, &.{ p("X", "*bf16", x), p("FN", "*bf16", fn_w), p("PART", "*fp32", part) }, &.{ ci("WIDE", streams * d), ci("NB", hc_blocks), ci("SUB", 128) });
        try t.run("_hc_finish", .{ rows, 1, 1 }, &.{ p("X", "*bf16", x), p("PART", "*fp32", part), p("BASE", "*fp32", base), p("SCALE", "*fp32", scale), p("NW", "*bf16", norm_w), p("OUT", "*bf16", out), p("XS", "*fp32", xs), p("POST", "*fp32", post), p("COMB", "*fp32", comb), aot.float("eps_norm", eps), aot.float("hc_eps", hc_eps) }, &.{ ci("D", d), ci("S", streams), ci("NB", hc_blocks), ci("ITERS", iters), ci("BLOCK", d) });
    }

    /// glue.hc_post (glue.py:202): gathered [world, R, D] fp32 rank partials (contiguous); x and xout may be the same.
    pub fn hcPost(t: Tri, x: u64, xout: u64, gathered: u64, post: u64, comb: u64, world: usize, rows: usize, d: usize, streams: usize) !void {
        const block = @min(1024, d);
        try t.run("_hc_post", .{ rows, d / block, 1 }, &.{ p("X", "*bf16", x), p("XOUT", "*bf16", xout), p("G", "*fp32", gathered), p("POST", "*fp32", post), p("COMB", "*fp32", comb), int("RS", rows * d) }, &.{ ci("D", d), ci("S", streams), ci("WORLD", world), ci("BLOCK", block) });
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
        if (m > 16 or b16SplitK(n, k) != 1) return error.F8ShapeNotCaptured;
        const ot = if (wide) "*fp32" else "*bf16";
        try t.run("_fmm", .{ 1, cdiv(n, f8_bn_decode), 1 }, &.{ p("X", "*bf16", x), p("W", "*fp8e4nv", w), p("S", "*fp32", sc), p("OUT", ot, out), p("PART", ot, out), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", 1), ci("BM", 16), ci("BLOCK_N", f8_bn_decode), ci("BK", b16_bk), flag("F32", wide), flag("SEQ", false) });
    }

    /// glue.combine_shared (v1.8 glue.py:514): `combine` with the shared expert's row read from sy [R, D] fp32 instead of
    /// y's last slot; the same terms in the same slot order.
    pub fn combineSy(t: Tri, y: u64, sy: u64, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
        const block = @min(1024, d);
        try t.run("_combine_sy", .{ rows, d / block, 1 }, &.{ p("Y", "*fp32", y), p("SY", "*fp32", sy), p("WTS", "*fp32", wts), p("OUT", "*fp32", out) }, &.{ ci("D", d), ci("SLOTS", slots), ci("BLOCK", block) });
    }

    /// qmm.group_sums (qmm.py:117): x [m, k] bf16 (rows strided) -> xs [m, k/64] fp32.
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

    /// latent.latent_write (latent.py:198): lat [R, LW] rows (strided) into cache [capacity, LW] at slots pos..pos + R - 1 (pos on the device).
    pub fn latentWrite(t: Tri, lat: u64, lat_stride: usize, cache: u64, pos: u64, rows: usize, lw: usize) !void {
        try t.run("_lat_write", .{ rows, 1, 1 }, &.{ p("LAT", "*bf16", lat), int("lat_stride", lat_stride), p("LC", "*bf16", cache), p("POS", "*i32", pos) }, &.{ci("LW", lw)});
    }

    /// latent.LatentScratch's device buffers (latent.py:304): fp32 chunk partials for `part_rows` rows and an i32 zero word.
    pub const Scratch = struct { po: u64, pm: u64, pl: u64, dummy: u64, part_rows: usize };

    /// latent.attention (latent.py:318): one dense call of `rows` rows at `pos`, `hb_` heads a program.
    fn latentAttention(t: Tri, qa: u64, lc: u64, pos: u64, s: Scratch, out: u64, rows: usize, heads: usize, lw: usize, nch: usize, hb_: usize, scale: f32) !void {
        if (rows > s.part_rows) return error.RowsPastScratch;
        try t.run("_dense_chunks", .{ rows, cdiv(heads, hb_), nch }, &.{ p("QA", "*bf16", qa), p("LC", "*bf16", lc), p("POS", "*i32", pos), p("PO", "*fp32", s.po), p("PM", "*fp32", s.pm), p("PL", "*fp32", s.pl), int("R", rows) }, &.{ ci("H", heads), ci("LW", lw), ci("CH", chunk), aot.cf("SCALE", scale), ci("HBT", hb_), ci("KTT", kt) });
        try t.run("_merge", .{ rows, heads, 1 }, &.{ p("PO", "*fp32", s.po), p("PM", "*fp32", s.pm), p("PL", "*fp32", s.pl), p("OUT", "*bf16", out), p("CNT", "*i32", s.dummy), int("R", rows) }, &.{ ci("H", heads), ci("LW", lw), ci("NCH", nch), flag("SPARSE", false) });
    }

    /// forward.dense_attention (forward.py:331) over latent.attention: qa [R, H, LW] causal through pos + R - 1 in `nch` 512-key
    /// chunks, in blocks of s.part_rows rows, every block with head_block(R) of the whole window.
    /// `pos_at[j]` is the device i32 position of block j's first row: pos_at[0] is pos itself.
    // TORCH-GLUE: forward.py:341 `at = pos_dev + r0` for block j >= 1 (r0 = j * part_rows): a new one-element int32 device
    //   tensor holding pos + r0 (int32 add, no clamp). The Zig caller fills pos_at[1..] with that value before this call;
    //   each address must be 16-byte aligned (the captured _dense_chunks variants have POS divisible by 16).
    // The caller picks nch as _dsa_latent does (forward.py:372): min(chunks_for(pos + R) or s.nch, s.nch).
    pub fn denseAttention(t: Tri, qa: u64, lc: u64, pos_at: []const u64, s: Scratch, out: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32) !void {
        const h = headBlock(rows);
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
        try t.run("_sparse_chunks", .{ rows, cdiv(heads, h), nch }, &.{ p("QA", "*bf16", qa), p("LC", "*bf16", lc), p("TOK", "*i32", tokens), p("CNT", "*i32", counts), p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows) }, &.{ ci("W", width), ci("H", heads), ci("LW", lw), ci("CH", chunk), aot.cf("SCALE", scale), ci("HBT", h), ci("KTT", kt) });
        try t.run("_merge", .{ rows, heads, 1 }, &.{ p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), p("OUT", "*bf16", out), p("CNT", "*i32", counts), int("R", rows) }, &.{ ci("H", heads), ci("LW", lw), ci("NCH", nch), flag("SPARSE", true) });
    }

    /// sparse.index_update (sparse.py:63): k_raw [R, 128] (rows strided, the first 128 columns of ikr) and gate [R, 128] fp32
    /// into the index caches ik, ig at pos.., then every pool the window completes into pk.
    pub fn indexUpdate(t: Tri, k_raw: u64, k_stride: usize, gate: u64, ln_w: u64, ln_b: u64, ape: u64, ik: u64, ig: u64, pk: u64, pos: u64, rows: usize) !void {
        try t.run("_index_write", .{ rows, 1, 1 }, &.{ p("KR", "*bf16", k_raw), int("k_stride", k_stride), p("GR", "*fp32", gate), p("LNW", "*bf16", ln_w), p("LNB", "*bf16", ln_b), p("IK", "*bf16", ik), p("IG", "*bf16", ig), p("POS", "*i32", pos), aot.float("eps", 1e-6) }, &.{ci("D", 128)});
        try t.run("_pool_keys", .{ rows / 4 + 2, 1, 1 }, &.{ p("IK", "*bf16", ik), p("IG", "*bf16", ig), p("APE", "*bf16", ape), p("PK", "*bf16", pk), p("POS", "*i32", pos), int("R", rows) }, &.{ci("D", 128)});
    }

    /// sparse.select_tokens (sparse.py:180), its kernels: per block of select_rows rows, _scores over `np` pools
    /// (poolBucket, or the graph's bucket) into `scores` [min(R, 512), np] fp32, then _select_rows (sparse.top_pools,
    /// sparse.py:163) writes the block's 512 best pools ascending into `pools` [R, 512] int64 at the block's rows
    /// (Python allocates each block's [n, 512] and torch.cat's them; writing in place gives the same tensor).
    /// qi [R, H*D] bf16 contiguous; w points at ikr's gate columns (ikr + index_dim * 2) with row stride w_stride elements.
    /// `pos_at[j]` is the device i32 position of block j's first row: pos_at[0] is pos itself.
    // TORCH-GLUE: sparse.py:198 `at = pos_dev + a` for block j >= 1 (a = j * 512): a one-element int32 device tensor pos + a,
    //   16-byte aligned (the captured variants have POS divisible by 16), filled into pos_at[j] before this call.
    // TORCH-GLUE: sparse.py:166 falls back to torch topk/sort (`_top_pools`) when np < 512 or scores are strided; that path
    //   was not captured and returns an error here.
    // TORCH-GLUE: sparse.py:204-215, after the kernels, all R rows at once (W = select_width = 2051, P = 4, K = 512):
    //   q[r]      = int64(pos) + r                                  (pos read from the device, r in 0..R-1)
    //   npool[r]  = (q[r] + 1) // 4                                 (floor division, q >= 0)
    //   tokens    = int32 [R, W]
    //   tokens[r, i*4 + j] = int32(pools[r, i] * 4 + j)             for i in 0..511, j in 0..3 (pools int64 -> int32)
    //   tail[r, j] = npool[r] * 4 + j                               for j in 0..2
    //   tail_ok[r, j] = tail[r, j] <= q[r]
    //   tokens[r, 2048 + j] = tail_ok[r, j] ? int32(tail[r, j]) : -1
    //   counts[r] = int32(npool[r] > 512 ? 2048 + sum_j tail_ok[r, j] : 0)
    //   tokens and counts are what sparseAttention takes (TOK, CNT).
    pub fn selectTokens(t: Tri, qi: u64, w: u64, w_stride: usize, pk: u64, pos_at: []const u64, scores: u64, pools: u64, rows: usize, np: usize, heads: usize, dim: usize) !void {
        if (np < topk_pools) return error.TopPoolsFallbackNotCaptured;
        const b = @min(rows, select_rows);
        if (pos_at.len < cdiv(rows, b)) return error.MissingBlockPositions;
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(dim), -0.5));
        const wscale: f32 = @floatCast(if (heads == 32) 1.0 / 5.656854249492381 else std.math.pow(f64, @floatFromInt(heads), -0.5));
        var a: usize = 0;
        while (a < rows) : (a += b) {
            const n = @min(b, rows - a);
            const at = pos_at[a / b];
            try t.run("_scores", .{ n, cdiv(np, 64), 1 }, &.{ p("QI", "*bf16", qi + a * heads * dim * bf16), p("W", "*bf16", w + a * w_stride * bf16), int("w_stride", w_stride), p("PK", "*bf16", pk), p("OUT", "*fp32", scores), p("POS", "*i32", at), int("R", n), int("NP", np), aot.float("scale", scale), aot.float("wscale", wscale) }, &.{ ci("H", heads), ci("HP", @max(16, pow2(heads))), ci("D", dim), ci("BP", 64), ci("RB", 1) });
            try t.run("_select_rows", .{ n, 1, 1 }, &.{ p("S", "*fp32", scores), p("OUT", "*i64", pools + a * topk_pools * 8), int("NP", np), p("POS", "*i32", at) }, &.{ ci("K", topk_pools), ci("BLOCK", 1024), flag("VISIBLE", true) });
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
