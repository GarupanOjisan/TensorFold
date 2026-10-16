//! GLM's segmented-window Triton launches (v1.10 segments.py): methods of glm_triton.zig's Tri.
const tri = @import("glm_triton.zig");
const aot = @import("cuda").aot;
const Tri = tri.Tri;
const Scratch = Tri.Scratch;
const p = tri.p;
const int = tri.int;
const ci = tri.ci;
const flag = tri.flag;
const cdiv = tri.cdiv;
const pow2 = tri.pow2;
const splitChunk = tri.splitChunk;
const splitChunks = tri.splitChunks;
const indexScale = tri.indexScale;
const indexWscale = tri.indexWscale;
const chunk = tri.chunk;
const kt = tri.kt;
const hb = tri.hb;
const hb_wide = tri.hb_wide;
const pool = tri.pool;
const topk_pools = tri.topk_pools;
const select_width = tri.select_width;
const split_blk = tri.split_blk;

// Segmented windows: several lanes' rows back to back in one CUDA graph; notes/glm_triton.md#segmented-windows

/// latent.seg_latent_write: lat [R, LW] rows into the arena at each row's base + pos.
pub fn segLatWrite(t: Tri, lat: u64, lat_stride: usize, cache: u64, pos: u64, base: u64, rows: usize, lw: usize) !void {
    try t.run("_seg_lat_write", .{ rows, 1, 1 }, &.{ p("LAT", "*bf16", lat), int("lat_stride", lat_stride), t.kvv("LC", cache), t.kvs("LS", cache), p("POS", "*i32", pos), p("BASE", "*i32", base) }, &.{ ci("LW", lw), ci("RS", t.kvRow(lw)), flag("FP8", t.fp8) });
}

/// sparse.seg_index_update: each row's index key and gate at ibase + pos % ring, then every pool a row completes.
pub fn segIndexUpdate(t: Tri, k_raw: u64, k_stride: usize, gate: u64, ln_w: u64, ln_b: u64, ape: u64, ik: u64, ig: u64, pk: u64, pos: u64, base: u64, ibase: u64, ring: i32, rows: usize) !void {
    try t.run("_seg_index_write", .{ rows, 1, 1 }, &.{ p("KR", "*bf16", k_raw), int("k_stride", k_stride), p("GR", "*fp32", gate), p("LNW", "*bf16", ln_w), p("LNB", "*bf16", ln_b), p("IK", "*bf16", ik), p("IG", "*bf16", ig), p("POS", "*i32", pos), p("IBASE", "*i32", ibase), aot.int("RING", ring), aot.float("eps", 1e-6) }, &.{ci("D", 128)});
    try t.run("_seg_pool_keys", .{ rows, 1, 1 }, &.{ p("IK", "*bf16", ik), p("IG", "*bf16", ig), p("APE", "*bf16", ape), t.kvv("PK", pk), t.kvs("PKS", pk), p("POS", "*i32", pos), p("BASE", "*i32", base), p("IBASE", "*i32", ibase), aot.int("RING", ring) }, &.{ ci("D", 128), ci("RS", t.kvRow(128)), flag("FP8", t.fp8) });
}

/// Scores a segmented selection's rows hold (CHS 4096 / CP 256); notes/glm_triton.md#seg-pools
pub const seg_pools = 1 << 20;

/// sparse.seg_select_tokens: each segment's pools scored, then the split selection; notes/glm_triton.md#seg-select
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

/// latent.seg_attention: dense and sparse rows in their own arena extents; notes/glm_triton.md#seg-attention
pub fn segAttention(t: Tri, qa: u64, lc: u64, pos: u64, base: u64, sparse: u64, tokens: ?u64, counts: ?u64, s: Scratch, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    const nch = 5; // latent.seg_chunks: cdiv(max(SPARSE_FROM, TOKENS), CHUNK)
    if (rows > s.part_rows) return error.RowsPastScratch;
    const hb_: usize = if (rows >= 8) hb_wide else hb; // latent.seg_head_block (SEG_WIDE_ROWS 8)
    const r_arg: aot.Arg = int("R", rows);
    try t.run("_seg_chunks", .{ rows, cdiv(heads, hb_), nch }, &.{ p("QA", "*bf16", qa), t.kvv("LC", lc), t.kvs("LS", lc), p("POS", "*i32", pos), p("BASE", "*i32", base), p("SPR", "*i32", sparse), p("TOK", "*i32", tokens orelse s.dummy), p("CNT", "*i32", counts orelse s.dummy), p("PO", "*fp32", s.po), p("PM", "*fp32", s.pm), p("PL", "*fp32", s.pl), r_arg }, &.{ ci("W", if (tokens != null) select_width else 1), ci("H", heads), ci("LW", lw), ci("CH", chunk), aot.cf("SCALE", scale), ci("HBT", hb_), ci("KTT", kt), ci("RS", t.kvRow(lw)), flag("FP8", t.fp8) });
    try t.run("_merge", .{ rows, heads, 1 }, &.{ p("PO", "*fp32", s.po), p("PM", "*fp32", s.pm), p("PL", "*fp32", s.pl), p("OUT", "*bf16", out), p("CNT", "*i32", s.dummy), int("R", rows) }, &.{ ci("H", heads), ci("LW", lw), ci("NCH", nch), flag("SPARSE", false) });
}

/// forward.conv_shift_segments: each segment's conv slot takes rows keep .. keep + 2; strides in elements.
pub fn convShiftSeg(t: Tri, conv: u64, proj: u64, seg: u64, nseg: usize, layers: usize, c: usize, conv_slot: usize, conv_layer: usize, proj_layer: usize, proj_row: usize) !void {
    try t.run("_conv_shift_seg", .{ layers, cdiv(c, 1024), nseg }, &.{ p("CONV", "*bf16", conv), p("PROJ", "*bf16", proj), p("SEG", "*i32", seg), int("conv_slot", conv_slot), int("conv_layer", conv_layer), int("proj_layer", proj_layer), int("proj_row", proj_row) }, &.{ ci("C", c), ci("TAPS", 3), ci("COLS", 6), ci("BLOCK", 1024) });
}
