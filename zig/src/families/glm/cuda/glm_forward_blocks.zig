//! The forward engine's layer blocks: matmuls, KDA, DSA, MLP, MoE and hyper-connection mixing.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const tri = @import("glm_triton.zig");
const kern = @import("glm_kernels.zig");
const fwd = @import("glm_forward.zig");
const buffers = @import("glm_forward_buffers.zig");

const Engine = fwd.Engine;
const Mat = wt.Mat;
const bf = fwd.bf;
const f4 = fwd.f4;
const undone = fwd.undone;
const envOn = fwd.envOn;
const denseLimit = fwd.denseLimit;
const Buffers = buffers.Buffers;
const KdaScratch = buffers.KdaScratch;
const max_rows = Engine.max_rows;

pub fn gather(e: *Engine, b: *const Buffers, rows: usize) !u64 {
    try e.comm.allGather(e.s, b.part, b.gath, rows * e.w.cfg.hidden);
    return b.gath;
}

/// forward.mm: x rows (stride in elements) times q.T into out (fp32 with `wide`); notes/glm_forward.md#matmul
pub fn mm(e: *Engine, b: *const Buffers, x: u64, x_stride: usize, xs: u64, q: Mat, out: u64, rows: usize, wide: bool) !void {
    switch (q.kind) {
        .b16 => try e.t.matmulB16(x, x_stride, q.ptr, out, b.sk, rows, q.n, q.k, wide),
        .q4 => if (b.prefill) try kern.q4Prefill(e.k, e.s, x, x_stride, q.q4(), out, rows, wide) else {
            // a shared window past 16 rows on the 32-row tile (each weight read once; bits as on the lane's 16)
            const qq = q.q4();
            if (rows > max_rows and envOn("TF_GLM_QMM32")) {
                try kern.qmmShared(e.k, e.s, x, x_stride, xs, qq, out, rows, wide);
            } else {
                const ob: usize = if (wide) f4 else bf;
                var r: usize = 0;
                while (r < rows) : (r += max_rows) {
                    const m = @min(max_rows, rows - r);
                    try kern.qmmDecode(e.k, e.s, x + r * x_stride * bf, x_stride, xs + r * (qq.k / 64) * f4, qq, out + r * qq.n * ob, m, wide);
                }
            }
        },
        .f8 => {
            // the captured _fmm shapes stop at a lane window (16 rows): a shared window past that in 16-row pieces
            const one = !wide and rows <= 32 and !envOn("TF_GLM_F8_PIECES"); // 17 to 32 rows: one 32-row pass
            const step = if (b.prefill or one) rows else max_rows;
            const ob: usize = if (wide) f4 else bf;
            var r: usize = 0;
            while (r < rows) : (r += step) {
                const m = @min(step, rows - r);
                try e.t.matmulF8(x + r * x_stride * bf, x_stride, q.ptr, q.s, out + r * q.n * ob, b.sk, m, q.n, q.k, wide);
            }
        },
    }
}

/// forward.out_proj: every rank's partials [world, R, D] (b.gath), or 0 while a prompt chunk runs row-split.
pub fn outProj(e: *Engine, b: *const Buffers, x: u64, xs: u64, q: Mat, rows: usize, site: ?struct { u32, usize }) !u64 {
    if (e.splitActive(b)) {
        try e.splitPartial(b, .{ .proj = .{ .x = x, .xs = xs, .q = q } });
        return 0;
    }
    try e.mm(b, x, q.k, xs, q, b.part, rows, true);
    if (site) |s| try e.pfSite(s[0], s[1]);
    return e.gather(b, rows);
}

/// forward.kda_front: a prompt chunk's KDA input projections of rows lo .. hi (row-independent).
pub fn kdaFront(e: *Engine, l: *const wt.Layer, b: *const Buffers, lo: usize, hi: usize) !void {
    const c = e.w.cfg;
    const kw = l.kda.?;
    const ll: usize = c.kda_heads / wt.world;
    const n = hi - lo;
    const p = b.kproj + lo * e.width * bf;
    try e.mm(b, b.normed + lo * c.hidden * bf, c.hidden, b.xs + lo * c.hidden / 64 * f4, kw.proj, p, n, false);
    try e.mm(b, p + kw.fa_off * bf, e.width, b.xs_fa + lo * 2 * f4, kw.fb, b.ka + lo * ll * 128 * bf, n, false);
    try e.mm(b, p + kw.ga_off * bf, e.width, b.xs_ga + lo * 2 * f4, kw.gb, b.kg + lo * ll * 128 * bf, n, false);
}

/// `done`: rows whose projections kdaFront already wrote (a prompt chunk, TF_GLM_PREFILL_OVERLAP=2).
pub fn kdaBlock(e: *Engine, l: *const wt.Layer, b: *const Buffers, rows: usize, host_pos: usize, done: ?[2]usize) !u64 {
    const c = e.w.cfg;
    const kw = l.kda.?;
    const li: usize = e.st.kda_index[l.index];
    const ll: usize = c.kda_heads / wt.world;
    const st = &e.st;
    const p = if (b.prefill) b.kproj else st.proj + li * max_rows * e.width * bf;
    if (b.prefill) {
        for (undone(rows, done)) |r| if (r[1] > r[0]) try e.kdaFront(l, b, r[0], r[1]);
    } else {
        try e.mm(b, b.normed, c.hidden, b.xs, kw.proj, p, rows, false);
        const fa = p + kw.fa_off * bf;
        const ga = p + kw.ga_off * bf;
        // the group sums f_b and g_b read when 4-bit; notes/glm_forward.md#matmul
        try e.t.groupSums(fa, e.width, b.xs_fa, rows, 128);
        try e.mm(b, fa, e.width, b.xs_fa, kw.fb, b.ka, rows, false);
        try e.t.groupSums(ga, e.width, b.xs_ga, rows, 128);
        try e.mm(b, ga, e.width, b.xs_ga, kw.gb, b.kg, rows, false);
        try e.pfSite(l.index, 2);
    }
    try e.kdaState(l, b, rows, host_pos);
    const scr_out = if (b.prefill) b.kscr.out else st.scr.out + li * max_rows * ll * 128 * bf;
    if (!b.prefill) try e.t.groupSums(scr_out, ll * 128, b.kxs, rows, ll * 128);
    return e.outProj(b, scr_out, b.kxs, kw.o, rows, .{ l.index, 0 });
}

/// kdaBlock after its projections: the chain over the KDA state (and a prompt chunk's conv move) into scratch out.
pub fn kdaState(e: *Engine, l: *const wt.Layer, b: *const Buffers, rows: usize, host_pos: usize) !void {
    const c = e.w.cfg;
    const kw = l.kda.?;
    const li: usize = e.st.kda_index[l.index];
    const ll: usize = c.kda_heads / wt.world;
    const st = &e.st;
    const p = if (b.prefill) b.kproj else st.proj + li * max_rows * e.width * bf;
    const rec_layer = ll * 128 * 128 * f4;
    const rec_half = st.kda_layers * rec_layer;
    const scr: KdaScratch = if (b.prefill) b.kscr else .{ .out = st.scr.out + li * max_rows * ll * 128 * bf, .k = st.scr.k + li * max_rows * ll * 128 * f4, .v = st.scr.v + li * max_rows * ll * 128 * bf, .g = st.scr.g + li * max_rows * ll * 128 * f4, .b = st.scr.b + li * max_rows * ll * f4 };
    const conv_layer = 3 * 3 * ll * 128 * bf;
    const chain: kern.Chain = .{
        .heads = ll,
        .p = p,
        .p_stride = e.width,
        .b_off = kw.b_off,
        .a = b.ka,
        .a_stride = ll * 128,
        .g = b.kg,
        .g_stride = ll * 128,
        .conv_state = st.conv + li * conv_layer,
        .conv_w = kw.conv,
        .state_in = st.rec + st.cur * rec_half + li * rec_layer,
        .a_log = kw.a_log,
        .dt_bias = kw.dt_bias,
        .norm_w = kw.norm,
        .eps = c.eps,
        .lower = c.lower_bound,
        .rows = rows,
        .out = scr.out,
        .state_out = st.rec + (1 - st.cur) * rec_half + li * rec_layer,
        .k_save = scr.k,
        .v_save = scr.v,
        .g_save = scr.g,
        .b_save = scr.b,
        .q_tmp = b.q_tmp,
        .y_tmp = b.y_tmp,
    };
    // kda.chain: a prompt chunk takes the chunked form under TF_GLM_KDA_CHUNKED=1 (scratch's k/v/g/b are not written)
    if (b.prefill and e.kda_chunked) try kern.chainChunked(e.k, e.s, chain, host_pos) else try kern.chain(e.k, e.s, chain);
    if (b.prefill) {
        // a prompt chunk keeps every row: the layer's conv window moves now (its state half flips in prefillChunk)
        try e.t.convShift(st.conv + li * conv_layer, b.kproj, rows, 1, 3 * ll * 128, 3 * 3 * ll * 128, b.rows * e.width, e.width);
    }
}

/// forward.dsa_front: DSA's projection, query and latent norms and query expansion of rows lo .. hi (row-independent).
pub fn dsaFront(e: *Engine, l: *const wt.Layer, b: *const Buffers, lo: usize, hi: usize) !void {
    const c = e.w.cfg;
    const a = l.dsa.?;
    const hl: usize = c.mla_heads / wt.world;
    const n = hi - lo;
    const qkv = c.q_lora + c.kv_lora;
    const dp = b.dp + lo * qkv * bf;
    const qr = b.qr + lo * c.q_lora * bf;
    const xs_qr = b.xs_qr + lo * c.q_lora / 64 * f4;
    try e.mm(b, b.normed + lo * c.hidden * bf, c.hidden, b.xs + lo * c.hidden / 64 * f4, a.proj, dp, n, false);
    try e.t.rmsnorm(dp, qkv, a.q_norm, qr, c.q_lora, xs_qr, n, c.q_lora, c.eps);
    try e.t.rmsnorm(dp + c.q_lora * bf, qkv, a.kv_norm, b.lat + lo * c.kv_lora * bf, c.kv_lora, b.xs_lat + lo * c.kv_lora / 64 * f4, n, c.kv_lora, c.eps);
    try e.mm(b, qr, c.q_lora, xs_qr, a.q_b, b.q + lo * hl * c.nope * bf, n, false);
}

/// `done`: rows dsaFront already ran on (a prompt chunk, TF_GLM_PREFILL_OVERLAP=2).
pub fn dsaBlock(e: *Engine, l: *const wt.Layer, b: *Buffers, kc: u64, pos_dev: u64, host_pos: usize, index: ?[3]u64, rows: usize, done: ?[2]usize) !u64 {
    const c = e.w.cfg;
    const a = l.dsa.?;
    const hl: usize = c.mla_heads / wt.world;
    for (undone(rows, done)) |r| if (r[1] > r[0]) try e.dsaFront(l, b, r[0], r[1]);
    try e.dsaState(l, b, kc, pos_dev, host_pos, index, rows);
    try e.t.groupSums(b.vn, hl * c.v_dim, b.xs_ao, rows, hl * c.v_dim);
    return e.outProj(b, b.vn, b.xs_ao, a.o, rows, .{ l.index, 0 });
}

/// dsaBlock after its fronts: the latent write, the indexer, absorb, dense or sparse attention and expand into b.vn.
pub fn dsaState(e: *Engine, l: *const wt.Layer, b: *const Buffers, kc: u64, pos_dev: u64, host_pos: usize, index: ?[3]u64, rows: usize) !void {
    const c = e.w.cfg;
    const a = l.dsa.?;
    const hl: usize = c.mla_heads / wt.world;
    const qk: usize = c.nope;
    const lw: usize = c.kv_lora;
    const dense_limit = denseLimit(e.w);
    // _dsa_latent
    try e.t.latentWrite(b.lat, c.kv_lora, kc, pos_dev, rows, lw);
    const all_sparse = host_pos >= dense_limit;
    const sparse_rows = index != null and (all_sparse or host_pos + rows - 1 >= dense_limit);
    if (index) |ix3| {
        const ix = a.index;
        try e.mm(b, b.normed, c.hidden, b.xs, ix.kw, b.ikr, rows, false);
        try e.t.router(b.normed, c.hidden, ix.gate, b.rpart, b.igr, rows, c.hidden, c.i_dim);
        try e.t.indexUpdate(b.ikr, c.i_dim + c.i_heads, b.igr, ix.ln_w, ix.ln_b, ix.ape, ix3[0], ix3[1], ix3[2], pos_dev, rows, e.index_ring);
    }
    if (b.prefill and e.latent_mma) try e.t.absorbQMma(b.q, a.wk, b.qa, rows, hl, qk, lw) else try e.t.absorbQ(b.q, a.wk, b.qa, rows, hl, qk, lw);
    try e.pfSite(l.index, 2);
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(qk), -0.5));
    if (!all_sparse) {
        const s = b.lat_s;
        const lat_nch = tri.chunksFor(@min(e.st.capacity, 2560) + b.rows);
        // a captured window serves every dense position: all chunks, as Python's graphs (pos unknown) run
        const nch = if (e.capturing) lat_nch else @min(tri.chunksFor(host_pos + rows), lat_nch);
        // the dense pass only on the rows before the dense limit (forward.dense_rows); notes/glm_forward.md#dense-rows
        const dense = if (sparse_rows and host_pos < dense_limit) @min(rows, dense_limit - host_pos) else rows;
        const blocks = (dense + s.part_rows - 1) / s.part_rows;
        var at: [Buffers.pos_slots]u64 = undefined;
        try e.blockPositions(b, pos_dev, host_pos, s.part_rows, blocks, &at);
        try e.t.denseAttention(b.qa, kc, at[0..blocks], s, b.ol, dense, rows, hl, lw, nch, scale);
    }
    if (sparse_rows) {
        const ix = a.index;
        try e.mm(b, b.qr, c.q_lora, b.xs_qr, ix.qb, b.qi, rows, false);
        const ix3 = index.?;
        if (b.prefill and rows >= tri.split_rows) { // select_tokens: a prompt chunk's rows in blocks (_select_prompt)
            const bsz = @min(rows, tri.select_rows);
            const blocks = (rows + bsz - 1) / bsz;
            var at: [Buffers.pos_slots]u64 = undefined;
            try e.blockPositions(b, pos_dev, host_pos, bsz, blocks, &at);
            try e.t.selectPrompt(b.qi, b.ikr + c.i_dim * bf, c.i_dim + c.i_heads, ix3[2], at[0..blocks], b.scores, b.pools, b.split, b.tokens, b.counts, rows, host_pos, e.np_max, c.i_heads, c.i_dim);
        } else {
            const np = tri.poolBucket(host_pos, rows, e.np_max);
            try e.t.selectDecode(b.qi, b.ikr + c.i_dim * bf, c.i_dim + c.i_heads, ix3[2], pos_dev, b.scores, b.pools, b.split, b.tokens, b.counts, rows, np, c.i_heads, c.i_dim);
        }
        if (b.prefill and e.sparse_onepass) { // prompt rows: one pass a row, no partials (their own bits)
            try e.t.sparseOnepass(b.qa, kc, b.tokens, b.counts, b.ol, rows, hl, lw, tri.select_width, scale);
        } else try e.t.sparseAttention(b.qa, kc, b.tokens, b.counts, b.ol, b.spo, b.spm, b.spl, rows, hl, lw, tri.select_width, scale);
    }
    if (b.prefill and e.latent_mma) try e.t.expandVMma(b.ol, a.wv, b.vn, rows, hl, c.v_dim, lw) else try e.t.expandV(b.ol, a.wv, b.vn, rows, hl, c.v_dim, lw);
}

/// TORCH-GLUE `pos_dev + r0`: block j's first position into its own 16-byte slot (the host knows pos).
pub fn blockPositions(e: *Engine, b: *const Buffers, pos_dev: u64, host_pos: usize, step: usize, blocks: usize, out: *[Buffers.pos_slots]u64) !void {
    if (blocks > Buffers.pos_slots) return error.TooManyBlocks;
    out[0] = pos_dev;
    for (1..blocks) |j| {
        out[j] = b.pos_at + j * 16;
        try e.setInt(out[j], host_pos + j * step);
    }
}

/// forward.mlp_front: the dense MLP's gate/up and SwiGLU of rows lo .. hi.
pub fn mlpFront(e: *Engine, m: wt.Mlp, b: *const Buffers, lo: usize, hi: usize) !void {
    const c = e.w.cfg;
    const gu = b.gu + lo * 2 * m.width * bf;
    try e.mm(b, b.normed + lo * c.hidden * bf, c.hidden, b.xs + lo * c.hidden / 64 * f4, m.gu, gu, hi - lo, false);
    try e.t.swiglu(gu, b.act + lo * m.width * bf, b.xs_act + lo * m.width / 64 * f4, hi - lo, m.width, c.swiglu_limit);
}

pub fn mlpBlock(e: *Engine, m: wt.Mlp, b: *const Buffers, rows: usize, layer: u32, done: ?[2]usize) !u64 {
    for (undone(rows, done)) |r| if (r[1] > r[0]) try e.mlpFront(m, b, r[0], r[1]);
    return e.outProj(b, b.act, b.xs_act, m.down, rows, .{ layer, 1 });
}

/// `done`: rows whose shared expert its front already ran (a prompt chunk, TF_GLM_PREFILL_OVERLAP=2).
pub fn moeBlock(e: *Engine, m: wt.Moe, b: *const Buffers, rows: usize, layer: u32, done: ?[2]usize) !u64 {
    const c = e.w.cfg;
    const d: usize = c.hidden;
    const slots: usize = c.topk + 1;
    try e.t.router(b.normed, d, m.router, b.rpart, b.mlog, rows, d, c.experts);
    try e.t.select(b.mlog, m.bias, b.pick, b.wts, rows, c.topk, c.experts, c.routed_scale, true);
    const sh = m.shared;
    // TF_GLM_L2PF: the shared expert before the routed experts; notes/glm_forward.md#moe-order
    if (e.l2pf_active) try e.sharedExpert(b, sh, rows);
    if (e.exl3_v18) try kern.routed18(e.k, e.s, b.normed, d, b.pick, m.ex, b.exl3, rows, c.swiglu_limit, b.prefill) else try kern.routed(e.k, e.s, b.normed, d, b.pick, m.ex, b.exl3, rows, c.swiglu_limit);
    if (e.splitActive(b)) { // the shared expert and combine a piece of rows at a time (forward.moe_block's fill)
        try e.splitPartial(b, .{ .moe = .{ .m = m, .done = done } });
        return 0;
    }
    if (!e.l2pf_active) try e.sharedExpert(b, sh, rows);
    // forward.py moe_block: the shared row read from sy where it is, else copied in; notes/glm_forward.md#moe-order
    if (e.exl3_v18 and (b.prefill or e.combine_sy)) {
        try e.t.combineSy(b.exl3.y, b.sy, b.wts, b.part, rows, d, slots);
    } else {
        try e.copyRows(b.sy, d * f4, b.exl3.y + c.topk * d * f4, slots * d * f4, d * f4, rows);
        try e.t.combine(b.exl3.y, b.wts, b.part, rows, d, slots);
    }
    try e.pfSite(layer, 1);
    return e.gather(b, rows);
}

/// forward.shared_front: the MoE's shared expert on b.normed into b.sy.
pub fn sharedExpert(e: *Engine, b: *const Buffers, sh: wt.Mlp, rows: usize) !void {
    return e.sharedRows(b, sh, 0, rows);
}

/// forward.shared_front: the shared expert on rows lo .. hi into b.sy.
pub fn sharedRows(e: *Engine, b: *const Buffers, sh: wt.Mlp, lo: usize, hi: usize) !void {
    const c = e.w.cfg;
    const n = hi - lo;
    const sgu = b.sgu + lo * 2 * sh.width * bf;
    const sact = b.sact + lo * sh.width * bf;
    const sxs = b.sxs + lo * sh.width / 64 * f4;
    try e.mm(b, b.normed + lo * c.hidden * bf, c.hidden, b.xs + lo * c.hidden / 64 * f4, sh.gu, sgu, n, false);
    try e.t.swiglu(sgu, sact, sxs, n, sh.width, c.swiglu_limit);
    try e.mm(b, sact, sh.width, sxs, sh.down, b.sy + lo * c.hidden * f4, n, true);
}

pub fn copyRows(e: *Engine, src: u64, src_ld: usize, dst: u64, dst_ld: usize, bytes: usize, rows: usize) !void {
    var a: cuda.Args = .{};
    a.add(src);
    a.add(@as(u64, src_ld));
    a.add(dst);
    a.add(@as(u64, dst_ld));
    a.add(@as(u64, bytes));
    try cuda.launch.launch(e.k.copy_rows, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, e.s, &a);
}

/// glue.hc_pre: the mixing dots (tensor cores, hc.cu or glue._hc_partial, by TF_GLM_HC_MMA / HC_DEC), then _hc_finish.
pub fn hcPre(e: *Engine, b: *const Buffers, h: wt.Hc, norm: u64, rows: usize) !void {
    return e.hcPreRows(b, h, norm, 0, rows);
}

/// hcPre of rows lo .. lo + n (every glue kernel is row-independent).
pub fn hcPreRows(e: *Engine, b: *const Buffers, h: wt.Hc, norm: u64, lo: usize, n: usize) !void {
    const c = e.w.cfg;
    const d: usize = c.hidden;
    const s: usize = c.hc;
    const x = b.x + lo * s * d * bf;
    const part = b.hcpart + lo * tri.hc_blocks * 32 * f4;
    if (b.prefill and e.hc_mma) try e.t.hcPartialMma(x, h.fn_, part, n, d, s) else if (e.hc_dec) try kern.hcPartial(e.k, e.s, x, h.fn_, part, n) else try e.t.hcPartial(x, h.fn_, part, n, d, s);
    try e.t.hcFinish(x, h.base, h.scale, norm, b.normed + lo * d * bf, b.xs + lo * d / 64 * f4, b.post + lo * s * f4, b.comb + lo * s * s * f4, part, n, d, s, c.eps, c.hc_eps, c.sinkhorn);
}

/// The layer's KDA or DSA block on b.normed (forward._mixer).
pub fn mixer(e: *Engine, l: *const wt.Layer, b: *Buffers, rows: usize, host_pos: usize, done: ?[2]usize) !u64 {
    if (b.prefill and e.pieces.len > 0) return e.mixPieces(l, b, rows, done);
    return switch (l.kind) {
        .kda => try e.kdaBlock(l, b, rows, host_pos, done),
        .mla => blk: {
            const di: usize = e.st.dsa_index[l.index];
            break :blk try e.dsaBlock(l, b, e.st.kc[di], e.st.pos_dev, host_pos, if (e.st.long_context) e.st.index[di] else null, rows, done);
        },
    };
}
