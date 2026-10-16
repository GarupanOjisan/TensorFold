//! The forward engine's shared and segmented verify windows across lanes, and the commits.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const tri = @import("glm_triton.zig");
const kern = @import("glm_kernels.zig");
const segments = @import("glm_segments.zig");
const fwd = @import("glm_forward.zig");
const buffers = @import("glm_forward_buffers.zig");

const Engine = fwd.Engine;
const bf = fwd.bf;
const f4 = fwd.f4;
const denseLimit = fwd.denseLimit;
const Buffers = buffers.Buffers;
const State = buffers.State;
const max_rows = Engine.max_rows;
const shared_rows = Engine.shared_rows;
const max_lanes = Engine.max_lanes;
const Seg = Engine.Seg;
const Keep = Engine.Keep;

const SegSel = struct { scores: u64, tokens: u64, counts: u64, split: u64 };

/// One forward over several lanes' windows back to back into buf.logits; notes/glm_forward.md#shared-windows
pub fn forwardShared(e: *Engine, segs: []const Seg, tokens: []const u32) !void {
    if (e.pool != null and e.seg_enabled) return e.forwardSeg(segs, tokens);
    var rows: usize = 0;
    for (segs) |g| {
        const st = e.laneState(g.lane);
        if (st.pos + g.rows > denseLimit(e.w) and !st.long_context) return error.PastDenseLimit;
        if (st.pos + g.rows > st.capacity) return error.PastCacheCapacity;
        st.row0 = rows;
        rows += g.rows;
    }
    if (rows != tokens.len or rows > max_rows) return error.WindowTooWide; // the lanes' own KDA scratch
    try e.upload(e.buf.ids, std.mem.sliceAsBytes(tokens));
    const b = &e.buf;
    const prefetch = !e.l2pf_active and if (e.l2pf) |p| rows <= p.s.rows else false;
    if (prefetch) e.l2pf_active = true;
    defer e.l2pf_active = false;
    const c = e.w.cfg;
    const d: usize = c.hidden;
    try e.t.embed(b.ids, e.w.embed, b.x, rows, d, c.hc);
    for (e.w.layers) |*l| {
        const h = l.attn_hc.?;
        try e.hcPre(b, h, l.in_norm, rows);
        const g = switch (l.kind) {
            .kda => try e.kdaShared(l, b, segs, rows),
            .mla => try e.dsaShared(l, b, segs, rows),
        };
        try e.t.hcPost(b.x, b.x, g, b.post, b.comb, wt.world, rows, d, c.hc);
        const f = l.ffn_hc.?;
        try e.hcPre(b, f, l.post_norm, rows);
        const g2 = if (l.mlp) |m| try e.mlpBlock(m, b, rows, l.index, null) else try e.moeBlock(l.moe.?, b, rows, l.index, null);
        try e.t.hcPost(b.x, b.x, g2, b.post, b.comb, wt.world, rows, d, c.hc);
        for (e.tap_layers, 0..) |tl, slot| if (tl == l.index and b.taps[slot] != 0) try e.t.streamMean(b.x, b.taps[slot], rows, d, c.hc);
    }
    try e.t.streamMean(b.x, b.hidden, rows, d, c.hc);
    try e.t.rmsnorm(b.hidden, d, e.w.norm, b.fnormed, d, b.fxs, rows, d, c.eps);
    try e.mm(b, b.fnormed, d, b.fxs, e.w.head, b.logits, rows, false);
    if (prefetch) try e.l2pf.?.join(e.s);
}

/// kdaBlock over a shared window: projections on every row, then each lane's chain; notes/glm_forward.md#shared-windows
pub fn kdaShared(e: *Engine, l: *const wt.Layer, b: *const Buffers, segs: []const Seg, rows: usize) !u64 {
    const c = e.w.cfg;
    const kw = l.kda.?;
    const li: usize = e.st.kda_index[l.index];
    const ll: usize = c.kda_heads / wt.world;
    const p = e.sproj;
    try e.mm(b, b.normed, c.hidden, b.xs, kw.proj, p, rows, false);
    const fa = p + kw.fa_off * bf;
    const ga = p + kw.ga_off * bf;
    try e.t.groupSums(fa, e.width, b.xs_fa, rows, 128);
    try e.mm(b, fa, e.width, b.xs_fa, kw.fb, b.ka, rows, false);
    try e.t.groupSums(ga, e.width, b.xs_ga, rows, 128);
    try e.mm(b, ga, e.width, b.xs_ga, kw.gb, b.kg, rows, false);
    try e.pfSite(l.index, 2);
    const rec_layer = ll * 128 * 128 * f4;
    const conv_layer = 3 * 3 * ll * 128 * bf;
    const out_row = ll * 128 * bf;
    for (segs) |g| {
        const st = e.laneState(g.lane);
        const lp = st.proj + li * max_rows * e.width * bf;
        try e.copyDev(lp, p + st.row0 * e.width * bf, g.rows * e.width * bf);
        const rec_half = st.kda_layers * rec_layer;
        try kern.chain(e.k, e.s, .{
            .heads = ll,
            .p = lp,
            .p_stride = e.width,
            .b_off = kw.b_off,
            .a = b.ka + st.row0 * ll * 128 * bf,
            .a_stride = ll * 128,
            .g = b.kg + st.row0 * ll * 128 * bf,
            .g_stride = ll * 128,
            .conv_state = st.conv + li * conv_layer,
            .conv_w = kw.conv,
            .state_in = st.rec + st.cur * rec_half + li * rec_layer,
            .a_log = kw.a_log,
            .dt_bias = kw.dt_bias,
            .norm_w = kw.norm,
            .eps = c.eps,
            .lower = c.lower_bound,
            .rows = g.rows,
            .out = e.sout + st.row0 * out_row,
            .state_out = st.rec + (1 - st.cur) * rec_half + li * rec_layer,
            .k_save = st.scr.k + li * max_rows * ll * 128 * f4,
            .v_save = st.scr.v + li * max_rows * ll * 128 * bf,
            .g_save = st.scr.g + li * max_rows * ll * 128 * f4,
            .b_save = st.scr.b + li * max_rows * ll * f4,
            .q_tmp = b.q_tmp,
            .y_tmp = b.y_tmp,
        });
    }
    try e.t.groupSums(e.sout, ll * 128, b.kxs, rows, ll * 128);
    return e.outProj(b, e.sout, b.kxs, kw.o, rows, .{ l.index, 0 });
}

/// dsaBlock over a shared window: the front on every row, then each lane's own; notes/glm_forward.md#shared-windows
pub fn dsaShared(e: *Engine, l: *const wt.Layer, b: *Buffers, segs: []const Seg, rows: usize) !u64 {
    const c = e.w.cfg;
    const a = l.dsa.?;
    const hl: usize = c.mla_heads / wt.world;
    const qk: usize = c.nope;
    const lw: usize = c.kv_lora;
    const dense_limit = denseLimit(e.w);
    const di: usize = e.st.dsa_index[l.index];
    try e.mm(b, b.normed, c.hidden, b.xs, a.proj, b.dp, rows, false);
    const qkv = c.q_lora + c.kv_lora;
    try e.t.rmsnorm(b.dp, qkv, a.q_norm, b.qr, c.q_lora, b.xs_qr, rows, c.q_lora, c.eps);
    try e.t.rmsnorm(b.dp + c.q_lora * bf, qkv, a.kv_norm, b.lat, c.kv_lora, b.xs_lat, rows, c.kv_lora, c.eps);
    try e.mm(b, b.qr, c.q_lora, b.xs_qr, a.q_b, b.q, rows, false);
    var any_sparse = false;
    for (segs) |g| {
        const st = e.laneState(g.lane);
        if (st.long_context and st.pos + g.rows - 1 >= dense_limit) any_sparse = true;
    }
    const ix = a.index;
    const iw = c.i_dim + c.i_heads;
    if (e.st.long_context) {
        try e.mm(b, b.normed, c.hidden, b.xs, ix.kw, b.ikr, rows, false);
        try e.t.router(b.normed, c.hidden, ix.gate, b.rpart, b.igr, rows, c.hidden, c.i_dim);
    }
    for (segs) |g| {
        const st = e.laneState(g.lane);
        try e.t.latentWrite(b.lat + st.row0 * lw * bf, c.kv_lora, st.kc[di], st.pos_dev, g.rows, lw);
        if (st.long_context) {
            const ix3 = st.index[di];
            try e.t.indexUpdate(b.ikr + st.row0 * iw * bf, iw, b.igr + st.row0 * c.i_dim * f4, ix.ln_w, ix.ln_b, ix.ape, ix3[0], ix3[1], ix3[2], st.pos_dev, g.rows, e.index_ring);
        }
    }
    try e.t.absorbQ(b.q, a.wk, b.qa, rows, hl, qk, lw);
    try e.pfSite(l.index, 2);
    if (any_sparse) try e.mm(b, b.qr, c.q_lora, b.xs_qr, ix.qb, b.qi, rows, false);
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(qk), -0.5));
    const qa_row = hl * lw * bf;
    for (segs) |g| {
        const st = e.laneState(g.lane);
        const host_pos = st.pos;
        const r = g.rows;
        const r0 = st.row0;
        const all_sparse = host_pos >= dense_limit;
        const sparse_rows = st.long_context and (all_sparse or host_pos + r - 1 >= dense_limit);
        if (!all_sparse) {
            const sc = b.lat_s;
            const lat_nch = tri.chunksFor(@min(st.capacity, 2560) + b.rows);
            const nch = lat_nch; // every chunk, as the lane's own captured windows run (dsaBlock under `capturing`)
            const dense = if (sparse_rows and host_pos < dense_limit) @min(r, dense_limit - host_pos) else r;
            const blocks = (dense + sc.part_rows - 1) / sc.part_rows;
            var at: [Buffers.pos_slots]u64 = undefined;
            try e.blockPositions(b, st.pos_dev, host_pos, sc.part_rows, blocks, &at);
            try e.t.denseAttention(b.qa + r0 * qa_row, st.kc[di], at[0..blocks], sc, b.ol + r0 * qa_row, dense, r, hl, lw, nch, scale);
        }
        if (sparse_rows) {
            const np = tri.poolBucket(host_pos, r, e.np_max);
            const tok = b.tokens + r0 * tri.select_width * 4;
            const cnt = b.counts + r0 * 4;
            try e.t.selectDecode(b.qi + r0 * c.i_heads * c.i_dim * bf, b.ikr + r0 * iw * bf + c.i_dim * bf, iw, st.index[di][2], st.pos_dev, b.scores, b.pools, b.split, tok, cnt, r, np, c.i_heads, c.i_dim);
            try e.t.sparseAttention(b.qa + r0 * qa_row, st.kc[di], tok, cnt, b.ol + r0 * qa_row, b.spo, b.spm, b.spl, r, hl, lw, tri.select_width, scale);
        }
    }
    try e.t.expandV(b.ol, a.wv, b.vn, rows, hl, c.v_dim, lw);
    try e.t.groupSums(b.vn, hl * c.v_dim, b.xs_ao, rows, hl * c.v_dim);
    return e.outProj(b, b.vn, b.xs_ao, a.o, rows, .{ l.index, 0 });
}

/// The segmented selection's buffers, allocated at first use; notes/glm_forward.md#segmented-windows
pub fn segSelBuffers(e: *Engine) !void {
    if (e.seg_sel != null) return;
    const np = tri.Tri.seg_pools;
    const bytes = shared_rows * np * f4 + shared_rows * tri.select_width * 4 + shared_rows * 4 + tri.Tri.splitInts(shared_rows, np) * 4 + 4 * 256;
    e.seg_sel = try cuda.DeviceBuffer.alloc(e.d, bytes);
}

pub fn segSel(e: *const Engine) SegSel {
    const base = e.seg_sel.?.ptr;
    const np = tri.Tri.seg_pools;
    const tokens = std.mem.alignForward(u64, base + shared_rows * np * f4, 256);
    const counts = std.mem.alignForward(u64, tokens + shared_rows * tri.select_width * 4, 256);
    const split = std.mem.alignForward(u64, counts + shared_rows * 4, 256);
    return .{ .scores = base, .tokens = tokens, .counts = counts, .split = split };
}

/// forwardShared through the segmented kernels (TF_GLM_SEG); notes/glm_forward.md#segmented-windows
pub fn forwardSeg(e: *Engine, segs: []const Seg, tokens: []const u32) !void {
    const pl = &e.pool.?;
    var spans: [max_lanes]segments.Span = undefined;
    var ks: [max_lanes]kern.SegRow = undefined;
    if (segs.len > max_lanes) return error.TooManySegments;
    var rows: usize = 0;
    for (segs, 0..) |g, i| {
        const st = e.laneState(g.lane);
        if (st.pos + g.rows > denseLimit(e.w) and !st.long_context) return error.PastDenseLimit;
        if (st.pos + g.rows > st.capacity) return error.PastCacheCapacity;
        st.row0 = rows;
        st.seg = true;
        spans[i] = .{ .base = st.slot * pl.extent_rows, .pos = st.pos, .rows = g.rows };
        ks[i] = .{ .row0 = 0, .rows = @intCast(g.rows), .slot = @intCast(st.slot), .parity = @intCast(st.cur), .conv = @intCast(st.slot), .keep = @intCast(g.rows) };
        rows += g.rows;
    }
    if (rows != tokens.len or rows > shared_rows) return error.WindowTooWide;
    const long = e.st.long_context;
    _ = try segments.SegRows.fill(spans[0..segs.len], if (long) denseLimit(e.w) else null, &e.seg_host);
    try kern.segmentTable(ks[0..segs.len], &e.kseg_host);
    try e.upload(pl.segrows, std.mem.sliceAsBytes(&e.seg_host));
    try e.upload(pl.kseg, std.mem.sliceAsBytes(&e.kseg_host));
    try e.upload(e.buf.ids, std.mem.sliceAsBytes(tokens));
    if (long) try e.segSelBuffers();
    if (e.use_graphs) return e.replay(.{ .lane = 0, .head = .seg, .rows = @intCast(rows), .cur = 0, .np = 0 }, segBody, .{ e, rows });
    try e.segBody(rows);
}

/// The segmented window's forward on e.buf: logits of every row (capturable: nothing reads a host position).
pub fn segBody(e: *Engine, rows: usize) !void {
    const b = &e.buf;
    const prefetch = !e.l2pf_active and if (e.l2pf) |pf| rows <= pf.s.rows else false;
    if (prefetch) e.l2pf_active = true;
    defer e.l2pf_active = false;
    const c = e.w.cfg;
    const d: usize = c.hidden;
    try e.t.embed(b.ids, e.w.embed, b.x, rows, d, c.hc);
    for (e.w.layers) |*l| {
        try e.hcPre(b, l.attn_hc.?, l.in_norm, rows);
        const g = switch (l.kind) {
            .kda => try e.kdaSeg(l, b, rows),
            .mla => try e.dsaSeg(l, b, rows),
        };
        try e.t.hcPost(b.x, b.x, g, b.post, b.comb, wt.world, rows, d, c.hc);
        try e.hcPre(b, l.ffn_hc.?, l.post_norm, rows);
        const g2 = if (l.mlp) |m| try e.mlpBlock(m, b, rows, l.index, null) else try e.moeBlock(l.moe.?, b, rows, l.index, null);
        try e.t.hcPost(b.x, b.x, g2, b.post, b.comb, wt.world, rows, d, c.hc);
        for (e.tap_layers, 0..) |tl, slot| if (tl == l.index and b.taps[slot] != 0) try e.t.streamMean(b.x, b.taps[slot], rows, d, c.hc);
    }
    try e.t.streamMean(b.x, b.hidden, rows, d, c.hc);
    try e.t.rmsnorm(b.hidden, d, e.w.norm, b.fnormed, d, b.fxs, rows, d, c.eps);
    try e.mm(b, b.fnormed, d, b.fxs, e.w.head, b.logits, rows, false);
    if (prefetch) try e.l2pf.?.join(e.s);
}

/// forward.kda_segments: the projections on every row, then kda.chain_segments; notes/glm_forward.md#segmented-windows
pub fn kdaSeg(e: *Engine, l: *const wt.Layer, b: *const Buffers, rows: usize) !u64 {
    const c = e.w.cfg;
    const kw = l.kda.?;
    const pl = &e.pool.?;
    const li: usize = e.st.kda_index[l.index];
    const ll: usize = c.kda_heads / wt.world;
    const p = pl.proj + li * shared_rows * e.width * bf;
    try e.mm(b, b.normed, c.hidden, b.xs, kw.proj, p, rows, false);
    const fa = p + kw.fa_off * bf;
    const ga = p + kw.ga_off * bf;
    try e.t.groupSums(fa, e.width, b.xs_fa, rows, 128);
    try e.mm(b, fa, e.width, b.xs_fa, kw.fb, b.ka, rows, false);
    try e.t.groupSums(ga, e.width, b.xs_ga, rows, 128);
    try e.mm(b, ga, e.width, b.xs_ga, kw.gb, b.kg, rows, false);
    try e.pfSite(l.index, 2);
    const rec_layer = ll * 128 * 128 * f4;
    const conv_layer = 3 * 3 * ll * 128 * bf;
    const out = pl.scr.out + li * shared_rows * ll * 128 * bf;
    try kern.chainSegments(e.k, e.s, .{
        .heads = ll,
        .seg = pl.kseg,
        .nseg = max_lanes,
        .p = p,
        .p_stride = e.width,
        .b_off = kw.b_off,
        .a = b.ka,
        .a_stride = ll * 128,
        .g = b.kg,
        .g_stride = ll * 128,
        .conv = pl.conv + li * conv_layer,
        .conv_slot_stride = pl.conv_slot / bf,
        .conv_w = kw.conv,
        .rec = pl.rec + li * rec_layer,
        .slot_stride = pl.rec_slot / f4,
        .parity_stride = pl.rec_slot / (2 * f4),
        .a_log = kw.a_log,
        .dt_bias = kw.dt_bias,
        .norm_w = kw.norm,
        .eps = c.eps,
        .lower = c.lower_bound,
        .rows = rows,
        .out = out,
        .k_save = pl.scr.k + li * shared_rows * ll * 128 * f4,
        .v_save = pl.scr.v + li * shared_rows * ll * 128 * bf,
        .g_save = pl.scr.g + li * shared_rows * ll * 128 * f4,
        .b_save = pl.scr.b + li * shared_rows * ll * f4,
        .q_tmp = b.q_tmp,
        .y_tmp = b.y_tmp,
    });
    try e.t.groupSums(out, ll * 128, b.kxs, rows, ll * 128);
    return e.outProj(b, out, b.kxs, kw.o, rows, .{ l.index, 0 });
}

/// forward.dsa_segments: the DSA block over a segmented window; notes/glm_forward.md#segmented-windows
pub fn dsaSeg(e: *Engine, l: *const wt.Layer, b: *Buffers, rows: usize) !u64 {
    const c = e.w.cfg;
    const a = l.dsa.?;
    const pl = &e.pool.?;
    const hl: usize = c.mla_heads / wt.world;
    const qk: usize = c.nope;
    const lw: usize = c.kv_lora;
    const di: usize = e.st.dsa_index[l.index];
    const S = segments.SegRows;
    const tab = pl.segrows;
    const pos = tab + S.pos_at * 4;
    const base = tab + S.base_at * 4;
    const sparse = tab + S.sparse_at * 4;
    try e.dsaFront(l, b, 0, rows);
    try e.t.segLatWrite(b.lat, c.kv_lora, pl.kc[di], pos, base, rows, lw);
    var toks: ?u64 = null;
    var cnts: ?u64 = null;
    if (e.st.long_context) {
        const ix = a.index;
        const iw = c.i_dim + c.i_heads;
        try e.mm(b, b.normed, c.hidden, b.xs, ix.kw, b.ikr, rows, false);
        try e.t.router(b.normed, c.hidden, ix.gate, b.rpart, b.igr, rows, c.hidden, c.i_dim);
        try e.t.segIndexUpdate(b.ikr, iw, b.igr, ix.ln_w, ix.ln_b, ix.ape, pl.ik[di], pl.ig[di], pl.pk[di], pos, base, tab + S.ibase_at * 4, 1 << 30, rows); // no ring: a 16-divisible RING past every position, the captured specialization
        try e.mm(b, b.qr, c.q_lora, b.xs_qr, ix.qb, b.qi, rows, false);
        const sel = e.segSel();
        try e.t.segSelect(b.qi, b.ikr + c.i_dim * bf, iw, pl.pk[di], sel.scores, sel.split, sel.tokens, sel.counts, pos, sparse, tab + S.start_at * 4, tab + S.rows_at * 4, tab + S.pbase_at * 4, tab + S.pools_at * 4, rows, segments.max_segs, c.i_heads, c.i_dim);
        toks = sel.tokens;
        cnts = sel.counts;
    }
    try e.t.absorbQ(b.q, a.wk, b.qa, rows, hl, qk, lw);
    try e.pfSite(l.index, 2);
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(qk), -0.5));
    try e.t.segAttention(b.qa, pl.kc[di], pos, base, sparse, toks, cnts, b.lat_s, b.ol, rows, hl, lw, scale);
    try e.t.expandV(b.ol, a.wv, b.vn, rows, hl, c.v_dim, lw);
    try e.t.groupSums(b.vn, hl * c.v_dim, b.xs_ao, rows, hl * c.v_dim);
    return e.outProj(b, b.vn, b.xs_ao, a.o, rows, .{ l.index, 0 });
}

/// verify.BatchedVerify.commit for the bound lane's segment; notes/glm_forward.md#segmented-windows
pub fn commitSeg(e: *Engine, rows: usize, keep: usize) !void {
    const pl = &e.pool.?;
    const st = &e.st;
    st.seg = false;
    const n = st.kda_layers;
    if (n > 0) {
        const ll: usize = e.w.cfg.kda_heads / wt.world;
        e.kseg_host = @splat(.{ .row0 = 0, .rows = 0, .slot = 0, .parity = 0, .conv = 0, .keep = 0 });
        e.kseg_host[0] = .{ .row0 = @intCast(st.row0), .rows = @intCast(rows), .slot = @intCast(st.slot), .parity = @intCast(st.cur), .conv = @intCast(st.slot), .keep = @intCast(keep) };
        try e.upload(pl.kseg, std.mem.sliceAsBytes(&e.kseg_host));
        if (keep < rows) try kern.replayLayersSegments(e.k, e.s, pl.kseg, 1, n, ll, pl.rec, pl.rec_slot / f4, pl.rec_slot / (2 * f4), pl.scr.k, pl.scr.v, pl.scr.g, pl.scr.b, shared_rows);
        try e.t.convShiftSeg(pl.conv, pl.proj, pl.kseg, 1, n, 3 * ll * 128, pl.conv_slot / bf, 3 * 3 * ll * 128, shared_rows * e.width, e.width);
        st.cur = 1 - st.cur;
    }
    try e.setPos(st.pos + keep);
}

/// commitSeg for several lanes of the last segmented window at once; notes/glm_forward.md#segmented-windows
pub fn commitSegs(e: *Engine, ks: []const Keep) !void {
    const m = ks.len;
    if (m == 0) return;
    if (m > max_lanes) return error.TooManySegments;
    var segs: [max_lanes]*State = undefined;
    for (ks) |it| {
        if (it.keep < 1 or it.keep > it.rows) return error.BadKeep;
        if (it.lane >= e.n_lanes or !e.laneState(it.lane).seg) return error.NotSegmented;
    }
    for (ks[0..m], 0..) |it, i| segs[i] = e.laneState(it.lane);
    const n = segs[0].kda_layers;
    if (n > 0) {
        const pl = &e.pool.?;
        const ll: usize = e.w.cfg.kda_heads / wt.world;
        e.kseg_host = @splat(.{ .row0 = 0, .rows = 0, .slot = 0, .parity = 0, .conv = 0, .keep = 0 });
        var r: usize = 0; // the replayed segments first: replayLayersSegments takes the first r, convShiftSeg all m
        for ([_]bool{ true, false }) |replayed| for (ks[0..m], segs[0..m]) |it, st| {
            if ((it.keep < it.rows) != replayed) continue;
            e.kseg_host[r] = .{ .row0 = @intCast(st.row0), .rows = @intCast(it.rows), .slot = @intCast(st.slot), .parity = @intCast(st.cur), .conv = @intCast(st.slot), .keep = @intCast(it.keep) };
            r += 1;
        };
        var replays: usize = 0;
        for (ks[0..m]) |it| replays += @intFromBool(it.keep < it.rows);
        try e.upload(pl.kseg, std.mem.sliceAsBytes(&e.kseg_host));
        if (replays > 0) try kern.replayLayersSegments(e.k, e.s, pl.kseg, replays, n, ll, pl.rec, pl.rec_slot / f4, pl.rec_slot / (2 * f4), pl.scr.k, pl.scr.v, pl.scr.g, pl.scr.b, shared_rows);
        try e.t.convShiftSeg(pl.conv, pl.proj, pl.kseg, m, n, 3 * ll * 128, pl.conv_slot / bf, 3 * 3 * ll * 128, shared_rows * e.width, e.width);
    }
    for (ks[0..m], segs[0..m]) |it, st| {
        st.seg = false;
        if (n > 0) st.cur = 1 - st.cur;
        st.pos += it.keep;
        try e.setInt(st.pos_dev, st.pos);
    }
}

/// forward.commit: keep the last decode window's first `keep` rows.
pub fn commit(e: *Engine, rows: usize, keep: usize) !void {
    if (keep < 1 or keep > rows) return error.BadKeep;
    if (e.st.seg) return e.commitSeg(rows, keep);
    const c = e.w.cfg;
    const ll: usize = c.kda_heads / wt.world;
    const n = e.st.kda_layers;
    if (n > 0) {
        const half = n * ll * 128 * 128 * f4;
        const cur = e.st.cur;
        if (keep < rows) try kern.replayLayers(e.k, e.s, n, ll, e.st.rec + cur * half, e.st.scr.k, e.st.scr.v, e.st.scr.g, e.st.scr.b, max_rows, keep, e.st.rec + (1 - cur) * half);
        e.st.cur = 1 - cur;
        try e.t.convShift(e.st.conv, e.st.proj, keep, n, 3 * ll * 128, 3 * 3 * ll * 128, max_rows * e.width, e.width);
    }
    try e.setPos(e.st.pos + keep);
}

/// forward.commit(R, R) after a prompt chunk on pbuf: one flip for every layer; notes/glm_forward.md#kept-states
pub fn commitChunk(e: *Engine, rows: usize) !void {
    if (e.st.kda_layers > 0) e.st.cur = 1 - e.st.cur;
    try e.setPos(e.st.pos + rows);
}
