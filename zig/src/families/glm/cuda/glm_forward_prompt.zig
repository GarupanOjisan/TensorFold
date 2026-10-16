//! The forward engine's prompt chunks (multi-prompt pieces, the row split) and its decode window.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const tri = @import("glm_triton.zig");
const fwd = @import("glm_forward.zig");
const buffers = @import("glm_forward_buffers.zig");

const Engine = fwd.Engine;
const bf = fwd.bf;
const f4 = fwd.f4;
const Comm = fwd.Comm;
const Fill = fwd.Fill;
const Front = fwd.Front;
const undone = fwd.undone;
const Buffers = buffers.Buffers;
const max_lanes = Engine.max_lanes;
const Piece = Engine.Piece;

/// multi_prefill.PieceMixer: the block on every row, its state work piece by piece; notes/glm_forward.md#pieces
pub fn mixPieces(e: *Engine, l: *const wt.Layer, b: *Buffers, rows: usize, done: ?[2]usize) !u64 {
    const c = e.w.cfg;
    const home = e.lane;
    switch (l.kind) {
        .kda => {
            const kw = l.kda.?;
            for (undone(rows, done)) |r| if (r[1] > r[0]) try e.kdaFront(l, b, r[0], r[1]);
            for (e.pieces) |pc| {
                try e.bind(pc.lane);
                const v = e.pieceView(b, pc.lo);
                try e.kdaState(l, &v, pc.rows, e.st.pos);
            }
            try e.bind(home);
            return e.outProj(b, b.kscr.out, b.kxs, kw.o, rows, .{ l.index, 0 });
        },
        .mla => {
            const a = l.dsa.?;
            const hl: usize = c.mla_heads / wt.world;
            for (undone(rows, done)) |r| if (r[1] > r[0]) try e.dsaFront(l, b, r[0], r[1]);
            for (e.pieces) |pc| {
                try e.bind(pc.lane);
                const v = e.pieceView(b, pc.lo);
                const di: usize = e.st.dsa_index[l.index];
                try e.dsaState(l, &v, e.st.kc[di], e.st.pos_dev, e.st.pos, if (e.st.long_context) e.st.index[di] else null, pc.rows);
            }
            try e.bind(home);
            try e.t.groupSums(b.vn, hl * c.v_dim, b.xs_ao, rows, hl * c.v_dim);
            return e.outProj(b, b.vn, b.xs_ao, a.o, rows, .{ l.index, 0 });
        },
    }
}

/// The prompt buffers from row `lo` on, for kdaState and dsaState; notes/glm_forward.md#pieces
pub fn pieceView(e: *const Engine, b: *const Buffers, lo: usize) Buffers {
    const c = e.w.cfg;
    const ll: usize = c.kda_heads / wt.world;
    const hl: usize = c.mla_heads / wt.world;
    const kh = ll * 128;
    var v = b.*;
    v.normed += lo * c.hidden * bf;
    v.xs += lo * c.hidden / 64 * f4;
    v.kproj += lo * e.width * bf;
    v.ka += lo * kh * bf;
    v.kg += lo * kh * bf;
    v.kscr = .{ .out = b.kscr.out + lo * kh * bf, .k = b.kscr.k + lo * kh * f4, .v = b.kscr.v + lo * kh * bf, .g = b.kscr.g + lo * kh * f4, .b = b.kscr.b + lo * ll * f4 };
    if (b.q_tmp != 0) v.q_tmp += lo * kh * f4;
    if (b.y_tmp != 0) v.y_tmp += lo * kh * bf;
    v.qr += lo * c.q_lora * bf;
    v.xs_qr += lo * c.q_lora / 64 * f4;
    v.lat += lo * c.kv_lora * bf;
    v.q += lo * hl * c.nope * bf;
    v.ikr += lo * (c.i_dim + c.i_heads) * bf;
    v.igr += lo * c.i_dim * f4;
    v.qi += lo * c.i_heads * c.i_dim * bf;
    v.qa += lo * hl * c.kv_lora * bf;
    v.ol += lo * hl * c.kv_lora * bf;
    v.vn += lo * hl * c.v_dim * bf;
    v.tokens += lo * tri.select_width * 4;
    v.counts += lo * 4;
    return v;
}

/// multi_prefill.prefill_pieces up to the commits: one prompt chunk over every piece; notes/glm_forward.md#pieces
pub fn prefillPieces(e: *Engine, pieces: []const Piece, ids: []const u32) !void {
    const d: usize = e.w.cfg.hidden;
    const b = &e.pbuf;
    const R = ids.len;
    if (pieces.len == 0 or pieces.len > max_lanes or R > b.rows) return error.BadChunk;
    const home = e.lane;
    for (pieces, 0..) |pc, i| {
        if (pc.rows == 0 or pc.lo % 64 != 0 or pc.lo + pc.rows > R) return error.BadChunk;
        if (i > 0 and pc.lo < pieces[i - 1].lo + pieces[i - 1].rows) return error.BadChunk;
        for (pieces[0..i]) |o| if (o.lane == pc.lane) return error.BadChunk;
        try e.bind(pc.lane);
        try e.checkRoom(pc.rows, e.st.pos);
    }
    try e.bind(home);
    try e.upload(b.ids, std.mem.sliceAsBytes(ids));
    e.pieces = pieces;
    defer e.pieces = &.{};
    try e.compute(b, R);
    for (pieces) |pc| {
        const r = pc.lo + pc.rows - 1;
        // prefillChunk's head: the last row times the head, with the group sums at its chunk's first row
        try e.mm(b, b.fnormed + r * d * bf, d, b.fxs + pc.lo * (d / 64) * f4, e.w.head, b.logits, 1, false);
        const st = e.laneState(pc.lane);
        try e.copyDev(st.last_logits, b.logits, e.w.head.n * bf);
        try e.copyDev(st.last_hidden, b.fnormed + r * d * bf, d * bf);
    }
}

pub fn layerForward(e: *Engine, l: *const wt.Layer, b: *Buffers, rows: usize, host_pos: usize) !void {
    const c = e.w.cfg;
    const d: usize = c.hidden;
    const s: usize = c.hc;
    const h = l.attn_hc.?;
    try e.hcPre(b, h, l.in_norm, rows);
    const g = try e.mixer(l, b, rows, host_pos, null);
    try e.t.hcPost(b.x, b.x, g, b.post, b.comb, wt.world, rows, d, s);
    const f = l.ffn_hc.?;
    try e.hcPre(b, f, l.post_norm, rows);
    const g2 = if (l.mlp) |m| try e.mlpBlock(m, b, rows, l.index, null) else try e.moeBlock(l.moe.?, b, rows, l.index, null);
    try e.t.hcPost(b.x, b.x, g2, b.post, b.comb, wt.world, rows, d, s);
}

pub fn splitActive(e: *const Engine, b: *const Buffers) bool {
    return b.prefill and if (e.split) |sp| sp.active else false;
}

/// Whether a prompt chunk of `rows` rows runs split (hcsplit.applies): long enough, its pad row fits, v1.8's experts.
pub fn splitApplies(e: *const Engine, b: *const Buffers, rows: usize) bool {
    const sp = e.split orelse return false;
    return b.prefill and e.exl3_v18 and rows >= sp.set.min_rows and (rows + 1) / 2 * 2 <= b.rows;
}

pub fn splitBegin(e: *Engine, b: *const Buffers, rows: usize) !void {
    const c = e.w.cfg;
    const sp = &e.split.?;
    sp.r = rows;
    sp.rp = (rows + 1) / 2 * 2;
    sp.h = sp.rp / 2;
    const np = sp.set.pieceCount();
    var size = (sp.h + np - 1) / np;
    if (size > 64) size = @min(sp.h, (size + 63) / 64 * 64); // whole 64-row tiles for the glue kernels where possible
    sp.n = 0;
    var off: usize = 0;
    while (off < sp.h) : (off += size) {
        sp.pieces[sp.n] = .{ off, @min(size, sp.h - off) };
        sp.n += 1;
    }
    if (sp.rp > rows) { // the pad row: zero streams and partial, never read
        try e.zero(b.x + rows * c.hc * c.hidden * bf, c.hc * c.hidden * bf);
        try e.zero(b.part + rows * c.hidden * f4, c.hidden * f4);
    }
    if (sp.side) |side| {
        try sp.ev_begin.record(e.s);
        try side.wait(sp.ev_begin);
    }
    sp.active = true;
}

pub fn splitFinish(e: *Engine) !void {
    const sp = &e.split.?;
    if (sp.side) |side| if (sp.active) {
        try sp.ev_done.record(side);
        try e.s.wait(sp.ev_done);
    };
    sp.active = false;
}

pub fn mine(e: *const Engine) usize {
    return e.w.rank * e.split.?.h;
}

pub fn theirs(e: *const Engine) usize {
    return (1 - e.w.rank) * e.split.?.h;
}

/// `f`'s rows lo .. hi of b.part, clipped to the chunk (hcsplit._fill).
pub fn fillRows(e: *Engine, b: *const Buffers, f: Fill, lo: usize, hi_in: usize) !void {
    const hi = @min(hi_in, e.split.?.r);
    if (hi <= lo) return;
    const c = e.w.cfg;
    const d: usize = c.hidden;
    switch (f) {
        .proj => |p| try e.mm(b, p.x + lo * p.q.k * bf, p.q.k, p.xs + lo * p.q.k / 64 * f4, p.q, b.part + lo * d * f4, hi - lo, true),
        .moe => |m| {
            for (undone(hi, m.done)) |r| {
                const a = @max(r[0], lo);
                if (a < r[1]) try e.sharedRows(b, m.m.shared, a, r[1]);
            }
            const slots: usize = c.topk + 1;
            try e.t.combineSy(b.exl3.y + lo * slots * d * f4, b.sy + lo * d * f4, b.wts + lo * slots * f4, b.part + lo * d * f4, hi - lo, d, slots);
        },
    }
}

/// Piece j: this rank's partial of the peer's rows out, the peer's partial of this rank's rows into b.gath.
pub fn swapPartial(e: *Engine, s: cuda.Stream, b: *const Buffers, j: usize) !void {
    const d: usize = e.w.cfg.hidden;
    const off, const n = e.split.?.pieces[j];
    try e.comm.exchange(s, &.{.{ .ptr = b.part + (e.theirs() + off) * d * f4, .bytes = n * d * f4 }}, &.{.{ .ptr = b.gath + off * d * f4, .bytes = n * d * f4 }}, 1 - e.w.rank);
}

/// Piece j of bf16 [rows, D] buffers: this rank's rows out, the peer's in.
pub fn swapRows(e: *Engine, s: cuda.Stream, j: usize, outs: []const u64) !void {
    const d: usize = e.w.cfg.hidden;
    const off, const n = e.split.?.pieces[j];
    var sends: [8]Comm.Seg = undefined;
    var recvs: [8]Comm.Seg = undefined;
    for (outs, 0..) |t, i| {
        sends[i] = .{ .ptr = t + (e.mine() + off) * d * bf, .bytes = n * d * bf };
        recvs[i] = .{ .ptr = t + (e.theirs() + off) * d * bf, .bytes = n * d * bf };
    }
    try e.comm.exchange(s, sends[0..outs.len], recvs[0..outs.len], 1 - e.w.rank);
}

/// hcsplit.partial: the peer's rows piece by piece, each piece's swap starting as it is done, then this rank's rows.
pub fn splitPartial(e: *Engine, b: *const Buffers, f: Fill) !void {
    const sp = &e.split.?;
    const side = sp.side orelse {
        try e.fillRows(b, f, 0, sp.r);
        for (0..sp.n) |j| try e.swapPartial(e.s, b, j);
        return;
    };
    for (sp.pieces[0..sp.n], 0..) |pc, j| {
        try e.fillRows(b, f, e.theirs() + pc[0], e.theirs() + pc[0] + pc[1]);
        try sp.ev_fill[j].record(e.s);
        try side.wait(sp.ev_fill[j]);
        try e.swapPartial(side, b, j);
        try sp.ev_part[j].record(side);
    }
    try e.fillRows(b, f, e.mine(), e.mine() + sp.h);
}

pub fn runFront(e: *Engine, b: *const Buffers, f: Front, lo: usize, hi_in: usize) !void {
    const hi = @min(hi_in, e.split.?.r);
    if (hi <= lo) return;
    switch (f) {
        .none => {},
        .kda => |l| try e.kdaFront(l, b, lo, hi),
        .dsa => |l| try e.dsaFront(l, b, lo, hi),
        .mlp => |m| try e.mlpFront(m, b, lo, hi),
        .shared => |m| try e.sharedRows(b, m, lo, hi),
    }
}

/// hcsplit.glue: the glue on own rows, swapped with the peer's piece by piece; notes/glm_forward.md#row-split
pub fn splitGlue(e: *Engine, b: *const Buffers, hc: ?wt.Hc, norm: u64, tap_layer: ?u32, final: bool, front: Front) !void {
    const c = e.w.cfg;
    const d: usize = c.hidden;
    const s: usize = c.hc;
    const sp = &e.split.?;
    var outs: [8]u64 = undefined;
    var no: usize = 0;
    if (tap_layer) |tl| for (e.tap_layers, 0..) |t, slot| if (t == tl and b.taps[slot] != 0) {
        outs[no] = b.taps[slot];
        no += 1;
    };
    if (final) {
        outs[no] = b.hidden;
        no += 1;
    }
    if (hc != null) {
        outs[no] = b.normed;
        no += 1;
    }
    for (sp.pieces[0..sp.n], 0..) |pc, j| {
        const lo = e.mine() + pc[0];
        const n = pc[1];
        if (sp.side != null) try e.s.wait(sp.ev_part[j]);
        // glue.hc_post_pair: rank 1's partial rows read at their distance from rank 0's (may be negative)
        const own = b.part + lo * d * f4;
        const got = b.gath + pc[0] * d * f4;
        const g0, const g1 = if (e.w.rank == 0) .{ own, got } else .{ got, own };
        const rs = @divExact(@as(i64, @intCast(g1)) - @as(i64, @intCast(g0)), f4);
        const x = b.x + lo * s * d * bf;
        try e.t.hcPostPair(x, x, g0, b.post + lo * s * f4, b.comb + lo * s * s * f4, rs, n, d, s);
        if (tap_layer) |tl| for (e.tap_layers, 0..) |t, slot| if (t == tl and b.taps[slot] != 0) try e.t.streamMean(x, b.taps[slot] + lo * d * bf, n, d, s);
        if (final) try e.t.streamMean(x, b.hidden + lo * d * bf, n, d, s);
        if (hc) |h| try e.hcPreRows(b, h, norm, lo, n);
        if (sp.side) |side| {
            try sp.ev_glue[j].record(e.s);
            try side.wait(sp.ev_glue[j]);
            try e.swapRows(side, j, outs[0..no]);
        } else try e.swapRows(e.s, j, outs[0..no]);
        try e.runFront(b, front, lo, lo + n);
    }
    if (sp.side) |side| {
        try sp.ev_done.record(side);
        try e.s.wait(sp.ev_done);
    }
}

/// forward.split_layers: every layer of a prompt chunk with its glue split by rows; notes/glm_forward.md#row-split
pub fn splitLayers(e: *Engine, b: *Buffers, rows: usize) !void {
    try e.splitBegin(b, rows);
    const fronts = e.split.?.set.fronts;
    const layers = e.w.layers;
    try e.hcPreRows(b, layers[0].attn_hc.?, layers[0].in_norm, 0, e.split.?.rp);
    var done: ?[2]usize = null;
    for (layers, 0..) |*l, i| {
        _ = try e.mixer(l, b, rows, e.st.pos, done);
        const ff: Front = if (!fronts) .none else if (l.mlp) |m| .{ .mlp = m } else .{ .shared = l.moe.?.shared };
        try e.splitGlue(b, l.ffn_hc.?, l.post_norm, null, false, ff);
        const own: [2]usize = .{ e.mine(), @min(e.mine() + e.split.?.h, rows) };
        _ = if (l.mlp) |m| try e.mlpBlock(m, b, rows, l.index, if (fronts) own else null) else try e.moeBlock(l.moe.?, b, rows, l.index, if (fronts) own else null);
        const nxt: ?*const wt.Layer = if (i + 1 < layers.len) &layers[i + 1] else null;
        const mf: Front = if (!fronts or nxt == null) .none else switch (nxt.?.kind) {
            .kda => .{ .kda = nxt.? },
            .mla => .{ .dsa = nxt.? },
        };
        try e.splitGlue(b, if (nxt) |n| n.attn_hc.? else null, if (nxt) |n| n.in_norm else 0, l.index, nxt == null, mf);
        done = if (fronts and nxt != null) own else null;
    }
    try e.splitFinish();
}

/// forward.compute with host_pos: logits of every row (decode) or of the last row (a prompt chunk).
pub fn compute(e: *Engine, b: *Buffers, rows: usize) !void {
    const prefetch = !b.prefill and !e.l2pf_active and if (e.l2pf) |p| rows <= p.s.rows else false;
    if (!prefetch) return e.computeWindow(b, rows);
    e.l2pf_active = true; // TF_GLM_L2PF: this decode window's sites prefetch (joined below)
    defer e.l2pf_active = false;
    try e.computeWindow(b, rows);
    try e.l2pf.?.join(e.s);
}

/// The L2 prefetch at `layer`'s site `which` (0 a, 1 f, 2 o) while a decode window runs.
pub fn pfSite(e: *Engine, layer: u32, which: usize) !void {
    if (e.l2pf_active) try e.l2pf.?.site(e.k, e.s, layer, which);
}

pub fn computeWindow(e: *Engine, b: *Buffers, rows: usize) !void {
    const c = e.w.cfg;
    const d: usize = c.hidden;
    try e.t.embed(b.ids, e.w.embed, b.x, rows, d, c.hc);
    if (e.splitApplies(b, rows)) {
        try e.splitLayers(b, rows); // also leaves the taps and b.hidden
    } else {
        for (e.w.layers) |*l| {
            try e.layerForward(l, b, rows, e.st.pos);
            for (e.tap_layers, 0..) |tl, slot| if (tl == l.index and b.taps[slot] != 0) try e.t.streamMean(b.x, b.taps[slot], rows, d, c.hc);
        }
        try e.t.streamMean(b.x, b.hidden, rows, d, c.hc);
    }
    try e.t.rmsnorm(b.hidden, d, e.w.norm, b.fnormed, d, b.fxs, rows, d, c.eps);
    if (b.prefill) {
        try e.mm(b, b.fnormed + (rows - 1) * d * bf, d, b.fxs, e.w.head, b.logits, 1, false);
    } else try e.mm(b, b.fnormed, d, b.fxs, e.w.head, b.logits, rows, false);
}

/// A decode window's forward: logits [R, V/2] in buf.logits.
pub fn forward(e: *Engine, tokens_in: []const u32) !void {
    const rows = try e.stage(&e.buf, tokens_in);
    e.st.row0 = 0;
    e.st.seg = false; // a lone window: its commit is the lone one (a segmented window left uncommitted, e.g. timing, is dropped)
    if (e.graphKey(.main, rows, e.st.pos)) |key| return e.replay(key, compute, .{ e, &e.buf, rows });
    try e.compute(&e.buf, rows);
}
