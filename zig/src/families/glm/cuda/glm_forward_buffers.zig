//! The forward engine's device buffers: the bump allocator, the window buffers, a sequence's state and the lane pool.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const tri = @import("glm_triton.zig");
const kern = @import("glm_kernels.zig");
const segments = @import("glm_segments.zig");
const fwd = @import("glm_forward.zig");

const Engine = fwd.Engine;
const Weights = wt.Weights;
const bf = fwd.bf;
const f4 = fwd.f4;

/// A bump allocator over one device allocation, 256-byte aligned pieces.
pub const Arena = struct {
    base: u64,
    used: usize = 0,
    size: usize,

    pub fn take(a: *Arena, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + bytes;
        std.debug.assert(a.used <= a.size or a.size == 0);
        return a.base + at;
    }
};

/// forward.Buffers: scratch for windows of up to `rows` rows; `prefill` for prompt chunks.
pub const Buffers = struct {
    rows: usize,
    prefill: bool,
    zero_first: bool = false,
    ids: u64,
    hin: u64,
    x: u64,
    normed: u64,
    xs: u64,
    post: u64,
    comb: u64,
    hcpart: u64,
    ka: u64,
    kg: u64,
    xs_fa: u64,
    xs_ga: u64,
    kxs: u64,
    kproj: u64, // prefill: [rows, width]
    kscr: KdaScratch, // prefill: one layer's chain outputs
    q_tmp: u64, // the wide chain's scratch
    y_tmp: u64,
    dp: u64,
    qr: u64,
    xs_qr: u64,
    lat: u64,
    xs_lat: u64,
    q: u64,
    vn: u64,
    xs_ao: u64,
    ikr: u64,
    igr: u64,
    qi: u64,
    gu: u64,
    act: u64,
    xs_act: u64,
    mlog: u64,
    rpart: u64, // the router's K-slice partials
    pick: u64,
    wts: u64,
    exl3: kern.Exl3Scratch,
    sgu: u64,
    sact: u64,
    sxs: u64,
    sy: u64,
    part: u64,
    gath: u64,
    sk: u64,
    hidden: u64,
    fnormed: u64,
    fxs: u64,
    logits: u64,
    me: u64,
    mcat: u64,
    mxs: u64,
    mx: u64,
    lat_s: tri.Tri.Scratch,
    qa: u64,
    ol: u64,
    tokens: u64,
    counts: u64,
    scores: u64,
    pools: u64,
    split: u64, // sparse.select_split's int32 scratch (the rows that take it: under 64)
    spo: u64,
    spm: u64,
    spl: u64,
    taps: [5]u64, // DFlash2's taps: the stream means after five layers, [rows, D] each (0: off)
    pos_at: u64, // block positions, 16 bytes apart (TORCH-GLUE `pos_dev + r0`)
    greedy: u64, // [rows, 2] u32: this rank's (value, token)
    greedy_all: u64, // [2, rows, 2] u32 gathered
    cand: u64, // [rows, 2 * max_k] u32: this rank's sampled candidates
    cand_all: u64, // [2, rows, 2 * max_k] gathered

    pub const pos_slots = 16;
};

pub const KdaScratch = struct { out: u64, k: u64, v: u64, g: u64, b: u64 };

/// Sizes every buffer for `rows`, then carves them out of `a` (a.size 0: count only).
pub fn carve(a: *Arena, w: *const Weights, rows: usize, capacity: usize, prefill: bool, np_max: usize, tap_n: usize) Buffers {
    const c = w.cfg;
    const d: usize = c.hidden;
    const s: usize = c.hc;
    const hl: usize = c.mla_heads / wt.world;
    const ll: usize = c.kda_heads / wt.world;
    const qk: usize = c.nope;
    const lw: usize = c.kv_lora;
    const slots: usize = c.topk + 1;
    const e_n: usize = c.experts;
    const ml: usize = c.moe_inter / wt.world;
    const sl = ml;
    const dl: usize = c.dense_inter / wt.world;
    const width = 3 * ll * 128 + 2 * 128 + ll;
    const head_rows: usize = if (prefill) 1 else rows;
    const part_rows = if (prefill) @min(rows, tri.prompt_att_rows) else rows;
    const nch = tri.chunksFor(@min(capacity, 2560) + rows);
    const snch = tri.chunksFor(tri.select_width);
    const srows = @min(rows, tri.select_rows);
    const bs = rows * slots;
    const maxu = @min(bs, e_n);
    var b: Buffers = undefined;
    b.rows = rows;
    b.prefill = prefill;
    b.zero_first = false;
    b.ids = a.take(rows * 4);
    for (&b.taps, 0..) |*t, i| t.* = if (i < tap_n) a.take(rows * d * bf) else 0;
    b.hin = a.take(rows * d * bf);
    b.x = a.take(rows * s * d * bf);
    b.normed = a.take(rows * d * bf);
    b.xs = a.take(rows * d / 64 * f4);
    b.post = a.take(rows * s * f4);
    b.comb = a.take(rows * s * s * f4);
    b.hcpart = a.take(rows * tri.hc_blocks * 32 * f4);
    b.ka = a.take(rows * ll * 128 * bf);
    b.kg = a.take(rows * ll * 128 * bf);
    b.xs_fa = a.take(rows * 2 * f4);
    b.xs_ga = a.take(rows * 2 * f4);
    b.kxs = a.take(rows * ll * 128 / 64 * f4);
    b.kproj = if (prefill) a.take(rows * width * bf) else 0;
    b.kscr = if (prefill) .{ .out = a.take(rows * ll * 128 * bf), .k = a.take(rows * ll * 128 * f4), .v = a.take(rows * ll * 128 * bf), .g = a.take(rows * ll * 128 * f4), .b = a.take(rows * ll * f4) } else .{ .out = 0, .k = 0, .v = 0, .g = 0, .b = 0 };
    const wide = rows >= kern.wide_rows;
    b.q_tmp = if (wide) a.take(rows * ll * 128 * f4) else 0;
    b.y_tmp = if (wide) a.take(rows * ll * 128 * bf) else 0;
    b.dp = a.take(rows * (c.q_lora + c.kv_lora) * bf);
    b.qr = a.take(rows * c.q_lora * bf);
    b.xs_qr = a.take(rows * c.q_lora / 64 * f4);
    b.lat = a.take(rows * lw * bf);
    b.xs_lat = a.take(rows * lw / 64 * f4);
    b.q = a.take(rows * hl * qk * bf);
    b.vn = a.take(rows * hl * c.v_dim * bf);
    b.xs_ao = a.take(rows * hl * c.v_dim / 64 * f4);
    b.ikr = a.take(rows * (c.i_dim + c.i_heads) * bf);
    b.igr = a.take(rows * c.i_dim * f4);
    b.qi = a.take(rows * c.i_heads * c.i_dim * bf);
    b.gu = a.take(rows * 2 * dl * bf);
    b.act = a.take(rows * dl * bf);
    b.xs_act = a.take(rows * dl / 64 * f4);
    b.mlog = a.take(rows * e_n * f4);
    b.rpart = a.take(tri.router_ks * rows * e_n * f4);
    b.pick = a.take(rows * slots * 4);
    b.wts = a.take(rows * slots * f4);
    b.exl3 = .{
        .rows = rows,
        .slots = slots,
        .xg = a.take(bs * d * 2),
        .xu = a.take(bs * d * 2),
        .xd = a.take(bs * ml * 2),
        .z = a.take(@max(2 * kern.Exl3Scratch.gu_sk * ml, kern.Exl3Scratch.d_sk * d) * bs * f4),
        .y = a.take(bs * d * f4),
        .ids = a.take(maxu * 4),
        .count = a.take(4),
        .members = a.take(maxu * rows * 4),
        .plan = .{
            .members = a.take(bs * 4),
            .items = a.take(kern.maxItems(bs, e_n + 1, 16) * 12),
            .counts = a.take(8),
            .rank = a.take(if (bs > 1024) bs * 4 else 4),
            .hist = a.take(if (bs > 1024) (bs + 1023) / 1024 * (e_n + 1) * 4 else 4),
            .order = a.take((bs / 16 + 1024) * 4),
            .done = a.take(kern.maxItems(bs, e_n + 1, 16) * 8 * 4),
        },
    };
    b.sgu = a.take(rows * 2 * sl * bf);
    b.sact = a.take(rows * sl * bf);
    b.sxs = a.take(rows * sl / 64 * f4);
    b.sy = a.take(rows * d * f4);
    b.part = a.take(rows * d * f4);
    b.gath = a.take(wt.world * rows * d * f4);
    b.sk = a.take(8 * rows * 16384 * f4);
    b.hidden = a.take(rows * d * bf);
    b.fnormed = a.take(rows * d * bf);
    b.fxs = a.take(rows * d / 64 * f4);
    b.logits = a.take(head_rows * w.head.n * bf);
    b.me = a.take(rows * d * bf);
    b.mcat = a.take(rows * 2 * d * bf);
    b.mxs = a.take(rows * 2 * d / 64 * f4);
    b.mx = a.take(rows * d * bf);
    b.lat_s = .{ .po = a.take(nch * part_rows * hl * lw * f4), .pm = a.take(nch * part_rows * hl * f4), .pl = a.take(nch * part_rows * hl * f4), .dummy = a.take(4), .part_rows = part_rows };
    b.qa = a.take(rows * hl * lw * bf);
    b.ol = a.take(rows * hl * lw * bf);
    b.tokens = a.take(rows * tri.select_width * 4);
    b.counts = a.take(rows * 4);
    b.scores = a.take(srows * np_max * f4);
    b.pools = a.take(rows * tri.topk_pools * 8);
    b.split = a.take(tri.Tri.splitInts(@min(rows, tri.split_rows - 1), np_max) * 4);
    b.spo = a.take(snch * rows * hl * lw * f4);
    b.spm = a.take(snch * rows * hl * f4);
    b.spl = a.take(snch * rows * hl * f4);
    b.pos_at = a.take(Buffers.pos_slots * 16);
    b.greedy = a.take(rows * 8);
    b.greedy_all = a.take(wt.world * rows * 8);
    const crows: usize = @min(rows, Engine.max_rows);
    b.cand = a.take(crows * 2 * Engine.max_k * 4);
    b.cand_all = a.take(wt.world * crows * 2 * Engine.max_k * 4);
    return b;
}

/// forward.State: one sequence's committed caches, and the MTP head's attention layer's.
pub const State = struct {
    capacity: usize,
    pos: usize = 0,
    mtp_len: usize = 0,
    mtp_drafted: usize = 0,
    pos_dev: u64,
    mtp_pos_dev: u64,
    kda_layers: usize,
    dsa_layers: usize,
    kda_index: [64]u8 = @splat(0),
    dsa_index: [64]u8 = @splat(0),
    conv: u64, // [n, 3, 3 LL 128] bf16
    rec: u64, // [2, n, LL, 128, 128] f32
    cur: usize = 0, // which half of rec is committed (every layer flips together)
    proj: u64, // [n, rows, width] bf16
    scr: KdaScratch, // [n, rows, ...]: each layer's window outputs for a replay
    scr_rows: usize,
    kc: [16]u64 = @splat(0), // [capacity, kv_lora] bf16 a DSA layer
    mtp_kc: u64,
    index: [16][3]u64 = undefined, // ik, ig [capacity, 128], pk [capacity/4 + 2, 128] bf16; the MTP layer's last
    long_context: bool,
    kv_fp8: bool, // the latent caches and pooled keys as e4m3 rows with a scale each (Engine.kv_fp8)
    last_logits: u64 = 0, // a prompt's last row's logits, kept past the MTP absorb that reuses pbuf
    last_hidden: u64 = 0, // and its final-normed hidden row, the first draft's input
    seg: bool = false, // the last verify window was segmented (forwardSeg): the commit takes commitSeg
    row0: usize = 0, // the lane's first row in the last verify window (a shared window's segment), which drafts read
    slot: usize = 0, // the lane's slot of Engine.pool (its arena extent starts at slot x Pool.extent_rows)

    pub fn recBytes(st: *const State, ll: usize) usize {
        return st.kda_layers * ll * 128 * 128 * f4;
    }

    /// One sequence's state for `w`'s layers, carved out of `a` (a.size 0: count only). `pooled`: its KDA states and DSA
    /// layers' caches are left to Pool.view (the MTP layer's stay here).
    pub fn carveIn(a: *Arena, w: *const Weights, capacity: usize, long_context: bool, kv_fp8: bool, pooled: bool) State {
        const c = w.cfg;
        const ll: usize = c.kda_heads / wt.world;
        const width = 3 * ll * 128 + 2 * 128 + ll;
        var st: State = .{ .capacity = capacity, .pos_dev = 0, .mtp_pos_dev = 0, .kda_layers = 0, .dsa_layers = 0, .conv = 0, .rec = 0, .proj = 0, .scr = undefined, .scr_rows = Engine.max_rows, .mtp_kc = 0, .long_context = long_context, .kv_fp8 = kv_fp8 };
        for (w.layers) |l| switch (l.kind) {
            .kda => {
                st.kda_index[l.index] = @intCast(st.kda_layers);
                st.kda_layers += 1;
            },
            .mla => {
                st.dsa_index[l.index] = @intCast(st.dsa_layers);
                st.dsa_layers += 1;
            },
        };
        const n_kda = st.kda_layers;
        const mr = Engine.max_rows;
        st.pos_dev = a.take(16);
        st.mtp_pos_dev = a.take(16);
        if (!pooled) {
            st.conv = a.take(n_kda * 3 * 3 * ll * 128 * bf);
            st.rec = a.take(2 * n_kda * ll * 128 * 128 * f4);
        }
        st.proj = a.take(n_kda * mr * width * bf);
        st.scr = .{ .out = a.take(n_kda * mr * ll * 128 * bf), .k = a.take(n_kda * mr * ll * 128 * f4), .v = a.take(n_kda * mr * ll * 128 * bf), .g = a.take(n_kda * mr * ll * 128 * f4), .b = a.take(n_kda * mr * ll * f4) };
        if (!pooled) for (0..st.dsa_layers) |i| {
            st.kc[i] = a.take(capacity * st.latentRow(c));
        };
        st.mtp_kc = a.take(capacity * st.latentRow(c));
        st.last_logits = a.take(w.head.n * bf);
        st.last_hidden = a.take(c.hidden * bf);
        const own_from: usize = if (pooled) st.dsa_layers else 0; // the MTP layer's indexer stays a lane's own
        if (long_context) for (own_from..st.dsa_layers + 1) |i| {
            st.index[i] = .{ a.take(capacity * c.i_dim * bf), a.take(capacity * c.i_dim * bf), a.take((capacity / 4 + 2) * st.pooledRow(c)) };
        };
        return st;
    }

    /// A latent cache row's bytes (kv8.row_bytes with fp8).
    pub fn latentRow(st: *const State, c: anytype) usize {
        return if (st.kv_fp8) c.kv_lora + 16 else c.kv_lora * bf;
    }

    /// A pooled key row's bytes.
    pub fn pooledRow(st: *const State, c: anytype) usize {
        return if (st.kv_fp8) c.i_dim + 16 else c.i_dim * bf;
    }

    /// carveIn's bytes, its alignment included.
    pub fn bytes(w: *const Weights, capacity: usize, long_context: bool, kv_fp8: bool, pooled: bool) usize {
        var count: Arena = .{ .base = 0, .size = 0 };
        _ = carveIn(&count, w, capacity, long_context, kv_fp8, pooled);
        return count.used + 256;
    }
};

/// A lane's DSA extent starts at a multiple of this many tokens (v1.10 segments.EXTENT; its pooled keys at base / 4).
pub const extent = 2048;

/// Every lane's KDA states and DSA layers' caches in one allocation, laid out as v1.10's slots and arenas (multi.py's
/// slots, segments.py) so the segmented kernels reach any lane from one base: KDA rec [slots, 2, layers, LL, 128, 128] f32
/// and conv [slots, layers, 3, 3 LL 128] bf16; a DSA layer's latents [slots x extent_rows, row], index keys and gates
/// [slots x extent_rows, 128] bf16 (no rings: token t of slot s at s x extent_rows + t, segments.SegRows' ibase = base),
/// pooled keys [slots x extent_rows / 4 + 2, row]. Each lane's State points into its slot (Pool.view), so its own calls
/// run as before. Also the segmented window's KDA scratch (one layer set of `max_rows` rows) and tables.
pub const Pool = struct {
    mem: cuda.DeviceBuffer,
    slots: usize,
    extent_rows: usize, // a slot's tokens in the DSA arenas: the lane capacity rounded up to `extent`
    rec: u64,
    rec_slot: usize, // bytes; parity stride rec_slot / 2, layer stride LL x 128 x 128 x 4
    conv: u64,
    conv_slot: usize, // bytes; layer stride 3 x 3 LL 128 x 2
    kc: [16]u64 = @splat(0), // a DSA layer's latent arena
    ik: [16]u64 = @splat(0), // index keys, gates and pooled keys (long contexts only)
    ig: [16]u64 = @splat(0),
    pk: [16]u64 = @splat(0),
    lat_row: usize,
    pk_row: usize,
    /// The segmented window's KDA projection rows and replay saves, [layers, shared_rows, ...] (kda.KDAScratchSet's layout).
    proj: u64,
    scr: KdaScratch,
    /// kda.segment_table [max_lanes, kern.seg_cols] i32 and segments.SegRows' tables (glm_segments.zig), device copies.
    kseg: u64,
    segrows: u64,

    /// A slot's tokens in the arenas for lanes of `capacity`: capacity rounded up to `extent`, with room for the pooled
    /// keys' two rows past capacity / 4 (a slot's pooled keys must end before the next slot's begin).
    pub fn extentRows(capacity: usize) usize {
        var n = std.mem.alignForward(usize, capacity, extent);
        if (capacity / 4 + 2 > n / 4) n += extent;
        return n;
    }

    /// Carves a pool of `slots` lanes out of `a` (a.size 0: count only).
    pub fn carvePool(a: *Arena, w: *const Weights, capacity: usize, long_context: bool, kv_fp8: bool, slots: usize) Pool {
        const c = w.cfg;
        const ll: usize = c.kda_heads / wt.world;
        const width = 3 * ll * 128 + 2 * 128 + ll;
        var n_kda: usize = 0;
        var n_dsa: usize = 0;
        for (w.layers) |l| switch (l.kind) {
            .kda => n_kda += 1,
            .mla => n_dsa += 1,
        };
        const ext = extentRows(capacity);
        const mr = Engine.shared_rows;
        var p: Pool = undefined;
        p.mem = undefined;
        p.slots = slots;
        p.extent_rows = ext;
        p.rec_slot = std.mem.alignForward(usize, 2 * n_kda * ll * 128 * 128 * f4, 256);
        p.conv_slot = std.mem.alignForward(usize, n_kda * 3 * 3 * ll * 128 * bf, 256);
        p.lat_row = if (kv_fp8) c.kv_lora + 16 else c.kv_lora * bf;
        p.pk_row = if (kv_fp8) c.i_dim + 16 else c.i_dim * bf;
        p.rec = a.take(slots * p.rec_slot);
        p.conv = a.take(slots * p.conv_slot);
        p.kc = @splat(0);
        p.ik = @splat(0);
        p.ig = @splat(0);
        p.pk = @splat(0);
        for (0..n_dsa) |i| {
            p.kc[i] = a.take(slots * ext * p.lat_row);
            if (long_context) {
                p.ik[i] = a.take(slots * ext * c.i_dim * bf);
                p.ig[i] = a.take(slots * ext * c.i_dim * bf);
                p.pk[i] = a.take((slots * ext / 4 + 2) * p.pk_row);
            }
        }
        p.proj = a.take(n_kda * mr * width * bf);
        p.scr = .{ .out = a.take(n_kda * mr * ll * 128 * bf), .k = a.take(n_kda * mr * ll * 128 * f4), .v = a.take(n_kda * mr * ll * 128 * bf), .g = a.take(n_kda * mr * ll * 128 * f4), .b = a.take(n_kda * mr * ll * f4) };
        p.kseg = a.take(Engine.max_lanes * kern.seg_cols * 4);
        p.segrows = a.take(segments.SegRows.words * 4);
        return p;
    }

    /// Bytes of a pool of `slots` lanes.
    pub fn bytes(w: *const Weights, capacity: usize, long_context: bool, kv_fp8: bool, slots: usize) usize {
        var count: Arena = .{ .base = 0, .size = 0 };
        _ = carvePool(&count, w, capacity, long_context, kv_fp8, slots);
        return count.used + 256;
    }

    /// Points `st`'s KDA states and DSA caches at slot `slot`.
    pub fn view(p: *const Pool, st: *State, slot: usize, i_dim: usize) void {
        st.slot = slot;
        st.rec = p.rec + slot * p.rec_slot;
        st.conv = p.conv + slot * p.conv_slot;
        const base = slot * p.extent_rows;
        for (0..st.dsa_layers) |i| {
            st.kc[i] = p.kc[i] + base * p.lat_row;
            if (st.long_context) st.index[i] = .{ p.ik[i] + base * i_dim * bf, p.ig[i] + base * i_dim * bf, p.pk[i] + base / 4 * p.pk_row };
        }
    }
};
