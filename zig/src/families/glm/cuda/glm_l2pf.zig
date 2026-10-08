//! TF_GLM_L2PF (v1.8 glm5_next/cuda/l2pf.py): in decode windows a small kernel on a side stream asks the memory system to
//! bring the weights the main stream reads next into L2 while it waits on an all-gather or runs latency-bound glue. It only
//! loads or prefetches, so no result changes. Three sites a layer: `a` at the attention block's all-gather (the FFN's
//! hyper-connection weights and norm, the router and shared expert, or a dense MLP), `f` at the FFN's all-gather (the next
//! layer's hyper-connection weights, norm and input projections; after the last layer the final norm and the head), `o` after
//! the attention block's input projections (its output projection). Each site takes up to TF_GLM_L2PF_MB (default 8) MiB:
//! small tensors first, then a 4-bit matrix's chunk heads where its decode launch is one wave, else the leading bytes of
//! the large tensors. The side stream forks from the main stream at a site (an event) and is joined before a window ends,
//! so it also works inside a captured graph.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("glm_weights.zig");
const kern = @import("glm_kernels.zig");
const max_layers = @import("glm_config").max_layers;

const align_b = 16; // cp.async.bulk wants 16-byte addresses and sizes
const piece: usize = 32 << 10; // l2pf.PIECE
const one_wave = 192; // l2pf.ONE_WAVE: qmm programs resident at once

pub const Settings = struct {
    mode: i32 = -1, // -1: off; 0 bulk, 1 lines, 2 touch
    mb: f64 = 8.0,
    sites: [3]bool = .{ true, true, true }, // a, f, o
    rows: usize = 64,
    threads: usize = 128,

    pub fn fromEnv() !Settings {
        var s: Settings = .{};
        const v = std.c.getenv("TF_GLM_L2PF");
        // v1.8's production setting turns it on; here it is on unless "0" / "off"
        const mode = if (v) |x| std.mem.span(x) else "1";
        if (std.mem.eql(u8, mode, "0") or std.mem.eql(u8, mode, "off")) return s;
        s.mode = if (std.mem.eql(u8, mode, "1") or std.mem.eql(u8, mode, "bulk") or mode.len == 0) 0 else if (std.mem.eql(u8, mode, "lines")) 1 else if (std.mem.eql(u8, mode, "touch")) 2 else return error.BadL2pf;
        if (std.c.getenv("TF_GLM_L2PF_MB")) |x| s.mb = try std.fmt.parseFloat(f64, std.mem.span(x));
        if (!(s.mb > 0 and s.mb <= 64)) return error.BadL2pf;
        if (std.c.getenv("TF_GLM_L2PF_SITES")) |x| {
            s.sites = .{ false, false, false };
            for (std.mem.span(x)) |ch| switch (ch) {
                'a' => s.sites[0] = true,
                'f' => s.sites[1] = true,
                'o' => s.sites[2] = true,
                else => return error.BadL2pf,
            };
        }
        if (std.c.getenv("TF_GLM_L2PF_ROWS")) |x| s.rows = try std.fmt.parseInt(usize, std.mem.span(x), 10);
        return s;
    }

    pub fn on(s: Settings) bool {
        return s.mode >= 0;
    }
};

const T = struct { ptr: u64, bytes: usize };
const Q4Head = struct { n: usize, k: usize, weight: T };
const Group = struct {
    ts: [8]T = undefined,
    n: usize = 0,
    q4: ?Q4Head = null,

    fn add(g: *Group, ptr: u64, bytes: usize) void {
        if (ptr == 0 or bytes == 0) return;
        for (g.ts[0..g.n]) |t| if (t.ptr == ptr) return; // each tensor once
        g.ts[g.n] = .{ .ptr = ptr, .bytes = bytes };
        g.n += 1;
    }

    fn mat(g: *Group, m: wt.Mat) void {
        switch (m.kind) {
            .b16 => g.add(m.ptr, m.n * m.k * 2),
            .f8 => {
                g.add(m.ptr, m.n * m.k);
                g.add(m.s, m.n * (m.k / 128) * 4);
            },
            .q4 => {
                const kg = m.k / 64;
                g.add(m.ptr, m.npad * m.k / 2);
                g.add(m.s, kg * m.npad * 2);
                g.add(m.b, kg * m.npad * 2);
                g.q4 = .{ .n = m.n, .k = m.k, .weight = .{ .ptr = m.ptr, .bytes = m.npad * m.k / 2 } };
            },
        }
    }
};

/// l2pf._span: the tensor's leading bytes (at most `left`), the start rounded down and the end up to 16 bytes.
fn span(t: T, left: usize) ?T {
    const lo = t.ptr - t.ptr % align_b;
    var hi = @min(t.ptr + t.bytes, lo + left);
    hi += (align_b - hi % align_b) % align_b;
    return if (hi > lo) .{ .ptr = lo, .bytes = hi - lo } else null;
}

/// l2pf.ranges: (address, bytes) ranges of `groups` up to `budget` bytes, appended to `out`.
fn ranges(gpa: std.mem.Allocator, groups: []const Group, budget: usize, out: *std.ArrayList(T)) !void {
    var left = budget;
    for (groups) |g0| {
        var g = g0;
        // within a group the small tensors first (a stable sort by size)
        std.mem.sort(T, g.ts[0..g.n], {}, struct {
            fn lt(_: void, a: T, b: T) bool {
                return a.bytes < b.bytes;
            }
        }.lt);
        var heads: std.ArrayList(T) = .empty;
        defer heads.deinit(gpa);
        var skip: u64 = 0;
        if (g.q4) |q| {
            var small: usize = 0;
            for (g.ts[0..g.n]) |t| if (t.ptr != q.weight.ptr) {
                small += t.bytes;
            };
            if (left > small) try q4Heads(gpa, q, left - small, &heads);
            if (heads.items.len > 0) skip = q.weight.ptr;
        }
        for (g.ts[0..g.n]) |t| {
            if (skip != 0 and t.ptr == skip) continue;
            if (left < align_b) return;
            if (span(t, left)) |sp| {
                try out.append(gpa, sp);
                left -= sp.bytes;
            }
        }
        if (heads.items.len > 0) {
            var total: usize = 0;
            for (heads.items) |h| total += h.bytes;
            if (left >= total) {
                try out.appendSlice(gpa, heads.items);
                left -= total;
            }
        }
    }
}

/// l2pf.q4_heads: the first `left / chunks` bytes of every (64-column tile, K slice) chunk of a 4-bit matrix whose decode launch is
/// one wave, so each program starts on L2 hits.
fn q4Heads(gpa: std.mem.Allocator, q: Q4Head, left: usize, out: *std.ArrayList(T)) !void {
    const chunks = (q.n + 63) / 64 * kern.denseSplitK(q.n, q.k);
    const total = q.weight.bytes;
    if (chunks == 0 or chunks > one_wave or total % chunks != 0) return;
    const size = total / chunks;
    const per = @min(size, left / chunks) / align_b * align_b;
    if (per < 512 or q.weight.ptr % align_b != 0 or size % align_b != 0) return;
    for (0..chunks) |c| try out.append(gpa, .{ .ptr = q.weight.ptr + c * size, .bytes = per });
}

pub const Site = struct { first: u32, count: u32, grid: u32 };

pub const Prefetch = struct {
    s: Settings,
    table: cuda.DeviceBuffer,
    sink: cuda.DeviceBuffer,
    sites: [max_layers][3]?Site = @splat(@splat(null)),
    side: cuda.Stream,
    fork: cuda.Event,
    join_ev: cuda.Event,
    forked: bool = false,
    bytes: [3]usize = .{ 0, 0, 0 },
    count: [3]usize = .{ 0, 0, 0 },

    fn norm(w: *const wt.Weights) usize {
        return @as(usize, w.cfg.hidden) * 2;
    }

    fn hc(g: *Group, w: *const wt.Weights, h: ?wt.Hc) void {
        const x = h orelse return;
        g.add(x.fn_, 24 * @as(usize, w.cfg.hc) * w.cfg.hidden * 2);
        g.add(x.base, 24 * 4);
        g.add(x.scale, 3 * 4);
    }

    /// Site `a`: the FFN's hyper-connection weights and norm, then the router and shared expert, or a dense MLP.
    fn ffn(w: *const wt.Weights, l: wt.Layer, gs: *std.ArrayList(Group), gpa: std.mem.Allocator) !void {
        var g: Group = .{};
        hc(&g, w, l.ffn_hc);
        try gs.append(gpa, g);
        var n: Group = .{};
        n.add(l.post_norm, norm(w));
        try gs.append(gpa, n);
        if (l.mlp) |m| {
            var a: Group = .{};
            a.mat(m.gu);
            try gs.append(gpa, a);
            var b: Group = .{};
            b.mat(m.down);
            try gs.append(gpa, b);
        } else if (l.moe) |m| {
            var r: Group = .{};
            r.add(m.router, @as(usize, w.cfg.experts) * w.cfg.hidden * 2);
            try gs.append(gpa, r);
            var bias: Group = .{};
            bias.add(m.bias, @as(usize, w.cfg.experts) * 4);
            try gs.append(gpa, bias);
            var a: Group = .{};
            a.mat(m.shared.gu);
            try gs.append(gpa, a);
            var b: Group = .{};
            b.mat(m.shared.down);
            try gs.append(gpa, b);
        }
    }

    /// Site `f`: the next layer's hyper-connection weights, norm and input projections; after the last layer the final norm and head.
    fn next(w: *const wt.Weights, nxt: ?wt.Layer, gs: *std.ArrayList(Group), gpa: std.mem.Allocator) !void {
        const n = nxt orelse {
            var a: Group = .{};
            a.add(w.norm, norm(w));
            try gs.append(gpa, a);
            var h: Group = .{};
            h.mat(w.head);
            try gs.append(gpa, h);
            return;
        };
        const c = w.cfg;
        var g: Group = .{};
        hc(&g, w, n.attn_hc);
        try gs.append(gpa, g);
        var nm: Group = .{};
        nm.add(n.in_norm, norm(w));
        try gs.append(gpa, nm);
        if (n.kda) |k| {
            const ll: usize = c.kda_heads / wt.world;
            inline for (.{ k.fb, k.gb }) |m| {
                var a: Group = .{};
                a.mat(m);
                try gs.append(gpa, a);
            }
            var cv: Group = .{};
            cv.add(k.conv, 3 * ll * 128 * 4 * 2);
            try gs.append(gpa, cv);
            var al: Group = .{};
            al.add(k.a_log, ll * 4);
            try gs.append(gpa, al);
            var dt: Group = .{};
            dt.add(k.dt_bias, ll * 128 * 4);
            try gs.append(gpa, dt);
            var pr: Group = .{};
            pr.mat(k.proj);
            try gs.append(gpa, pr);
        } else if (n.dsa) |a| {
            var qn: Group = .{};
            qn.add(a.q_norm, @as(usize, c.q_lora) * 2);
            try gs.append(gpa, qn);
            var kn: Group = .{};
            kn.add(a.kv_norm, @as(usize, c.kv_lora) * 2);
            try gs.append(gpa, kn);
            inline for (.{ a.proj, a.q_b, a.index.kw }) |m| {
                var x: Group = .{};
                x.mat(m);
                try gs.append(gpa, x);
            }
            var gt: Group = .{};
            gt.add(a.index.gate, @as(usize, c.i_dim) * c.hidden * 2);
            try gs.append(gpa, gt);
        }
    }

    /// Site `o`: KDA's output projection; DSA's (fired after absorb_q) kv_b's value half, then the output projection.
    fn out(w: *const wt.Weights, l: wt.Layer, gs: *std.ArrayList(Group), gpa: std.mem.Allocator) !void {
        const c = w.cfg;
        if (l.kda) |k| {
            var g: Group = .{};
            g.mat(k.o);
            try gs.append(gpa, g);
        } else if (l.dsa) |a| {
            var v: Group = .{};
            v.add(a.wv, @as(usize, c.mla_heads / wt.world) * c.v_dim * c.kv_lora * 2);
            try gs.append(gpa, v);
            var o: Group = .{};
            o.mat(a.o);
            try gs.append(gpa, o);
        }
    }

    pub fn build(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wt.Weights, s: Settings) !Prefetch {
        var p: Prefetch = .{ .s = s, .table = undefined, .sink = undefined, .side = undefined, .fork = undefined, .join_ev = undefined };
        const budget: usize = @intFromFloat(s.mb * (1 << 20));
        var rows: std.ArrayList(i64) = .empty;
        defer rows.deinit(gpa);
        for (w.layers, 0..) |l, i| {
            const nxt: ?wt.Layer = if (i + 1 < w.layers.len) w.layers[i + 1] else null;
            inline for (0..3) |si| {
                if (s.sites[si]) {
                    var gs: std.ArrayList(Group) = .empty;
                    defer gs.deinit(gpa);
                    switch (si) {
                        0 => try ffn(w, l, &gs, gpa),
                        1 => try next(w, nxt, &gs, gpa),
                        else => try out(w, l, &gs, gpa),
                    }
                    var rs: std.ArrayList(T) = .empty;
                    defer rs.deinit(gpa);
                    try ranges(gpa, gs.items, budget, &rs);
                    const first = rows.items.len / 2;
                    var total: usize = 0;
                    for (rs.items) |sp| {
                        total += sp.bytes;
                        var o: usize = 0;
                        while (o < sp.bytes) : (o += piece) {
                            try rows.append(gpa, @intCast(sp.ptr + o));
                            try rows.append(gpa, @intCast(@min(piece, sp.bytes - o)));
                        }
                    }
                    const count = rows.items.len / 2 - first;
                    if (count > 0) {
                        const per: usize = if (s.mode == 0) s.threads else s.threads / 32;
                        p.sites[l.index][si] = .{ .first = @intCast(first), .count = @intCast(count), .grid = @intCast(@max(1, @min(48, (count + per - 1) / per))) };
                        p.bytes[si] += total;
                        p.count[si] += 1;
                    }
                }
            }
        }
        if (rows.items.len == 0) try rows.appendSlice(gpa, &.{ 0, 0 });
        p.table = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(rows.items));
        errdefer p.table.free();
        p.sink = try cuda.DeviceBuffer.alloc(d, 256);
        errdefer p.sink.free();
        try p.sink.fill8(0, null);
        p.side = try cuda.Stream.init(d, true);
        errdefer p.side.deinit();
        p.fork = try cuda.Event.init(d, false);
        errdefer p.fork.deinit();
        p.join_ev = try cuda.Event.init(d, false);
        return p;
    }

    pub fn deinit(p: *Prefetch) void {
        p.join_ev.deinit();
        p.fork.deinit();
        p.side.deinit();
        p.sink.free();
        p.table.free();
    }

    /// l2pf.Prefetch.site: fork the side stream at this point of `main` and prefetch the site's ranges there.
    pub fn site(p: *Prefetch, k: *const kern.Kernels, main: cuda.Stream, layer: u32, which: usize) !void {
        const st = p.sites[layer][which] orelse return;
        try p.fork.record(main);
        try p.side.wait(p.fork);
        try kern.l2Prefetch(k, p.side, p.table.ptr, st.first, st.count, p.s.mode, st.grid, p.s.threads, p.sink.ptr);
        p.forked = true;
    }

    /// The main stream waits for the side stream (a captured graph's branches rejoin before its end).
    pub fn join(p: *Prefetch, main: cuda.Stream) !void {
        if (!p.forked) return;
        try p.join_ev.record(p.side);
        try main.wait(p.join_ev);
        p.forked = false;
    }
};
