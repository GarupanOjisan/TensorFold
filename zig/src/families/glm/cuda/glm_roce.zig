//! TF_GLM_COMM=roce (v1.10's default): small all-gathers as one-shot RDMA writes; notes/glm_roce.md#design
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("glm_kernels.zig");

extern "c" fn roce_dl_open() c_int;
extern "c" fn roce_abi_version() c_int;
extern "c" fn roce_layout(world: c_int, slot_bytes: u64, out: *[7]u64) c_int;
extern "c" fn roce_blob_bytes() u64;
extern "c" fn roce_create(world: c_int, rank: c_int, names: [*]const [*:0]const u8, n_hca: c_int, gids: [*]const c_int, tc: c_int, region: ?*anyopaque, region_bytes: u64, slot_bytes: u64, err: [*]u8, err_len: u64) ?*anyopaque;
extern "c" fn roce_local_blob(c: ?*anyopaque, out: [*]u8, len: u64) c_int;
extern "c" fn roce_connect(c: ?*anyopaque, blobs: [*]const u8, len: u64) c_int;
extern "c" fn roce_start(c: ?*anyopaque) c_int;
extern "c" fn roce_stop(c: ?*anyopaque) void;
extern "c" fn roce_failed(c: ?*anyopaque) c_int;
extern "c" fn roce_error(c: ?*anyopaque) [*:0]const u8;
extern "c" fn roce_destroy(c: ?*anyopaque) void;

const slot_align = 4096;
const pack = 16;
const threads = 512;
const blocks = 8; // the largest grid; smaller shards launch fewer (power-of-two) blocks
const classes = 4; // blocks.bit_length()
const world = 2;

/// TF_GLM_COMM: `nccl` all NCCL, `roce` required, unset: RoCE when it sets up on both ranks; notes/glm_roce.md#mode
pub const Mode = enum { nccl, roce, auto };

pub fn mode(default: Mode) !Mode {
    const v = std.c.getenv("TF_GLM_COMM") orelse return default;
    const s = std.mem.trim(u8, std.mem.span(v), " ");
    if (s.len == 0) return default;
    if (std.mem.eql(u8, s, "nccl")) return .nccl;
    if (std.mem.eql(u8, s, "roce")) return .roce;
    if (std.mem.eql(u8, s, "auto")) return .auto;
    std.log.err("TF_GLM_COMM is nccl, roce or auto, not {s}", .{s});
    return error.BadComm;
}

fn envInt(name: [*:0]const u8) ?i64 {
    const v = std.c.getenv(name) orelse return null;
    const s = std.mem.trim(u8, std.mem.span(v), " ");
    if (s.len == 0) return null;
    return std.fmt.parseInt(i64, s, 0) catch null;
}

/// TF_ROCE_MAX_KB (the recipe's default 512): the largest gather a rank sends over RoCE.
pub fn maxBytes() usize {
    const v = envInt("TF_ROCE_MAX_KB") orelse 512;
    return @intCast(std.math.clamp(v, 1, 4096) << 10);
}

pub const Roce = struct {
    gpa: std.mem.Allocator,
    ctx: ?*anyopaque,
    region: cuda.HostBuffer,
    counters: cuda.DeviceBuffer,
    gather_fn: cuda.Function,
    max_bytes: usize,
    slot_bytes: usize,
    flag_stride: usize,
    slots: usize,
    n_hca: usize,
    spin: u32,
    rank: u32,
    base: u64, // the region's device address
    recv: u64,
    flag: u64,
    send: u64,
    ctrl: u64,
    ctrl_words: [*]volatile u32,

    /// Both ranks' HCAs, pinned region and proxy threads, connected through `pair`; notes/glm_roce.md#init
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const kern.Kernels, pair: cuda.Pair, rank: u32, max_bytes: usize) !*Roce {
        var r = try gpa.create(Roce);
        errdefer gpa.destroy(r);
        r.gpa = gpa;
        r.ctx = null;
        r.rank = rank;
        r.max_bytes = max_bytes;
        r.gather_fn = k.roce_gather;
        r.slot_bytes = (max_bytes + slot_align - 1) / slot_align * slot_align;
        var layout: [7]u64 = undefined;
        if (roce_layout(world, r.slot_bytes, &layout) != 0) return error.RoceLayout;
        const total: usize = @intCast(layout[4]);
        r.flag_stride = @intCast(layout[5]);
        r.slots = @intCast(layout[6]);
        r.region = try cuda.HostBuffer.allocMapped(d, total);
        errdefer r.region.free();
        @memset(r.region.bytes, 0); // flags and ctrl start at 0
        r.base = try r.region.device();
        r.recv = r.base + layout[0];
        r.flag = r.base + layout[1];
        r.send = r.base + layout[2];
        r.ctrl = r.base + layout[3];
        r.ctrl_words = @ptrCast(@alignCast(r.region.bytes.ptr + layout[3]));
        r.counters = try cuda.DeviceBuffer.alloc(d, (2 + 2 * classes) * 4);
        errdefer r.counters.free();
        try r.counters.fill8(0, null);
        const wait_s: f64 = if (std.c.getenv("TF_ROCE_WAIT_S")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 300.0) else 300.0;
        r.spin = @intFromFloat(@min(@max(wait_s, 1.0), 3600.0) * 1e6);

        // the local side: the HCAs, their queue pairs and the connection blob
        var err: [512]u8 = undefined;
        @memset(&err, 0);
        var failed: u32 = 0;
        var hca_names: [world][64:0]u8 = undefined;
        var name_ptrs: [world][*:0]const u8 = undefined;
        var n_hca: usize = 0;
        const raw = std.c.getenv("TF_ROCE_HCA") orelse std.c.getenv("NCCL_IB_HCA");
        if (raw) |text| {
            var it = std.mem.splitScalar(u8, std.mem.span(text), ',');
            while (it.next()) |part| {
                var name = std.mem.trim(u8, part, " =^");
                if (std.mem.indexOfScalar(u8, name, ':')) |c| name = name[0..c];
                if (name.len == 0 or name.len >= 64 or n_hca == world) continue;
                @memcpy(hca_names[n_hca][0..name.len], name);
                hca_names[n_hca][name.len] = 0;
                name_ptrs[n_hca] = &hca_names[n_hca];
                n_hca += 1;
            }
        }
        const blob_len: usize = @intCast(roce_blob_bytes());
        const blob_mem = try gpa.alloc(u8, 2 * blob_len);
        defer gpa.free(blob_mem);
        @memset(blob_mem, 0);
        const mine = blob_mem[0..blob_len];
        if (n_hca == 0) {
            failed = 1;
            _ = std.fmt.bufPrint(&err, "no HCA named (TF_ROCE_HCA or NCCL_IB_HCA)", .{}) catch {};
        } else if (roce_dl_open() == 0) {
            failed = 1;
            _ = std.fmt.bufPrint(&err, "libibverbs.so.1 does not load", .{}) catch {};
        } else if (roce_abi_version() != 5) {
            failed = 1;
            _ = std.fmt.bufPrint(&err, "unexpected RoCE proxy ABI version", .{}) catch {};
        } else {
            const gid: c_int = @intCast(envInt("TF_ROCE_GID_INDEX") orelse envInt("NCCL_IB_GID_INDEX") orelse 3);
            const tc: c_int = @intCast(envInt("NCCL_IB_TC") orelse 0);
            var gids: [world]c_int = @splat(gid);
            r.ctx = roce_create(world, @intCast(rank), &name_ptrs, @intCast(n_hca), &gids, tc, r.region.bytes.ptr, total, r.slot_bytes, &err, err.len);
            if (r.ctx == null or roce_local_blob(r.ctx, mine.ptr, blob_len) != 0) failed = 1;
        }
        r.n_hca = n_hca;

        // both ranks' verdict and geometry, and the blobs
        const head_mine = [4]u32{ failed, @intCast(n_hca), @intCast(r.slot_bytes >> 12), @intCast(r.flag_stride) };
        var head_theirs: [4]u32 = undefined;
        const theirs = blob_mem[blob_len..];
        try swap(pair, rank, std.mem.asBytes(&head_mine), std.mem.asBytes(&head_theirs));
        try swap(pair, rank, mine, theirs);
        const same = head_mine[1] == head_theirs[1] and head_mine[2] == head_theirs[2] and head_mine[3] == head_theirs[3];
        if (failed != 0 or head_theirs[0] != 0 or !same) {
            const why: []const u8 = if (failed != 0) std.mem.sliceTo(&err, 0) else if (!same) "the ranks' RoCE geometry differs" else "the other rank's setup failed";
            std.log.warn("RoCE all-gather setup failed: {s}", .{why});
            r.close();
            return error.RoceSetupFailed;
        }
        // rank order: rank 0's blob first
        if (rank == 1) {
            const tmp = try gpa.alloc(u8, blob_len);
            defer gpa.free(tmp);
            @memcpy(tmp, blob_mem[0..blob_len]);
            @memcpy(blob_mem[0..blob_len], blob_mem[blob_len..]);
            @memcpy(blob_mem[blob_len..], tmp);
        }
        var ok: u32 = if (roce_connect(r.ctx, blob_mem.ptr, blob_mem.len) == 0 and roce_start(r.ctx) == 0) 0 else 1;
        var verdict: u32 = undefined;
        try swap(pair, rank, std.mem.asBytes(&ok), std.mem.asBytes(&verdict));
        if (ok != 0 or verdict != 0) {
            std.log.warn("RoCE all-gather could not connect: {s}", .{if (r.ctx != null) std.mem.span(roce_error(r.ctx)) else "no context"});
            r.close();
            return error.RoceConnectFailed;
        }
        return r;
    }

    /// Rank 0 sends then receives, rank 1 the other way round.
    fn swap(pair: cuda.Pair, rank: u32, out: []const u8, in: []u8) !void {
        if (rank == 0) {
            try pair.send(out);
            try pair.recv(in);
        } else {
            try pair.recv(in);
            try pair.send(out);
        }
    }

    pub fn close(r: *Roce) void {
        if (r.ctx != null) {
            roce_stop(r.ctx);
            roce_destroy(r.ctx);
            r.ctx = null;
        }
    }

    pub fn deinit(r: *Roce) void {
        r.close();
        r.counters.free();
        r.region.free();
        r.gpa.destroy(r);
    }

    /// Whether a gather of `bytes` a rank goes over RoCE (the same answer on every rank).
    pub fn fits(r: *const Roce, bytes: usize) bool {
        return bytes > 0 and bytes <= r.max_bytes and bytes % pack == 0;
    }

    /// RoceGather.all_gather: `bytes` from `send` to every rank's slot of `recv` in rank order; both 16-byte aligned.
    pub fn gather(r: *Roce, s: cuda.Stream, send: u64, recv: u64, bytes: usize) !void {
        const packs = bytes / pack;
        const need = @max(1, (packs + 2 * threads - 1) / (2 * threads));
        const grid = @min(std.math.ceilPowerOfTwo(usize, need) catch unreachable, blocks);
        const klass: usize = std.math.log2_int(usize, grid);
        const c = r.counters.ptr;
        var a: cuda.Args = .{};
        a.add(send);
        a.add(recv);
        a.add(@as(i32, @intCast(packs)));
        a.add(@as(u32, @intCast(bytes)));
        a.add(@as(i32, @intCast(packs)));
        a.add(r.recv);
        a.add(r.flag);
        a.add(r.send);
        a.add(r.ctrl);
        a.add(@as(i64, @intCast(r.slot_bytes)));
        a.add(c);
        a.add(c + 4 * (1 + klass));
        a.add(c + 4 * (1 + classes + klass));
        a.add(c + 4 * (1 + 2 * classes));
        a.add(r.spin);
        a.add(@as(i32, world));
        a.add(@as(i32, @intCast(r.rank)));
        a.add(@as(i32, @intCast(r.slots)));
        a.add(@as(i32, @intCast(r.flag_stride)));
        a.add(@as(i32, @intCast(r.n_hca)));
        try cuda.launch.launch(r.gather_fn, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = threads } }, s, &a);
    }

    /// Whether a wait timed out on the GPU or the proxy thread failed.
    pub fn check(r: *const Roce) !void {
        if (r.ctrl_words[2] != 0 or (r.ctx != null and roce_failed(r.ctx) != 0)) {
            std.log.err("RoCE all-gather failed (sequence {d}, peer {d}, hca {d}): {s}", .{ r.ctrl_words[2], r.ctrl_words[3], r.ctrl_words[6], if (r.ctx != null) std.mem.span(roce_error(r.ctx)) else "" });
            return error.RoceFailed;
        }
    }
};
