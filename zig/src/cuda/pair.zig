//! Two ranks' TCP link: rank 0 listens and rank 1 connects. It carries NCCL's unique id and the head's commands.

const std = @import("std");
const posix = std.posix;
const Threaded = std.Io.Threaded;

pub const Error = error{ SocketFailed, BindFailed, ListenFailed, AcceptFailed, ConnectFailed, BadAddress, PeerClosed, IoFailed };

pub const Pair = struct {
    fd: posix.socket_t,

    /// Rank 0: listen on `port` (all interfaces) and take the one peer.
    pub fn listen(port: u16) Error!Pair {
        const address = std.Io.net.IpAddress.parse("0.0.0.0", port) catch return error.BadAddress;
        var storage: Threaded.PosixAddress = undefined;
        const len = Threaded.addressToPosix(&address, &storage);
        const l = try socket();
        defer _ = posix.system.close(l);
        const one: c_int = 1;
        posix.setsockopt(l, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one)) catch return error.SocketFailed;
        if (posix.errno(posix.system.bind(l, &storage.any, len)) != .SUCCESS) return error.BindFailed;
        if (posix.errno(posix.system.listen(l, 1)) != .SUCCESS) return error.ListenFailed;
        while (true) {
            const rc = posix.system.accept(l, null, null);
            switch (posix.errno(rc)) {
                .SUCCESS => return nodelay(@intCast(rc)),
                .INTR => continue,
                else => return error.AcceptFailed,
            }
        }
    }

    /// Rank 1: connect to rank 0 at `host:port`, retrying for up to `wait_s` seconds (rank 0 may start later).
    pub fn connect(host: []const u8, port: u16, wait_s: u32) Error!Pair {
        const address = std.Io.net.IpAddress.parse(host, port) catch return error.BadAddress;
        var storage: Threaded.PosixAddress = undefined;
        const len = Threaded.addressToPosix(&address, &storage);
        var tries: u32 = 0;
        while (true) : (tries += 1) {
            const fd = try socket();
            if (posix.errno(posix.system.connect(fd, &storage.any, len)) == .SUCCESS) return nodelay(fd);
            _ = posix.system.close(fd);
            if (tries >= wait_s * 10) return error.ConnectFailed;
            var ts: posix.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
            _ = posix.system.nanosleep(&ts, null);
        }
    }

    pub fn close(p: *Pair) void {
        _ = posix.system.close(p.fd);
    }

    pub fn send(p: Pair, bytes: []const u8) Error!void {
        var off: usize = 0;
        while (off < bytes.len) {
            const rc = posix.system.write(p.fd, bytes[off..].ptr, bytes.len - off);
            switch (posix.errno(rc)) {
                .SUCCESS => off += @intCast(rc),
                .INTR => {},
                else => return error.IoFailed,
            }
        }
    }

    pub fn recv(p: Pair, out: []u8) Error!void {
        var off: usize = 0;
        while (off < out.len) {
            const rc = posix.system.read(p.fd, out[off..].ptr, out.len - off);
            switch (posix.errno(rc)) {
                .SUCCESS => {
                    if (rc == 0) return error.PeerClosed;
                    off += @intCast(rc);
                },
                .INTR => {},
                else => return error.IoFailed,
            }
        }
    }

    fn socket() Error!posix.socket_t {
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        return @intCast(rc);
    }

    fn nodelay(fd: posix.socket_t) Error!Pair {
        const one: c_int = 1;
        posix.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.NODELAY, std.mem.asBytes(&one)) catch return error.SocketFailed;
        return .{ .fd = fd };
    }
};
