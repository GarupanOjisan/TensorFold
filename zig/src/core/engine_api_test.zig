//! The core's tests that engine_api.zig's test block runs.
test {
    _ = @import("lane_host.zig");
    _ = @import("lane_host_test.zig");
    _ = @import("prompt_cache.zig");
    _ = @import("prompt_radix.zig");
    _ = @import("prompt_imprint.zig");
    _ = @import("keepalive.zig");
    _ = @import("originals_test.zig");
}
