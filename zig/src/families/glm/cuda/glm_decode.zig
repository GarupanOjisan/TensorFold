//! Prefill, serial decoding and MTP-drafted decoding over the forward (v0.6.6 glm5_next/cuda/decode.py): both ranks
//! run the same calls, and every choice comes from the gathered greedy picks, so they agree without messages.
const std = @import("std");
const fwd = @import("glm_forward.zig");
const Engine = fwd.Engine;

const bf = 2;

/// decode.prefill without a resume point: the prompt in chunks, each also absorbed by the MTP head (its rows from
/// the chunk's second token on), then the first token. The last row's hidden stays in e.last_hidden for the first draft.
pub fn prefill(e: *Engine, prompt: []const u32, use_mtp: bool) !u32 {
    if (prompt.len == 0) return error.EmptyPrompt;
    const d: usize = e.w.cfg.hidden;
    try e.reset();
    var start: usize = 0;
    while (start < prompt.len) : (start += Engine.prefill_rows) {
        const chunk = prompt[start..@min(prompt.len, start + Engine.prefill_rows)];
        const rows = chunk.len;
        _ = try e.stage(&e.pbuf, chunk);
        try e.compute(&e.pbuf, rows);
        try e.copyDev(e.last_logits, e.pbuf.logits, e.w.head.n * bf); // `last = compute(..).clone()`
        try e.copyDev(e.last_hidden, e.pbuf.fnormed + (rows - 1) * d * bf, d * bf);
        if (use_mtp and e.w.mtp != null) {
            const nxt = prompt[start + 1 .. @min(prompt.len, start + rows + 1)];
            if (nxt.len > 0) {
                try e.mtp(&e.pbuf, nxt, e.pbuf.fnormed);
                try e.setMtpLen(e.st.mtp_len + nxt.len);
            }
        }
        try e.commitChunk(rows);
    }
    var first: [1]u32 = undefined;
    try e.greedy(&e.buf, e.last_logits, 1, e.w.head.n, &first);
    return first[0];
}

/// decode.serial_decode, greedy: one token a step until `count` tokens (the first included) or an end token.
pub fn serial(e: *Engine, pending: u32, count: usize, stop_eos: bool, out: *std.ArrayList(u32), gpa: std.mem.Allocator) !void {
    try out.append(gpa, pending);
    while (out.items.len < count and !(stop_eos and isEos(e, out.items[out.items.len - 1]))) {
        try e.forward(&.{out.items[out.items.len - 1]});
        var tok: [1]u32 = undefined;
        try e.greedy(&e.buf, e.buf.logits, 1, e.w.head.n, &tok);
        try e.commit(1, 1);
        try out.append(gpa, tok[0]);
    }
}

fn isEos(e: *const Engine, t: u32) bool {
    for (e.w.cfg.eos[0..e.w.cfg.eos_n]) |x| if (x == t) return true;
    return false;
}

/// decode.absorb: kept rows (hidden, next token) into the MTP cache, after trimming the last round's draft entries;
/// the last row's draft logits in mbuf.logits.
fn absorb(e: *Engine, hidden: u64, next: []const u32) !void {
    const d: usize = e.w.cfg.hidden;
    if (e.st.mtp_drafted > 0) {
        try e.setMtpLen(e.st.mtp_len - e.st.mtp_drafted);
        e.st.mtp_drafted = 0;
    }
    var s0: usize = 0;
    while (s0 < next.len) : (s0 += e.mbuf.rows) {
        const part = next[s0..@min(next.len, s0 + e.mbuf.rows)];
        try e.mtp(&e.mbuf, part, hidden + s0 * d * bf);
        try e.setMtpLen(e.st.mtp_len + part.len);
    }
}

/// decode.draft without a confidence cut: absorb, then chain `count` drafts, each from the head's own output row.
pub fn draft(e: *Engine, hidden: u64, next: []const u32, count: usize, out: []u32) !void {
    try absorb(e, hidden, next);
    for (0..count) |j| {
        var t: [1]u32 = undefined;
        try e.greedy(&e.mbuf, e.mbuf.logits, 1, e.w.head.n, &t);
        out[j] = t[0];
        if (j + 1 < count) {
            try e.mtp(&e.mbuf, &.{t[0]}, e.mbuf.fnormed);
            try e.setMtpLen(e.st.mtp_len + 1);
            e.st.mtp_drafted += 1;
        }
    }
}

/// decode.mtp_decode, greedy, `depth` drafts a round: verify the pending token and the drafts, keep through the
/// first mismatch, draft again from the kept rows.
pub fn mtpDecode(e: *Engine, pending: u32, count: usize, depth: usize, stop_eos: bool, out: *std.ArrayList(u32), gpa: std.mem.Allocator) !struct { rounds: usize, accepted: usize } {
    if (depth + 1 > Engine.max_rows) return error.DepthPastWindow;
    try out.append(gpa, pending);
    var drafts: [Engine.max_rows]u32 = undefined;
    var n_drafts = @min(depth, count - out.items.len);
    if (n_drafts > 0) try draft(e, e.last_hidden, &.{pending}, n_drafts, drafts[0..n_drafts]);
    var rounds: usize = 0;
    var accepted: usize = 0;
    while (out.items.len < count and !(stop_eos and isEos(e, out.items[out.items.len - 1]))) {
        var window: [Engine.max_rows]u32 = undefined;
        window[0] = out.items[out.items.len - 1];
        @memcpy(window[1 .. 1 + n_drafts], drafts[0..n_drafts]);
        const rows = 1 + n_drafts;
        try e.forward(window[0..rows]);
        var sampled: [Engine.max_rows]u32 = undefined;
        try e.greedy(&e.buf, e.buf.logits, rows, e.w.head.n, sampled[0..rows]);
        var keep: usize = 1;
        for (drafts[0..n_drafts], 0..) |dt, i| {
            if (sampled[i] != dt or (stop_eos and isEos(e, sampled[i]))) break;
            keep += 1;
        }
        try e.commit(rows, keep);
        rounds += 1;
        accepted += keep - 1;
        try out.appendSlice(gpa, sampled[0..keep]);
        if (out.items.len >= count or (stop_eos and isEos(e, out.items[out.items.len - 1]))) break;
        n_drafts = @min(depth, count - out.items.len);
        if (n_drafts > 0) try draft(e, e.buf.fnormed, sampled[0..keep], n_drafts, drafts[0..n_drafts]);
    }
    if (out.items.len > count) out.shrinkRetainingCapacity(count);
    return .{ .rounds = rounds, .accepted = accepted };
}
