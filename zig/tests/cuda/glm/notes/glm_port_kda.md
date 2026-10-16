# glm_port_kda.cu notes

`zig/kernels/cuda/glm_port_kda.cu` is the device code of src/tensorfold/families/glm5_next/cuda/kda.cu [v1.8]
(lines 1-529, ATen includes dropped), checked by zig/tests/cuda/copies.py; v1.8 and v1.10 hold the same file.

## design

GLM-5.3-Flash's KDA (Kimi delta attention) on CUDA: one block of 1024 threads per head, a chain of R rows.

Per row, following the Hugging Face definition: the depthwise conv over [conv state; q|k|v rows] (4 taps,
fp32) with SiLU and one bf16 rounding; fp32 L2 norms of q and k (eps inside the sum, q times DK^-0.5); the
per-channel decay g_i = exp(lower * sigmoid(exp(A_log) * (a_i + dt_bias_i))) in fp32; beta = bf16(sigmoid(b));
the delta rule in fp32 (decay along the key channel, read with k, correct toward v, read out with q);
read-out rounded to bf16; then the gated RMSNorm bf16(w * (y * rsqrt(mean(y^2) + eps)) * sigmoid(gate)).
Warp w owns value rows 4w .. 4w + 3, lane l the key columns 4l .. 4l + 3. The state update is one routine
(``update``) shared with ``replay``, compiled without FMA contraction, so a replayed prefix of a window gives
the bits of the serial steps.

## segments

Several streams' windows in one launch. A segment table (int32 [nseg, SEG_COLS]: first row, rows, state slot,
parity, conv slot, keep) cuts the window's rows into streams; each segment reads its conv taps from its own
conv slot and runs the recurrence from rec[slot, parity] into rec[slot, 1 - parity]. Per row the arithmetic is
the solo kernels' line for line (same ``update``), so a segment's bits equal a solo run of its rows. A segment
of 0 rows is skipped: a table sized for the most streams serves fewer (grids stay fixed for CUDA graphs).

## replay-seg

replay_layers_kernel for block (segment, layer, head): the segment's first ``keep`` rows from rec[slot, parity]
into rec[slot, 1 - parity]; keep == rows (the chain already left that state) or rows == 0 skips the segment.
