# glm_kda_chunk.cu notes

The helpers are in `zig/kernels/cuda/glm_kda_chunk.cu` (Python lines 1-225), the kernel in
`glm_kda_chunk_kernel.cuh` (Python lines 226-710), included at the end of `namespace tf_kda_chunk`. Both carry a
`// Device code of ...` header, so `zig/tests/cuda/copies.py` checks each against its line range.

## design

GLM-5.3-Flash's KDA prompt chunks in chunked (WY) form: one block of 256 threads per head runs the prompt
chunk's 32-row sub-chunks in order (sub-chunks at absolute positions), the 128 x 128 state in registers.

Per row the inputs are kda.cu's (this file is built like kda.cu, without FMA contraction, with CUDA's expf and
IEEE division): conv + SiLU rounded to bf16, fp32 L2 norms (q times 128^-0.5), beta = bf16(sigmoid(b)), the
per-channel decay g = expf(lower * sigmoid(exp(A_log) * (a + dt_bias))). A sub-chunk of rows 0..31 entering state
S0 [value, key]:

```
    E(t -> s)  = g_{t+1} ... g_s per key channel: a running product multiplied in row order, as the serial chain
                 decays its state (empty: 1); E(s) = g_0 ... g_s from the sub-chunk's start;
                 E(s ->) = g_{s+1} ... g_31
    L[s, t]    = beta_s sum_c k_s k_t E(t -> s)                  t < s
    M[s, t]    =        sum_c q_s k_t E(t -> s)                  t <= s
    T          = (I + L)^-1                                     forward substitution (16-row blocks)
    Y          = beta (v - (k E(s)) S0^T),   V' = T Y             the delta rule's corrections ("v_new")
    O          = (q E(s)) S0^T + M V'                            read-outs, rounded to bf16, then the gated RMSNorm
    S_end      = S0 diag(E(31)) + V'^T (k E(s ->))
```

Every factor is a product of decays <= 1 (no e^(+x), no difference of cumulative log-decays), so a decay carries
the serial chain's rounding; all sums run in fp32 with explicit FMAs and shuffle trees in a fixed order: the kernel
is deterministic, and a row's bits depend only on its sub-chunk's rows and the state entering it, so they are the
same whichever prefill chunk the row came in, as long as prompt chunks start on multiples of 32
(TF_GLM_PROMPT_GRID). Rows outside the prompt chunk (a short first or last sub-chunk) are zero rows: decay 1, no
key, no value, beta 0.

Layout. The state: thread (vg, kp) = (tid / 8, tid % 8) holds value rows 4 vg .. 4 vg + 3 and key quads kp + 8 i
(i < 4): an 8-lane group reads 128 contiguous bytes of a key row per load, and products with S reduce over the 8
lanes by a transposing shuffle tree (7 shuffles for 8 sums). Read-outs and corrections: thread (vg, kp) finishes
value column 4 vg + kp % 4 of the rows of one parity (odd for kp < 4, even otherwise), 16 rows. The next
sub-chunk's projection, decay and beta rows are loaded into registers while this one computes, and staged into
shared memory that is free at the sub-chunk's start.

## div-rn

a / b with IEEE rounding, branch-free: the fast path of CUDA's division (a refined reciprocal, then one
correction), which is the correctly rounded quotient whenever ``div_ok`` holds (checked on 742 M operand pairs
against a / b); callers take a / b for the rare operands outside (a huge denominator, a tiny numerator).

`tree_level`: one level of a transposing shuffle tree: this lane keeps the half of a[0..N) its ``hi`` bit selects,
adds the partner's (lane ^ mask) copy of that half, and sends the other half.

## warm

Warm L2 with the rows of the sub-chunk whose row 0 is prompt-chunk row r0 (projection rows r0 - 3 .. r0 + 31 of
this head's q | k | v channels, its decay rows and beta logits; not when ``next`` is false) and the output gate
rows from gate_r0.

## stage

Stage the rows of the sub-chunk whose row 0 is prompt-chunk row r0 into shared memory: projection rows r0 - 3 ..
r0 + 31 (the conv state before row 0, zero rows outside), decay rows and the beta logits. Global to shared
directly (cp.async) when the rows are 16-byte aligned; the caller waits (cp.async.wait_all) and syncs.

## step-1

1\. conv + SiLU (kda.cu's arithmetic) in 4-row blocks: warp w takes rows 4 w .. 4 w + 3 of q, then of k,
then of v, lane l the channels 4 l .. 4 l + 3 (kda.cu's norm lanes), and the L2 norms of its q and k rows
(kda.cu's sums); then the decays and beta.

## step-3

3\. L and M. Group pr = tid / 16 takes column t = pr (rows s > pr), then t = 31 - pr (31 entries in all,
every group alike), in batches of 8; lane cg holds the key quads cg and cg + 16, k_t E(t -> s) multiplied
row after row; a batch's 16 sums reduce over the 16 lanes by a transposing tree.

## step-4

4\. T = (I + L)^-1 by warp 0: lane l inverts column l % 16 of diagonal block l / 16 by forward
substitution, then block (1, 0) = -T11 (L10 T00). Warps 1-7 per key channel: k E(s ->), then k E(s) in
place, and E(31) (threads 32-159); q E(s) in place (threads 160-255, then 160-191 for the last 32 channels).

## step-6

6\. load this sub-chunk's output gates (warm in L2); (q E) S0^T into k (free now); V' = T Y into q. Tiles:
warp w the rows 4 w' .. 4 w' + 3 with w' = w (w < 4) or 11 - w (the two warps of a scheduler, w and w + 4,
then run 36 steps of t together), lane l the value columns 4 l .. 4 l + 3 (T is zero above its diagonal:
t runs to the tile's last row).

Step 7: read-outs O = (q E) S0^T + M V' of the same tile, bf16, the gated RMSNorm (kda.cu's formula; a warp
holds its rows' 128 values); the state update.
