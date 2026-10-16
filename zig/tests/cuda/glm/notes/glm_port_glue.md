# glm_port_glue.cu notes

## scope

GLM's torch glue between kernels, as small exact kernels (ours): what the Python engine does with torch ops
between its Triton and CUDA launches (v0.6.6 families/glm5_next/cuda): copies, the sparse token lists and the
greedy pick. Every result is a copy or an integer, so bits cannot differ from torch's.

`tf_glm_copy_rows`: rows of `bytes` bytes (a multiple of 4) from src (row stride src_ld bytes) to dst (dst_ld):
forward.py:407's `b.ey[:R, top_k].copy_(b.sy[:R])`. grid (rows), block 256.

## tokens

`tf_glm_tokens`, sparse.py:204-215: each row's selected token list and count from its 512 best pools (int64,
ascending). q = pos + r; npool = (q + 1) / 4; tokens[r, 4i + j] = pools[r, i] * 4 + j; then the 3 tail tokens past
the last whole pool if visible (else -1); count = npool > 512 ? 2048 + visible tails : 0. grid (rows), block 256.

`tf_glm_greedy`, decode.py:34-49 for greedy rows on one rank: torch.topk(logits.float(), 1) gives the row's
maximum and its first column; out[r] = (value as fp32 bits, column + offset) for the all-gather. grid (rows),
block 1024.

## topk

decode.py:31-37 for sampled rows on one rank: torch.topk(logits.float(), k) of the rank's vocabulary slice, as
(value as fp32 bits, column + offset) pairs; which pairs, not their order, matters (the host chooser sorts them).
The k-th largest bf16 key is found in two 8-bit radix passes; columns above it are all taken, columns equal to it
in column order until k. grid (rows), block 1024, k <= 1024.

## ties

Ties at the threshold, lowest columns first: a tile of blockDim.x columns at a time, each tie ranked in column order
by a warp ballot and the warps' counts (the same pairs as a serial scan from column 0, without one thread walking
the row).

## quantize4

qmm.quantize4 (v0.6.6 glm5_next/cuda/qmm.py:158): bf16 (n, k) -> MLX affine 4-bit groups of 64, one thread a group.
scale = bf16(max((hi - lo) / 15, 1e-8)), bias = bf16(lo), q = round-half-even((w - bias) / scale) in 0..15, input
8 i + j of a group at bits 4 j of its word i; fp32 arithmetic as torch's. torch's CUDA tensor / Python scalar
multiplies by the fp32 reciprocal (BinaryDivTrueKernel.cu), tensor / tensor divides correctly rounded.

## group-error

The squared error of one group of 64 under scale s and bias b, summed in torch's order for `(...).sum(-1)` over 64
contiguous fp32 values (Reduce.cuh: 32 lanes, lane t adds x[t] and x[t + 32], then shfl_down sums at offsets
16, 8, 4, 2, 1; checked bit for bit against torch 2.13 on this checkpoint's groups): every op rounds, as torch's
separate kernels do (q * s, + b, - g, squared).

## quantize4-mse

qmm.quantize4_mse_raw (v1.8 glm5_next/cuda/qmm.py:179), TF_GLM_DENSE=q4: groups of 64 as quantize4, each group's
range the one of 16 clippings a = 1 - 0.45 i / 15 of its min and max with the least squared error (the first on a
tie), scale and bias stored as bf16. a is a Python float, cast to fp32 where torch multiplies the fp32 tensor by it.
MLX layout out, as tf_glm_quantize4.

## make-f8

qmm.make_f8 (v1.8 qmm.py:412): bf16 (n, k) -> e4m3 (n, k) and an fp32 scale per row and 128 columns, one thread a
block: scale = max(absmax * (1 / 448), 1e-30) (torch multiplies by the reciprocal of a Python scalar), value =
e4m3(clamp(w / scale, -448, 448)), round to nearest even.

`tf_glm_df_scatter`: `self.kc[i].index_copy_(1, idx, k)`: rows [kvh, rows, hd] into a cache [kvh, cap, hd] at the
positions pos + r (modulo the ring). grid (rows, kvh), block 128.

## df-scatter-seg

dflash2_multi's `self.kc[i].index_copy_(1, idx, k)`: the rows [kvh, rows, hd] of blocks of n rows side by side,
block s into its slot of the pooled caches (`cache` is slot 0; slot[s] counts rows of cap, the kv heads cap apart)
at the positions pos[s] + r (modulo the ring). grid (rows, kvh), block 128.

## df-scatter-rows

dflash2_multi `_taps_compute`'s `index_copy_(1, idx, k)`: rows [kvh, rows, hd] into the pooled caches (`cache` is
slot 0), row r at dest[r] (its slot's first row plus its ring row), the kv heads cap apart. grid (rows, kvh),
block 128.

## df-proj

`F.linear(h, hproj).float()`: h [rows, k] times w [n, k] transposed, fp32 sums, rounded to bf16 and back. One warp
an output. grid (ceil(n / 8), rows), block 256.

## nucleus

The top_p nucleus draw's device side (cuda/sampling.py nucleus_rows, TENSORFOLD_NUCLEUS_UNION): per row, the mass
floor(exp(v - top) 2^40) of every token in fixed point (v = logit * (1 / max(T, 1e-6)) in float64 as torch's
tensor / scalar computes it, top the maximum over the ranks' row maxima, which `all` holds as tf_glm_greedy wrote
them, gathered: [world][rows][value bits, id]), the shard's sum, and the top `count` candidates' (value, id, mass).
No fused multiply-add (the module's flags), the same double exp as torch's CUDA kernel.

## nuc-pack

`tf_glm_nuc_pack`, grid (ceil(count / 256), rows), block 256: out[r] = [count value bits (float64)][count ids]
[count masses][sum][width] as 64-bit words, from tf_glm_topk's (value bits, id) pairs `cand` [rows][count][2].
