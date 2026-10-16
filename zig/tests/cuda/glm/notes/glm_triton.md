# glm_triton.zig and glm_triton_seg.zig

## segmented-windows

`glm_triton_seg.zig`:

-- segmented windows (v1.10 segments.py, latent.seg_*, sparse.seg_*): several lanes' rows back to back, each row's
position and arena base in the SegRows tables (glm_segments.zig), so one CUDA graph a row count serves every mix.

## seg-pools

`glm_triton_seg.zig`:

Scores a segmented selection's rows hold (segments.SelectScratch over a pool of up to 4,194,304 tokens: the
split selection's CHS 4096 / CP 256, the shapes v1.10's captured graphs launch).

## seg-select

`glm_triton_seg.zig`:

sparse.seg_select_tokens: every segment's pools up to its device-side count scored by `_seg_scores` (`segs` programs,
one a table segment), then the split selection over each sparse row's visible pools into tokens / counts.

## seg-attention

`glm_triton_seg.zig`:

latent.seg_attention: dense rows over their stream's keys 0 .. pos, sparse rows over their selected tokens, each in its
own arena extent; seg_chunks() = 5 chunk programs a row, `_merge` over them (a row's bits are its lone window's).

## header

`glm_triton.zig`:

GLM-5.3-Flash's Triton kernels from the captured cubins (aot/c2): each launch has the Python 0.6.6 wrapper's grid,
argument names and types, and constexprs (py: src/tensorfold/families/glm5_next/cuda/). num_warps and num_stages
are baked into the cubins; `aot.Set.run` picks the specialization (ints equal to 1, 16-divisibility) itself.

## score-split

`glm_triton.zig`:

sparse.py: decode rows' `_scores_rows` programs a row (ROW_SCORE_G), prompt rows a `_scores` program takes (TF_GLM_SCORE_RB,
default 4), and where the decode selection splits a row's scores across programs (SPLIT_FROM scores, under SPLIT_ROWS rows).

## split-blk

`glm_triton.zig`:

sparse.SPLIT_BLK (v1.10 patch 0085): step 4's earlier chunks an iteration. And sparse.SCORE_LOOP (patch 0086): the pool blocks a
`_scores` program takes (one row a program; speed only).

## hc-pre

`glm_triton.zig`:

glue.hc_pre (glue.py:162): x [R, S*D] streams -> out [R, D] normed and xs, post [R, S], comb [R, S, S]; part is [R, 16, 32] fp32.
`scale` is the layer's 3 fp32 mixing scales (a pointer, the kernel's SCALE parameter).

## hc-post-pair

`glm_triton.zig`:

glue.hc_post_pair (v1.10 glue.py:265, TF_GLM_HC_SPLIT): hc_post of two ranks' fp32 partials [R, D] held apart, rank 0's
at `g0` and rank 1's `rs` elements from it (may be negative): the same kernel, sum order and bits.

## router

`glm_triton.zig`:

glue.router (glue.py:333), the sliced path: `part` holds router_ks * m * ne fp32 (Python's torch.empty), `out` [m, ne] fp32 contiguous.
Python's single-pass `_router` (d not a multiple of 512, or a strided out) was never captured, so it is an error here.

## combine

`glm_triton.zig`:

glue.combine (glue.py:406): y [R, slots, D] fp32 (the EXL3 scratch; only fp32 was captured) -> out [R, D] fp32.
The caller first copies the shared expert's sy [R, D] fp32 into slot top_k of y (forward.py:407,
`b.ey[:R, c.top_k].copy_(b.sy[:R])`, a strided copy: row r's D values to y + ((r * slots + top_k) * D) * 4).

## matmul-b16

`glm_triton.zig`:

qmm._matmul_b16 (qmm.py:245): out [m, n] (bf16, or fp32 with `wide`, Python's f32=True, contiguous) = x [m, k] (rows strided) . w [n, k].T.
When b16SplitK(n, k) > 1, `part` holds sk * m * n fp32 (Python's b.sk); otherwise it is unused (Python passes OUT as PART).

## matmul-f8

`glm_triton.zig`:

qmm._matmul_f8 (v1.8 qmm.py:417) for a window of up to 16 rows on one K slice (the head's shape): `_fmm`, BM 16 and
BLOCK_N F8_BN_DECODE columns a program, over e4m3 `w` [n, k] with its fp32 block scales `sc` [n, k / 128].
`part` is unused (Python passes OUT as PART when sk is 1).

## combine-sy

`glm_triton.zig`:

glue.combine_shared (v1.8 glue.py:514): `combine` with the shared expert's row read from sy [R, D] fp32 instead of
y's last slot; the same terms in the same slot order.

## dconv

`glm_triton.zig`:

dflash2._dconv: the two-tap grouped dynamic convolution of x [rows, d] over rows r and r - 1 (`branch` 0 the first
mix, 1 the second), plus the residual `res` when given.

## dattn

`glm_triton.zig`:

dflash2's `_dattn_kernel`: each KV head attends its queries to the window's tiles and the block's keys. `pos` is the
context's end on the device.

## dattn-seg

`glm_triton.zig`:

dflash2_multi's `_dattn_seg_kernel`: `_dattn_kernel` for blocks of `n` rows side by side (q [heads, rows, hd], out
[rows, heads hd]), block s reading its slot slot[s] of the pooled caches (kc, vc: slot 0; a slot counts rows of cap, the
kv heads `slot_rows` apart) at position pos[s].

## sparse-onepass

`glm_triton.zig`:

latent.sparse_onepass (v1.8 latent.py:709): a prompt chunk's rows with counts > 0 attend their selected tokens in one
online softmax, a program a row (all the rank's heads, ONEPASS_TILE (32, 32, 1), launch (8 warps, 1 stage, prefetch)).

## dense-attention

`glm_triton.zig`:

forward.dense_attention (forward.py:331) over latent.attention: qa [R, H, LW] causal through pos + R - 1 in `nch` 512-key
chunks, in blocks of s.part_rows rows, every block with head_block(R) of the whole window.
`pos_at[j]` is the device i32 position of block j's first row: pos_at[0] is pos itself.
TORCH-GLUE: forward.py:341 `at = pos_dev + r0` for block j >= 1 (r0 = j * part_rows): a new one-element int32 device
  tensor holding pos + r0 (int32 add, no clamp). The Zig caller fills pos_at[1..] with that value before this call;
  each address must be 16-byte aligned (the captured _dense_chunks variants have POS divisible by 16).
The caller picks nch as _dsa_latent does (forward.py:372): min(chunks_for(pos + R) or s.nch, s.nch).

## sparse-attention

`glm_triton.zig`:

latent.sparse_attention (latent.py:335): rows with counts > 0 attend their selected tokens [R, W] (ascending, -1 padded)
into out [R, H, LW]; other rows are left alone. po, pm, pl are Python's torch.empty scratch: nch * R * H (* LW) fp32,
nch = cdiv(W, 512).

## index-update

`glm_triton.zig`:

sparse.index_update (v1.8 sparse.py:76): k_raw [R, 128] (rows strided, the first 128 columns of ikr) and gate [R, 128] fp32
into the index rings ik, ig at pos.. (token t at row t % ring; `ring` is at least the context here: no row wraps), then
every pool the window completes into pk (bf16, or kv8 rows under TF_GLM_KV=fp8).

## select-decode

`glm_triton.zig`:

sparse.select_tokens for decode rows (v1.8 sparse.py:323; also a prompt chunk of under 64 rows): `_scores_rows` over the
`np` pools of the graph's bucket into `scores` [rows, np], then the K best pools and each row's tokens and count:
`_select_split` (five launches, np >= 8192; `split` is its scratch) writes them itself, else `_select_rows` (visible
pools only) into `pools` and `_tokens`. qi [rows, H * D] contiguous; w points at ikr's gate columns (row stride w_stride).

## select-split

`glm_triton.zig`:

sparse.select_split: the K best pools of `rows` rows of `np` scores across chunk programs, five launches over `split`
(zeroed totals first). `toks`: rows at pos, pos + 1, .. through `_tokens`' formulas into tokens / counts (VIS), else the
pool ids into `pools` [rows, K] int64.

## select-prompt

`glm_triton.zig`:

sparse._select_prompt (v1.8 sparse.py:358) for a prompt chunk of 64 rows or more at host position `host_pos`: its rows in
blocks of select_rows, each scored over `pool_count` pools by `_scores` (score_rows() = 4 rows a program) into `scores`,
then the best pools (`_select_split` for a block of under 64 rows at 8192 pools or more, else `_select_rows`) and
`_tokens`. `pos_at[j]` is the device i32 position of block j's first row (pos_at[0] is pos itself).

## shift-conv

`glm_triton.zig`:

forward._shift_conv (forward.py:525): conv [L, 3, C] (in place) takes rows keep .. keep + 2 of [conv; proj rows]
(proj [L, R, >= C]); strides in elements as Python's conv.stride(0), proj.stride(0), proj.stride(1).
