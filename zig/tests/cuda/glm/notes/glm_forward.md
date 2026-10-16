# glm_forward notes

The full comments moved out of `zig/src/families/glm/cuda/glm_forward.zig` and its pieces
(`glm_forward_{buffers,lanes,blocks,prompt,shared,sample}.zig`) by the lean rule. Each heading is the anchor a
one-line comment in the code points to.

## engine

GLM-5.3-Flash's tensor-parallel forward, MTP step and commit on CUDA, launch for launch as the Python engine runs
them (v0.6.6 families/glm5_next/cuda/forward.py and mtp.py, TF_GLM_LATENT=1, the EXL3 checkpoint), eagerly: every
buffer is static, positions live on the device, and the host knows them too (Python's host_pos path).

## comm

`Comm.allGather`: small gathers (up to TF_ROCE_MAX_KB a rank, a multiple of 16 bytes) go over RoCE when it is up, the
rest through NCCL. The choice is by byte count alone, which both ranks' matching calls share.

## row-split

`SplitSettings`: TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP (v1.10 hcsplit.SplitSettings; the recipe's default SPLIT=1
sets 1 and 2, as here). A prompt chunk's hyper-connection glue is split by rows between the ranks (each rank sends the
other its partial of the other's rows and glues its own), the swaps run on a second stream in row pieces (overlap), and
the next block's row-independent front runs on own rows while the swaps run (fronts). Every glue kernel is
row-independent and the partials are summed rank 0 first as unsplit, so every bit is the unsplit path's. Swaps are p2p
only (no TF_GLM_HC_EXCHANGE=gather).

`Split`: one prompt buffer's row split (hcsplit.HcSplit, two ranks, p2p): rank r owns rows [r H, r H + H), H = Rp / 2
(an odd chunk gets one zero pad row, never read); the second stream and its events when overlapping.

`Fill`: what a split site writes into b.part for rows lo .. hi (forward.partials' `fill`): an output projection's
rows, or the MoE's shared expert on the rows its front has not run on, then the combine.

`splitGlue` (hcsplit.glue): hc_post of this rank's rows (both partials summed rank 0 first), then the layer's DFlash2
taps, the final stream mean (`final`) and the next hc_pre (`hc`), their rows swapped with the peer's; `front` runs on
own rows of each piece while its swap is on the wire.

`splitLayers` (forward.split_layers): every layer of a prompt chunk with its glue split by rows; the first hc_pre runs
on every row as unsplit, and b.normed, the taps and b.hidden come back whole from the peer.

`swapPartial`: b.gath receives the peer's partial; it is unused while the chunk runs split.

## switches

- `exl3_v18`, TF_GLM_EXL3_DEC (default on): v1.8's routed-expert kernels, plan and direct shared-expert combine; 0:
  the generic grouped kernel of v0.6.6 (the A/B path).
- `kv_fp8`, TF_GLM_KV (default fp8, v1.8's production setting; bf16 the earlier format): the DSA latent cache and the
  indexer's pooled keys as e4m3 rows with a power-of-two scale each (kv8.py).
- `l2pf`, TF_GLM_L2PF (default on): decode windows of up to TF_GLM_L2PF_ROWS rows prefetch the next kernels' weights
  into L2 from a side stream (glm_l2pf.zig); `l2pf_active` is set while such a window runs.
- `kda_chunked`, TF_GLM_KDA_CHUNKED (default on, v1.8's production setting): prompt chunks run KDA in chunked (WY)
  form, which needs prompt chunks to start on a grid of 32 (`kdaChunked`, the server's `promptGrid`); 0: the serial
  chain.
- `index_ring`: the indexer's key rings hold the whole context here, so no row wraps; the ring length only picks the
  kernels' 16-divisibility specialization, as Python's does.
- `shared_rows`: a shared round's rows across lanes (v1.10's TF_GLM_MULTI_WINDOW, 32): the window buffers and the
  pool's KDA scratch hold that many; each lane's own part stays within max_rows.

## wide-graph

TF_GLM_WIDE_GRAPHS (v1.8 production: 16; 0: off): the one verify-window width past 8 rows that replays a CUDA graph
(copy drafts pad their windows up to it, glm_copy.zig); the others of 9..15 rows run eagerly. `wideGraph` takes a
width from 9 to max_rows (default 16, v1.8's production setting; 0 or empty: none). Python takes a list of widths; one
is all the copy drafts' padding uses.

## graph-count

`max_graphs`: windows of 1..8 and `wide_graph` rows x 2 KDA parities and MTP steps of 1..8 rows, each dense or at one
of ~10 pool buckets (9 widths x 2 x 11 + 8 x 11 = 286), for each lane (graphs bake in its state's addresses).

`graphKey`: a window wholly below the dense limit, or wholly past it at one pool bucket, replays one graph per key; one
across the limit runs eagerly (decode.py's same rule). Positions live on the device, so a graph serves every position
of its key.

## pieces

`Piece`: a lane's rows of a multi-prompt chunk: its next `rows` prompt rows from its position, at row `lo` of the
prompt buffers (a multiple of 64, so each row keeps the place in the prompt kernels' 64-row tiles a chunk of its own
gives it).

`mixPieces` (multi_prefill.PieceMixer): the block's fronts and output projection over every row; its state work piece
by piece, on the piece's lane at its position, through the calls a chunk of its own makes (the piece's rows a view at
`lo`).

`pieceView`: the prompt buffers from row `lo` on, for kdaState and dsaState: every buffer they index by row moves; their
scratch (attention partials, selection, block positions) stays, a piece's work running before the next one's.

`prefillPieces` (multi_prefill.prefill_pieces up to the commits): one prompt chunk over every piece (`ids`: the chunk's
rows, a piece's tokens at its `lo`, the rows between pieces any token), then per piece the head on its last row and
that row's hidden state into its lane's last_logits and last_hidden, as prefillChunk leaves them. The taps and each
lane's commitChunk are the caller's (Node.chunkLanes).

## prompt-grid

`promptGrid` (engine.prompt_grid): the rows a prompt chunk and a kept prompt state start on. Chunked KDA depends on the
alignment of its 32-row sub-chunks, so under TF_GLM_KDA_CHUNKED the grid is 64 (a multiple of 32 that divides the
2,048-row chunks).

## pool-layout

`Pool`: every lane's KDA states and DSA layers' caches in one allocation, laid out as v1.10's slots and arenas
(multi.py's slots, segments.py) so the segmented kernels reach any lane from one base: KDA rec [slots, 2, layers, LL,
128, 128] f32 and conv [slots, layers, 3, 3 LL 128] bf16; a DSA layer's latents [slots x extent_rows, row], index keys
and gates [slots x extent_rows, 128] bf16 (no rings: token t of slot s at s x extent_rows + t, segments.SegRows' ibase =
base), pooled keys [slots x extent_rows / 4 + 2, row]. Each lane's State points into its slot (Pool.view), so its own
calls run as before. Also the segmented window's KDA scratch (one layer set of `max_rows` rows) and tables.

`extentRows`: a slot's tokens in the arenas for lanes of `capacity`: capacity rounded up to `extent`, with room for the
pooled keys' two rows past capacity / 4 (a slot's pooled keys must end before the next slot's begin).

## lanes

`laneBytes`: one more lane's device bytes once pooled: its own part and its share of the pool (the pool's segmented
scratch and tables come with the first lane).

`addLanes`: lanes up to `total` (at most max_lanes), all zeroed, their KDA states and DSA caches in one Pool; returns
how many there are now. Called once, before serving (lane 0's state is dropped: a load's timings left nothing to keep).
Lane 0's own buffer is freed first, so the pool takes its room; a pool that does not fit is tried with fewer lanes.

`copyLaneRows`: lane `src`'s attention-cache rows below `rows` (every DSA layer's latents, the MTP layer's, and with a
long context the indexer's keys and pooled keys) into lane `dst`: a kept prompt state then resumes in `dst`.

## kept-states

`snapshotBytes`: a kept prompt state's bytes: every KDA layer's recurrent state (the half in use) and conv window. The
attention caches (DSA, indexer, MTP) are not copied: they are positional, and the caller only restores a state while
their rows below its position still hold its tokens.

`commitChunk` (forward.commit(R, R) after a prompt chunk on pbuf): Python flips each KDA layer's state half as the
layer runs (st.cur[li]); every layer flips once a chunk, so one flip for all after the chunk reads and writes the same
halves.

## matmul

`mm` (forward.mm): x rows (stride in elements) times q.T into out (fp32 with `wide`). BF16 and FP8 weights ignore the
group sums `xs` (qmm.matmul); a 4-bit q takes the prompt matmul in a prompt chunk, else the decode matmul on `xs`.

`kdaBlock`: f_b and g_b read these group sums when 4-bit; BF16 weights ignore them (qmm.matmul), but Python still
launches _group_sums.

## dense-rows

`dsaState`: rows past the dense limit are recomputed sparsely below, so the dense pass needs only the rows before it (a
row's dense bits do not depend on the others), with the window's head block (forward.dense_rows).

## moe-order

`moeBlock`, TF_GLM_L2PF: the shared expert runs before the routed experts (disjoint buffers, the same kernels: the
same bits), while the L2 still holds what site "a" prefetched for it.

`moeBlock`, forward.py moe_block: the shared row is read from sy where it is (a prompt chunk always, a decode window
unless TF_GLM_COMBINE_SY=0), else copied into the routed rows' last slot.

## shared-windows

`forwardShared`: one forward over several lanes' windows back to back (`tokens`: each segment's rows in turn), logits
[R, V/2] in buf.logits: the row-independent parts on the whole window, each lane's KDA and DSA on its own state with
the kernels its window alone would take (its row count). Eager; each lane then commits on its own (bind, commit).

`kdaShared`: kdaBlock over a shared window: the projections on every row, then each lane's chain on its state; its rows
of the projection go to its `proj` (its commit's conv shift reads them there) and the chain's saves to its `scr`.

`dsaShared`: dsaBlock over a shared window: the projections, norms and the latent absorb on every row; each lane's
latent and indexer writes, token selection and attention on its own caches, positions and row count.

## segmented-windows

Segmented shared windows (v1.10 verify.BatchedVerify): every lane's rows back to back through the segmented kernels
(KDA chain_segments, latent.seg_*, sparse.seg_*). Each row's position, arena base and sparseness and each segment's
slot and parity live in the pool's device tables, so one CUDA graph a row count serves every mix of lanes and
positions. Each segment's rows get the bits of its lane's window alone (the kernels' contract); the commit replays and
shifts one lane's segment at a time (commitSeg).

`segSelBuffers`: the segmented selection's buffers (segments.SelectScratch): scores [shared_rows, seg_pools], tokens,
counts and the split selection's scratch, allocated at the first segmented window of a long-context engine (outside any
capture).

`forwardSeg`: TF_GLM_SEG is on by default once lanes share a pool.

`kdaSeg` (forward.kda_segments): the projections on every row (into the pool's [layers, max_rows, width] rows the
commit's conv shift reads), then kda.chain_segments, each segment from its slot's state into its other parity.

`dsaSeg` (forward.dsa_segments): the front on every row, each row's latent and index key into its lane's arena extent,
the segmented selection (every row: dense rows ignore it, as v1.10's graphs run it), absorb, segmented attention,
expand.

`commitSeg` (verify.BatchedVerify.commit for the bound lane's segment of the last segmented window): its KDA states
replayed to `keep` rows (when it keeps fewer), its conv window shifted, its parity flipped, its position advanced.

`commitSegs`: commitSeg for several lanes of the last segmented window at once (verify.BatchedVerify.commit over its
segments): one table, one replay of the segments that keep fewer rows than they ran, one conv shift; then each lane's
parity and position. Every lane's last window must be segmented (Node.keepLanes commits the others on their own).

## sampling

`mtp` (mtp_stage + mtp_compute): the head on rows (next token, hidden), cache slots mtp_len.. onward; the last row's
draft-head logits in b.logits. The caller advances mtp_len.

`draftHead`: the MTP head's logits from the 4-bit draft head (mtp.py: `w.draft_head`, forward.mm): a prompt chunk's on
the prefill matmul, a decode step's on qmm_group with the rows' group sums (b.fxs, from the final norm).

`nucleusGather` (sampling.nucleus_rows' `_shares` for `rows` rows of `logits`, this rank's half of the vocabulary,
bf16): every rank's top `count` (value, id, mass) per row and its shard's mass total and width, as 64-bit words
[rank][row][count float64 values, count ids, count masses, sum, width], gathered on every rank. The row maxima come
from `greedy`'s gather. Its device scratch is laid out as [sums: max_rows words][candidate pairs: max_rows * 2 *
max_nucleus u32][packed: max_rows * per_row words][gathered].

`greedy` (sample_rows, greedy): each rank's row maximum (first column on ties), gathered; the larger value wins, the
smaller token on a tie (np.lexsort((tokens, -values))).
