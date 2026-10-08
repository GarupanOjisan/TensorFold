# GLM-5.3-Flash's CUDA kernel set and reference replies

The native GLM engine (`zig/src/families/glm/cuda`) launches the Python engine's Triton kernels from captured
cubins, as Nemotron's does (`zig/tests/cuda/nemotron/box`). The Python engine is TensorFold v0.6.6 on two DGX
Sparks, and its replies are the reference the native engine must equal token for token (`tf-glm check`).

## 1. Capture (both GPUs, about 15 minutes)

`capture.sh RUN [capture_glm.py options]` runs `capture_glm.py` on both ranks in `nvcr.io/nvidia/pytorch:26.07-py3`
(Triton 3.7.1, nvcc 13.3) against a v0.6.6 checkout, with `--tools` so rank 0 records every Triton launch. It writes
`tokens.json`, `prompts.json`, `launches.json` and `jit.json` beside the Triton cache. Set `PY066` (a v0.6.6
checkout), `MODEL` (the EXL3 snapshot), `WORKER` (rank 1's ssh target), `MASTER` (rank 0's RoCE address) and the
NCCL variables of your link (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`).

The kernel set used so far comes from three runs:

- `c2`: the prompt set (four chats, 2,918 and 4,422 tokens), 128 tokens each, serial and MTP-drafted;
- `c3`: `--max-tokens 16 --lengths 2,3,15,16,17,18,31,32,33,34,47,48,49,63,64,65,66,79,80,81,127,128,129,130,200,1024,1040,4098,4099,4112,4113,4128,4129,4144,4160,4161,4176,4177,4224,4225,4240,6161`
  (every prompt chunk remainder class the `_bmm`, router and attention specializations depend on);
- `c4`: `--max-tokens 48 --lengths 8192,32768`.

## 2. Pack (host)

```
python3 -B tools/zig/triton_aot_manifest.py --cache RUN/triton --launches RUN/launches.json --mount /cap/triton --out RUN/manifest.json
python3 -B zig/tests/cuda/nemotron/aot_pack.py --manifest c2/manifest.json --cache c2/triton \
  --manifest c3/manifest.json --cache c3/triton --manifest c4/manifest.json --cache c4/triton --jit c4/jit.json --out sm121
```

200 specializations of 26 functions. `TENSORFOLD_CUDA_KERNELS` names the folder.

## 3. Check

`tf-glm check MODEL --rank R --master HOST --prompts RUN/prompts.json --want RUN/tokens.json` on both Sparks
(rank 0 first or second): each prompt prefilled, then decoded serially and with MTP drafts (`--depth`), against the
capture's serial reply. Build with the container's nvcc so the lifted CUDA kernels compile as the Python
extensions did.

## 4. TF_GLM_DENSE=q4 (v1.8 patches 0002 and 0005)

The engine quantizes the dense projections at load, as v1.8 does by default: every `q4()` / `stack()` matrix of
`weights.py` (KDA input stack, `f_b`, `g_b`, `o_proj`; DSA `q_a`+`kv_a`, `q_b`, `o_proj`; the indexer's `wk`+`weights_proj`
and `wq_b`; the dense MLP's and the shared expert's gate/up and down; the MTP `eh_proj`) becomes 4-bit groups of 64
(`tf_glm_quantize4_mse`, `qmm.quantize4_mse`) after the two-rank split, and the head becomes e4m3 with fp32 block scales
(`tf_glm_make_f8`, `qmm.make_f8`). `TF_GLM_DENSE=b16` keeps the BF16 path (`bf16` also works). The decode matmul is
v1.8's `qmm.cu` (`qmm_kernel`, copied and checked by `copies.py` against `$TF_PYTHON_REPO_V18`), the prompt matmul
`qmm_prefill.cu`, the head `_fmm` (Triton).

Byte equality of the quantizer with v1.8: `qdump.py MODEL 0 DIR` (v1.8 image, GPU, `PYTHONPATH=py18/src`) writes the
tiled words, scales and biases of 13 tensors, the FP8 head and the 4-bit draft head; `tf-glm qdump MODEL --rank 0 --out
DIR2` writes the same files from the Zig kernels; `cmp` them. The quantizer's error sum follows torch's CUDA reduction
order (32 lanes of x[t] + x[t + 32], then shfl_down at 16, 8, 4, 2, 1), and `/ 15`, `/ 448` multiply by the fp32
reciprocal as torch does for a Python scalar.

Reference tokens: `pycap18.sh` with the production settings captures features this port lacks, so `cap18.sh RUN bf16|q4` (a reference for
the port) sets those switches to their v0.6.6 behaviour (`TF_GLM_KV=bf16`, `KDA_CHUNKED=0`, `EXL3_*=0`, `L2PF=0`,
`HC_DEC=0`, `HC_MMA=0`, `LATENT_MMA=0`, `SPARSE_ONEPASS=0`, `KDA_WIDE_ROWS=64`, `SEQ_ROWS=1000000`, `COMBINE_SY=0`,
`SCORE_RB=1`, `WIDE_GRAPHS=0`) and `TF_GLM_DENSE=q4` or `bf16`. With `bf16` those replies equal the v0.6.6 set
(`c2`) token for token. The `_fmm` kernels join `c234` with `aot_pack.py --base aot/c234 --only _fmm`.

## 5. v1.8's routed-expert kernels (TF_GLM_EXL3_DEC)

`moeBlock` runs v1.8's `exl3_mm.routed` at two ranks by default: `experts.route` (the plan kernels of `experts.cu`, items of 16
pairs, 64 for a prompt chunk), then, for a decode window, `rot_rows` (one rotated input a row where every expert shares one
gate/up `suh`, found at load), `dec_kernel` for gate/up (16-byte trellis loads one step ahead, `TF_GLM_EXL3_LOADS=nc`),
`gateup_epilogue_kernel` and `dec_kernel` for down with its epilogue fused (`FUSE=1`); for a prompt chunk `rot_rows`,
`prompt_order_kernel`, `prompt_gateup_kernel<64, 8, shx>` and `prompt_down_kernel<64, 8>`. The shared expert's row goes into
the combine where it is (`_combine_sy`, `TF_GLM_COMBINE_SY`). The device code is `glm_exl3d.cu`, a copy of v1.8's `exl3.cu` lines
1-912. `TF_GLM_EXL3_DEC=0` runs the generic grouped path of v0.6.6 (the A/B path; it also copies the shared row as before).
Not ported: the decode path's `LOADS=0`, `FUSE=0/2/3` and `PASS=128` variants, the grouped fallback's `NFIRST/SEQ/PF`, and
the shared expert before the routed ones (it matters with `TF_GLM_L2PF` only).

The reference sets `m1`, `m2`, `m3` are `q1`, `q2`, `q3` with `OFF_EXTRA="-e TF_GLM_EXL3_LOADS=nc -e TF_GLM_EXL3_DEC=1
-e TF_GLM_EXL3_FUSE=1 -e TF_GLM_EXL3_XROW=1 -e TF_GLM_EXL3_PROMPT=1 -e TF_GLM_EXL3_ORDER=1 -e TF_GLM_EXL3_ROWROT=1
-e TF_GLM_EXL3_PASS=64 -e TF_GLM_COMBINE_SY=1"` (`cap18.sh` appends it). `aot/m18` is `aot/q18` plus `_combine_sy`.

## 6. v1.8's KDA kernels (TF_GLM_KDA_CHUNKED, TF_GLM_KDA_WIDE_ROWS)

A prompt chunk runs each KDA layer through `chunk_kernel` (`glm_kda_chunk.cu`, a copy of v1.8's `kda_chunk.cu` lines 1-711):
32-row sub-chunks at absolute positions, one block of 256 threads a head, no per-row k/v/g/b saved. `TF_GLM_KDA_CHUNKED=0`
runs the serial chain as before. Its rows depend on the alignment of the sub-chunks, so under it chunk starts and kept prompt
states sit on a grid of 64 rows (`engine.prompt_grid`): the server rounds its chunk starts and the cut before the
conversation's text down to it (`Info.prefill_grid`, `chunk_plan.Plan.grid`); the 2,048-row chunks of `tf-glm check` start on it anyway.
Decode windows take the three-kernel chain (`prep`, `step`, `out`) from one row on (`TF_GLM_KDA_WIDE_ROWS`, default 1; 64 was
v0.6.6's); `glm_kda.cu` already holds those kernels, v1.8 only appended the multi-stream segment variants after line 353, and
`replay_layers` and the conv shift are unchanged for one stream.

The reference sets `k1`, `k2`, `k3` are `m1`, `m2`, `m3` with `-e TF_GLM_KDA_CHUNKED=1 -e TF_GLM_KDA_WIDE_ROWS=1`.

## 7. v1.8's DSA side (TF_GLM_KV, TF_GLM_LATENT_MMA, TF_GLM_SPARSE_ONEPASS, the decode selection)

- `TF_GLM_KV=fp8` (default; `bf16` the earlier format): the latent cache is `[rows, 512 + 16]` bytes and the indexer's pooled
  keys `[rows / 4 + 2, 128 + 16]` (e4m3 codes, an fp32 power-of-two scale after them, 12 zero bytes; kv8.py), written by
  `_lat_write` / `_pool_keys` and read by `_dense_chunks`, `_sparse_chunks`, `_sparse_onepass`, `_scores`, `_scores_rows`. The
  index keys and gates stay bf16. Prompt-cache snapshots hold only the KDA state, so they do not depend on the layout. At a
  1,048,576-token window it frees about 6.5 GiB of the Spark's memory (13.9 to 20.5 GiB available with the server up).
- Prompt chunks: `_absorb_mma` / `_expand_mma` (`TF_GLM_LATENT_MMA`), `_sparse_onepass` (`TF_GLM_SPARSE_ONEPASS`).
- Selection: decode windows (and a prompt chunk of under 64 rows) `_scores_rows`, then `_select_split` (five launches from 8,192
  scores) or `_select_rows` and `_tokens`; a prompt chunk of 64 rows or more `_select_prompt` (`_scores` four rows a program,
  blocks of 512 rows). The torch glue of v0.6.6's port is gone.
- Reference sets `v1`..`v4` (v4: one 123,312-token prompt, context 262,144) are `k1`..`k3` with `-e TF_GLM_KV=fp8
  -e TF_GLM_LATENT_MMA=1 -e TF_GLM_SPARSE_ONEPASS=1 -e TF_GLM_SCORE_RB=4`; `b1` is the six prompts with `TF_GLM_KV=bf16`. `aot/v18` is
  `aot/m18` plus the kernels of `v1`..`v4` and `b1` (`aot_pack.py --base aot/m18 --only ...`).

## 8. The rest of v1.8's single-stream production configuration

- `TF_GLM_HC_DEC` (default on): a decode window's hyper-connection mixing dots in `hc_partial_kernel` (`glm_hc.cu`, a copy of v1.8's
  `hc.cu`, 48 blocks a row, `_hc_partial`'s bits), `TF_GLM_HC_MMA` (default on): a prompt chunk's in `_hc_partial_mma`; `_hc_finish` after both.
- `TF_GLM_L2PF` (default on; `0`, `1`/`bulk`, `lines`, `touch`; `_MB`, `_SITES`, `_ROWS`): `glm_l2pf.zig` and `glm_l2pf.cu` (a copy
  of `l2pf.cu`) prefetch the next kernels' weights into L2 from a side stream at the sites a / f / o of decode windows of up
  to 64 rows, forked and joined with events so that it also runs inside the captured graphs; the MoE then runs its shared
  expert before the routed experts, as v1.8 does under it.
- Not ported, because the Zig engine does not use them: multi-stream serving (`TF_GLM_MULTI_*`, `FILL_*`, `PREFILL_OVERLAP`, `HC_SPLIT`, `HC_EXCHANGE`), the
  reply-shaping and pacing settings of the Python server (`ASSISTANT_ENDS`, `CLEAR_THINKING`, `KEEP_REASONING*`,
  `STREAM_SMOOTH*`, `SHARED_PREFIX`, `CACHE_*`, `DISPLAY_KV_*`). `LOADS=0`, `FUSE`, `PASS=128`, `Q4_TILE`, `SEQ_ROWS`, `L2PF_BLOCKS/KB/THREADS`,
  `DRAFT_QUANT=mse`, `LATENT_STAGES`, `SCORE_RB` stay at v1.8's defaults.

Acceptance: `tf-glm check` against the production-setting captures `v18a` (6 prompts), `v18b` (42 lengths, 8k, 32k) and `p4` (one
123k-token prompt), `aot/p18b` = `aot/v18` plus their kernels, at depth 3 and 7.

## 9. Copy drafts and windows of up to 16 rows (v1.8 patches 0007, 0013, 0032)

- `glm_copy.zig` is `copy_drafts.py` with the production settings (`TF_GLM_COPY_DRAFTS=1`, `_MAX=15`, `_MATCH=8`, and COPY_CODE's
  `_REPLY_MATCH=16` and `TF_GLM_WIDE_GRAPHS=16`; the same variables, `TF_GLM_COPY_DRAFTS=0` turns every copy off): when the context's last 8
  tokens (the pending one included) occurred before, the tokens after the latest earlier occurrence that has that many tokens
  after it (else the earliest) are verified in place of the MTP head's drafts; an occurrence that starts in the reply needs 16
  matching tokens; a window of 9 to 15 rows is padded to 16 by repeating its last draft when `room` allows, so it replays the
  16-row graph. `zig test zig/src/families/glm/cuda/glm_copy.zig` runs its unit tests; a generated cross-check of 1,500 random
  contexts (964 with a copy) against v1.8's `CopyDrafts.propose` gives the same drafts in every case.
- The lane core's own `SuffixLookup` (3-gram index, longest match, silent after four misses) is not v1.8's rule, so
  `glm_native.zig` puts `CopyRule` (the same rule as a `lanes.proposer`) on each stream at its prompt pass, or none with
  `TF_GLM_COPY_DRAFTS=0`. The core still takes no copy of one token (`copyProposal` wants two) and asks for the head's one draft
  behind a copy, where v1.8 only absorbs the kept rows.
- The engine: `Engine.max_rows` 16 (`mtp_rows` 8, so the MTP head absorbs a kept backlog in steps of 8 as v1.8 does), CUDA graphs
  for 1..8 and `wide_graph` (`TF_GLM_WIDE_GRAPHS`, 16) rows, eager launches for the others. Everything else was already generic in
  the rows: KDA replay up to `keep` 16, the decode matmuls (`rows <= 16`), the router and expert plans.
- `tf-glm check ... --copy 1` runs `mtpDecode` with copy drafts and prints the copied rounds and a histogram of the windows' rows.
- Reference sets (v1.8 with `-e TF_GLM_COPY_DRAFTS=1 -e TF_GLM_COPY_MAX=15 -e TF_GLM_COPY_REPLY_MATCH=16` on top of `pycap18.sh`'s
  production settings; `capture_glm.py --set copy` for the first): `cp1` seven prompts whose replies repeat the prompt (a code edit, a rename, a JSON reformat and
  edit, 3.2k and 5.2k-token module edits, a quote; 320 tokens), `cp2` = v18a's prompts, `cp3` = v18b's lengths, `cp4` = p4's
  123k-token prompt. Their serial replies equal v18a / v18b / p4's, and v1.8's own drafted replies equal its serial ones.
  `aot/p18c` is `aot/p18b` plus the kernels of `cp1`..`cp4`.
- `capture_glm.py --window-ms` times v1.8's verify windows of 1 to 16 rows (`window_ms.json`), the Python side of the server's
  `GLM window ms by width` line.

## 10. DFlash2 drafts (v1.8 dflash2.py, DRAFTER=dflash2)

- `TF_GLM_DRAFTER_DIR` (a snapshot of incoai/GLM-5.3-Flash-DFlash2, the same on both ranks) loads the draft model and leaves the MTP head
  out, as v1.8's `TF_GLM_MTP=auto` does beside DFlash2; `tf-glm check --drafter DIR` is the same for the check. `TF_GLM_DFLASH_POLICY`
  (default `fnc7:0.3`, v1.8's production rule; `fcN:P` also) cuts a chain. Unset: the MTP head drafts as before.
- `glm_weights.loadDraft`: five layers, split between the ranks as `tp.draft_parts` (16 query heads, 4 KV heads, 6,144 MLP rows each), every
  matrix 4-bit with min/max ranges at load (`qmm.draft_quantize4`), the selector's projection in BF16, the codebooks on rank 0's host
  (the chains are picked there; its tokens ride the next verify command to rank 1). About 1.8 GiB a rank.
- `glm_dflash.zig` is `Drafter`: `addTaps` (the target's stream means after layers 5, 14, 24, 33, 42 of the committed rows, `Buffers.taps`,
  filled by every forward) through `fc`, `hidden_norm`, and each layer's K and V into a ring of 2,176 rows (`TF_GLM_DRAFT_RING=0`: flat); `block`
  (`[pending, mask x 7]` through the five layers: two-tap dynamic convolutions `_dconv_kernel`, `_prep_kernel`, sliding-window `_dattn_kernel`, the
  o and down projections' fp32 partials all-gathered and added in rank order, then the draft head's top 16 a rank gathered and the selector's
  projection); `chain` (the selector's edge scores and the target's keyed noise, the noisy-confidence stop rule, float64 on the host). Both
  context updates (1..16 rows) and the block pass replay CUDA graphs. The glue torch did is five small kernels in `glm_glue.cu`
  (`tf_glm_df_*`); `tf_glm_topk` takes its ties by a ballot instead of one thread's walk along the row (the same pairs).
- Who proposes: a copy first (glm_copy.zig; a round with a copy ahead runs no block, the taps still join), else DFlash2's chain of up to 7 drafts (windows of up to 8
  rows; copies up to 16). The lane core sees DFlash2 as its head: `draftFn` gives the chain as host tokens through the backend's `tree` hook,
  and the rule's own depth is only asked to be non-zero (the first round after a prompt takes its drafts as held ones).
- Kept prompt states carry the drafter's ring and position (`Node.snapshotBytes`).
- References: `df1` = `cp1`'s prompts, `df2` = v18a's, `df4` = p4's, all from v1.8 with `TF_GLM_MTP=auto TF_GLM_DFLASH_POLICY=fnc7:0.3` and the copy settings
  (`capture_glm.py --drafter`); `aot/p18d` = `aot/p18c` plus their kernels. Their serial replies equal the earlier sets'.
- Acceptance: `tf-glm check --copy 1 --drafter DIR` against v18a, v18b, p4 and cp1 (the sets' serial replies), and the rounds equal v1.8's:
  `cp1` cedit 41 rounds, 17 copied (v1.8's DFlash2 capture: 41, 17), story 67 rounds, code 21, facts 39, json 22.

## 11. v1.10's speed patches (recipe 7d42f90, patches 0084-0097)

- References: `w1` = v18a's prompts, `w2` = the copy-heavy ones (`--set copy`), `w3` = p4's 123k-token prompt, `w4` = v18b's lengths, recorded from
  the published image (`ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:bc34d7d6...`, extracted to `~/tf-native/py110`; `capv110.sh`) at its
  defaults with one stream: DRAFTER=dflash2, COPY=1, NUCLEUS_UNION=1, DEC_ORDER=0. Their replies and round counts equal the v1.8 DFlash2 captures'
  (`df1`, `df2`), so the token references did not move. `aot/p110` = `aot/p18d` plus their `_scores` and `_select_split` (400 kernels).
- 0085, 0086: `_select_split` gets `BLK=16` (step 4's earlier chunks an iteration), `_scores` one row and `L=32` pool blocks a program, grid `(n, cdiv(cdiv(np, 64), 32))`
  (`glm_triton.zig`: `split_blk`, `score_loop`). Same bits; `tf-glm check` passes on w1..w4 (the 123k prompt of w3 exercises both).
- 0090: `dec_kernel` takes `SWZ` (a launch order: blocks of one k slice together); `TF_GLM_EXL3_DEC_ORDER` 0 (default), 1, 2 picks one of nine instantiations. A/B in one
  session, greedy, native, tok/s (0 / 1 / 2): cedit 104.9 / 106.1 / 106.0, quote 137.9 / 139.2 / 138.8, story 46.3 / 46.2 / 45.6, code 90.0 / 88.8 / 89.5,
  facts 59.5 / 59.4 / 58.6, json 87.5 / 87.7 / 86.3; a second run of order 0 differs from the first by as much (code 87.4). No gain, so the default stays 0, as v1.10's.
- 0092: a block pass's candidate ids at or past the vocabulary are a copy read before it had landed: the rows are copied again once the device is idle, a second failure is an error.
- Sampled draws over the top_p nucleus (top_k 0; patch 0034, `TENSORFOLD_NUCLEUS_UNION`, default on): `Engine.nucleusGather` runs the row maxima's gather, `tf_glm_nuc_stats`
  (the shard's mass total, an integer), `tf_glm_topk` for the first `TF_GLM_NUCLEUS` (1024) candidates a rank and `tf_glm_nuc_pack` (value, id and mass floor(exp(v - top) 2^40)
  in float64 with v = logit * (1 / T) as torch's tensor / scalar computes it), then one gather; `Glm.drawNucleus` is `_draw` with the union test (the merged walk to
  ceil(top_p total) stops above every rank's last candidate), then 16,384 candidates a rank, then the whole shards (`fullRows` and the host `nucleus`, as before).
  Without the union rule (0) a row whose nucleus is not inside one rank's candidates goes straight to the shards. Both give v1.10's draws.
  The server's default top_k differs: v1.10 uses 20 (its `Sampling` default; the checkpoint's generation_config has no top_k), the native server 0. Run it with `--top-k 20`
  to draw as v1.10 does by default (the candidates path); a request with `top_k: 0` takes the union path.
- Sampled replies (temperature 1, top_p 0.95, seed 11) of the 13 benchmark prompts equal v1.10's, with top_k 20 (default) and with `top_k: 0`, and with the union rule off.
  Native, top_k 0, tok/s: union on / off: story 42.4 / 33.0, code 59.9 / 45.7, facts 48.0 / 37.0, cenc 125.3 / 82.8; v1.10: 42.9, 59.2, 48.9, 120.8.

## 12. RoCE all-gathers (v1.8 patch 0006, ABI 5 of v1.10; TF_GLM_COMM, TF_ROCE_MAX_KB)

- `glm_roce.zig` + `glm_roce.cu` + `roce_proxy.c`: a gather of up to `TF_ROCE_MAX_KB` (512) KiB a rank whose size is a multiple of 16 bytes is one kernel that writes the rank's shard into
  pinned host memory, which the ConnectX-7 RDMA-writes to the peer (the proxy thread of b12x's "RoCEnante", `roce_proxy.c` as v1.10 ships it, Apache-2.0), then spins on
  the peer's flags and copies the shards out in rank order; larger gathers go through NCCL. The device-side epoch lets CUDA graphs replay it. Same layout, same bits.
- libibverbs is opened by dlopen: `roce_dl.c` defines the 17 `ibv_*` entry points the proxy calls and resolves each from `libibverbs.so.1` at its first call, so the
  binaries link nothing new (`ldd` shows libc only) and a machine without rdma-core runs on NCCL. The rest of the transport is Zig: the region (`HostBuffer.allocMapped`),
  the HCAs (`TF_ROCE_HCA`, else `NCCL_IB_HCA`, the first two), the GID index (`TF_ROCE_GID_INDEX`, else `NCCL_IB_GID_INDEX`, 3), the connection blobs and verdicts swapped over the
  ranks' TCP pair, the GPU wait limit `TF_ROCE_WAIT_S` (default 300 s, since the ranks load weights on their own).
- `TF_GLM_COMM`: unset or `auto` (the server): RoCE when it sets up on both ranks, else NCCL with a warning; `roce`: required; `nccl`: off. `tf-glm check` runs on NCCL unless
  `TF_GLM_COMM=roce`. `Comm.allGather` decides by byte count, which the matching call on the other rank shares; the row maxima's gather of `greedy` goes this way too.
- `tf-glm check` against the v1.10 references with `TF_GLM_COMM=roce` (DFlash2, copies): w1 12, w2 14, w3 2, w4 88 PASS; w1 with the MTP head 12 PASS.
- Native, DFlash2 + copies, greedy, one session (tok/s, NCCL / RoCE; two runs each): cedit 101.7, 101.3 / 105.1, 105.8; quote 135.4, 132.8 / 137.9, 138.2; story 43.0, 43.1 / 46.1, 46.6;
  code 85.5, 86.6 / 87.4, 88.0; facts 56.1, 56.8 / 59.1, 59.5; json 83.0, 83.7 / 87.0, 86.5; long 52.0, 52.7 / 54.6, 54.6; huge 65.1, 66.1 / 68.3, 68.1.
  The eager block pass (`TF_GLM_DF_PROF=1`) goes from 5.9 to 3.2 ms (the o and down projections' gathers 1.83 + 1.64 -> 0.36 + 0.58); the captured block's startup
  timing from 3.8-4.1 to 3.5-3.8 ms (v1.10: 3.0), a graph having hidden most of NCCL's launch cost already.

