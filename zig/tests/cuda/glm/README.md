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
- Not ported, because the Zig engine does not use them: DFlash2
  (`TF_GLM_DFLASH_POLICY`), multi-stream serving (`TF_GLM_MULTI_*`, `FILL_*`, `PREFILL_OVERLAP`, `HC_SPLIT`, `HC_EXCHANGE`), the
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
