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
