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
