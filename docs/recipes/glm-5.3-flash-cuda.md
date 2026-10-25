# GLM-5.3-Flash on CUDA

The native CUDA build serves GLM-5.3-Flash at tensor parallel over hosts: from NVIDIA's ModelOpt NVFP4 checkpoint
([1.0.12](../../CHANGELOG.md#1012)), and on two ranks from Mia-AiLab's EXL3/TR3 checkpoint
([1.1.0](../../CHANGELOG.md#110), [the checkpoint](glm-5.3-flash.md#exl3)). Rank 0 serves HTTP and the other ranks
follow its steps, with MTP drafts, concurrent streams and kept prompt states. The
[Python 0.6.6 recipe](glm-5.3-flash.md) and its `--tp`, `--rank` and `--master` flags belong to the frozen engine.

## Run

`TF_GLM_RANK` of `TF_GLM_WORLD` places each rank, and `NCCL_COMM_ID` names rank 0's address and port on the link
between the hosts ([placement](../multi-rank-receipts.md)). Give every rank the same arguments: before the weights load
the ranks compare their settings, model and build, and name the field and rank that differ. `--context` takes a
positive limit for GLM. On two GB10s, with 192.0.2.1 as rank 0's address on their direct link:

```bash
tensorfold pull nvidia/GLM-5.3-Flash-NVFP4@423acf37583782c51c142d145aef733d72943d93
# on rank 1
TF_GLM_RANK=1 TF_GLM_WORLD=2 NCCL_COMM_ID=192.0.2.1:29583 \
  tensorfold serve nvidia/GLM-5.3-Flash-NVFP4 --context 245760 --parallel 1
# on rank 0 (192.0.2.1)
TF_GLM_RANK=0 TF_GLM_WORLD=2 NCCL_COMM_ID=192.0.2.1:29583 \
  tensorfold serve nvidia/GLM-5.3-Flash-NVFP4 --context 245760 --parallel 1
```

Set `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA` for the link between the hosts as NCCL documents them. Each rank keeps a
TCP line to rank 0 on the port after `NCCL_COMM_ID`'s (29584 here), opened before NCCL: a rank that fails, refuses or
dies stops the others.

The contributed runs read that revision. The repository's later commit `da920bb0`, which declares the MTP layer as
BF16 in the quantization ignore lists, was not among them. On shipped 1.1.0 these settings loaded the whole checkpoint
in 302.1 s, and rank 0 reported `model 96.11 GiB; 1 stream at once, 3.16 GiB each at a 245760-token window, of 5.8 GiB
left after a 12.2 GiB reserve` and a 5.8 GiB prompt cache
([#1040](https://github.com/ashhart/TensorFold/issues/1040)).

## Requests

- `--vision` on every rank reads images and video: rank 0's tower encodes them and sends their rows and spans to the
  other ranks ([1.1.0](../../CHANGELOG.md#110)). Video needs `libtfvideo`
  ([video](../vision.md#video-on-the-native-server)).
- Nonstreaming `/v1/completions` requests return `prompt_logprobs`, structured output follows its schema with drafts
  on, and required and named tool calls use the model's tool markup ([1.1.1](../../CHANGELOG.md#111): #1016 to #1018).
- `--kv-cache fp8` is an opt-in FP8 latent cache. BF16 remains the default, and its native quality and device
  qualification are pending ([1.1.2](../../CHANGELOG.md#112)).

## More ranks

`TF_GLM_WORLD=3` places a third rank the same way. From 1.1.2 the uneven KDA input projections of TP=3 (22, 21 and
21 heads) run instead of being refused; TP=3 device qualification is pending
([#1067](https://github.com/ashhart/TensorFold/issues/1067)). How a prompt's rows reach the other ranks
(`TF_GLM_PREFILL_REDUCE`) and the two-rank RoCE gathers (`TF_ROCE`) are in
[placement](../multi-rank-receipts.md#the-settings-the-ranks-agreed-on).

## Copy windows

Rank 0 plans copies. The buffers below are allocated on every rank.

| Variable | Default | Effect |
| --- | --- | --- |
| `TF_GLM_COPY_MAX` | 15 | Drafted tokens a copy may hold. `7` is the previous 8-row window. Clamped to 1..15. |
| `TF_GLM_COPY_REPLY_MATCH` | 16 | Matching tokens required when the copied continuation starts in the reply. Prompt copies stay at 8. |
| `TF_GLM_COPY_EXTEND` | off | `1` enables context copies, root copies, and a 16-token match that may use the whole copy cap at once. |

DFlash remains a 7-draft block. The MTP head still chunks its absorb at 8 rows. A copy may grow past its first 3
tokens after a fully accepted copy, in a lone round and in a shared round. The shared forward still admits at most
32 rows. Native GLM verify does not capture CUDA graphs, so there is no 16-row graph set; the KDA wide kernel
already accepts a 16-row window. Set the same values on every rank. Compare `token_sha256` with `"draft": false`,
and watch memory: the KDA save and the shared per-stream chain are sized for 16 rows.

## Long runs on GB10

Long prompts and long runs of replies heat a GB10. The runbook's [heat bands](../../RUNBOOK.md#linux-and-cuda) (with a
1,035,295-token prompt at TP=3), [CPU pinning](../../RUNBOOK.md#gb10-cpu-pinning-for-long-cuda-runs) and
[GPU clock cap](../../RUNBOOK.md#gb10-gpu-clock-cap-for-long-cuda-runs) each carry a measured GLM-5.3-Flash run.
