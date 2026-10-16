# glm_main.zig

## usage

`tf-glm`: GLM-5.3-Flash on two CUDA ranks without Python, and its check against the Python engine's replies.

```
tf-glm check MODEL --rank R --master HOST [--port P] --prompts prompts.json --want tokens.json
       [--max-tokens N] [--context N] [--depth D] [--copy 0|1] [--drafter DIR] [--only NAME,...] [--kernels DIR]
tf-glm qdump MODEL --rank R --out DIR [--only NAME,...]   (TF_GLM_DENSE=q4's quantization of a few tensors, as bytes)
```

Run once on each Spark (rank 0 on HOST). Each prompt is prefilled and decoded serially, then with MTP drafts;
rank 0 prints whether both equal the capture's serial reply. `--copy 1` drafts by v1.8's prompt lookup where the
context repeats (glm_copy.zig; TF_GLM_COPY_MAX, _MATCH, _REPLY_MATCH and TF_GLM_WIDE_GRAPHS as in Python) and the MTP
head elsewhere. `--drafter DIR` (a DFlash2 snapshot) drafts with it instead of the MTP head, as v1.8's DRAFTER=dflash2
does (`TF_GLM_DFLASH_POLICY`, default fnc7:0.3).
