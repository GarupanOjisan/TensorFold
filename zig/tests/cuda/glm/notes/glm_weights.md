# glm_weights.zig notes

Full text of comments shortened in `glm_weights.zig`, `glm_weights_loader.zig` and `glm_weights_draft.zig`.

One rank's half of GLM-5.3-Flash's EXL3 checkpoint on the GPU, laid out as the Python engine lays it out (v0.6.6
families/glm5_next/cuda/weights.py `load`, split.py `rule`): BF16 projections, rows or columns split between the two
ranks, the routed experts' EXL3 trellises stacked per layer with pointer and width tables.

## kind

`Kind`: how a dense (non-expert) projection is held. TF_GLM_DENSE (v1.8 qmm.dense_kind): `q4`, the default, quantizes
the projections at load to 4-bit groups of 64 (MSE-searched) and the head to FP8; `b16` keeps the checkpoint's BF16.

## mat

`Mat`: a projection [n, k]: BF16 rows (qmm.B16: ptr), tiled 4-bit groups (qmm.Q4: ptr the words, s and b the bf16 scales
and biases, npad the columns padded to 128) or e4m3 rows with fp32 scales per 128 columns (qmm.F8: ptr, s).

## draft

`Draft`: the DFlash2 draft model on this rank (dflash2.Drafter's constructor): config, 4-bit weights (min/max ranges,
TF_GLM_DRAFT_QUANT unset), the selector's projection and, on rank 0 (which picks the chains), its codebooks.

## head

In `load`: the head is this rank's half of the vocabulary rows, as BF16 rows only until the draft head (min/max 4-bit,
from the BF16 rows as weights.py does) and, under TF_GLM_DENSE=q4, the FP8 head are made from them.

## qdump

`dumpQuant` (`tf-glm qdump`): the dense quantization of a few tensors of `rank` (the loader's own path), the head's FP8
and draft-head 4-bit rows too, as raw bytes in `out`: NAME.w, .s, .b (words, scales, biases), head.q8, head.sc, draft.w,
.s, .b. `only` (comma separated names, or empty) picks some.

## loaddraft

`loadDraft` (glm_weights_draft.zig): dflash2.Drafter's weights of the model in `dir` for rank `rank`: KV heads in whole
query groups (halves), the MLP in units of 128 (tp.draft_parts), every matrix 4-bit with min/max ranges on the GPU.
`books`: also the host codebooks.

## mat-loader

`Loader.mat` (glm_weights_loader.zig): a dense projection (weights.py `q4`, `stack`): the rank parts of `names` in rows,
BF16 as stored or, under TF_GLM_DENSE=q4, quantized on the GPU after the tp split, as `qmm.make_dense_q4`.

## quantize

`Loader.quantize`: bf16 rows [n, k] on the device as a dense q4 projection: groups of 64 along k as 4-bit (`mse`: the
least-squares search, else min/max), else, where k is not a multiple of 64, as FP8 (make_dense_q4's rule).
