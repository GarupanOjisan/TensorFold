# glm_kernels.zig and its split files

## qmm-configs

`glm_kernels.zig`:

v1.8 qmm.cu's qmm_kernel<64, 16, bn, 1, wn, stages> for a decode window (rows <= 16). Config 0 is `qmm` with the
16-row tile; 1 and 2 are `qmm_cfg` 2 and 4. dec_tile (glm qmm.py DEC_TILES) picks one by the projection's shape.

## e18

`glm_kernels.zig`:

glm_exl3d.cu's and experts.cu's instantiations (cuobjdump -symbols), and the prompt kernels' dynamic shared memory
(exl3.cu prompt_smem<64, XM>: three stages of the members' rows and the trellis words, or an epilogue half).

## wide-rows

`glm_kernels.zig`:

kda.py WIDE_ROWS (TF_GLM_KDA_WIDE_ROWS, v1.8's default 1; 64 was v0.6.6's): from this many rows a chain runs as prep, step
and out (the same bits as the one-block chain). The engine sets it from the environment before it sizes its buffers.

## dec-order

`glm_kernels.zig`:

TF_GLM_EXL3_DEC_ORDER (v1.10 patch 0090): the decode expert kernel's block launch order, 0 as launched, 1 or 2 (default) a k slice's
blocks together; the same bits. The engine sets it from the environment.

## fuse-gu

`glm_kernels.zig`:

TF_GLM_EXL3_FUSE2 (default on; 0 off): the gate/up epilogue in the gate/up kernel's last block (the same bits). The engine sets it.

## routed18

`glm_kernels.zig`:

exl3_mm.routed for GLM at two ranks (D 4096, an expert width of 1024) on `plan` (route): Y per routed slot into
`sc.y`. A prompt chunk (`prompt`: plan items of 64 pairs) runs the Y^T kernels, a decode window (under 64 rows,
items of 16) the `dec` kernels with TF_GLM_EXL3_FUSE 1 (the down epilogue fused) and the 16-byte loads
(TF_GLM_EXL3_LOADS=nc). `ex.shared_suh` (0: none) is the one gate/up suh of a layer whose experts all share it.

## quant

`glm_kernels_q4.zig`:

Which ranges a group's 16 levels span: its min and max (qmm.quantize4, the draft head's), or the least-squares one
of 16 clippings (qmm.quantize4_mse, TF_GLM_DENSE=q4's projections).

## qmm-shared

`glm_kernels_q4.zig`:

qmm.matmul of 17 to 32 rows (a shared window): `qmm` on the 32-row tile, K slices as qmmDecode's (denseSplitK), so each
row's bits are those of its lane's own window on a 16-row tile.

## qmm-decode

`glm_kernels_q4.zig`:

qmm.matmul of a 4-bit q for a decode window (shared.matmul -> `qmm` or, with dec_tile, `qmm_cfg`: v1.8's qmm_kernel):
x (rows <= 16, k) bf16, rows `x_stride` elements apart, with its group sums xs -> out (rows, n), fp32 with `f32`.

## prompt-smem

`glm_kernels_exl3.zig`:

exl3.cu prompt_smem<64, XM>(): the larger of the stages (XM member-row copies and 2 x PWORDS halves, 3 deep) and the epilogue halves

## chain

`glm_kernels_kda.zig`:

One KDA layer's chain (kda.chain): the conv, gates and delta rule over `rows` rows from `state_in`, the new state
into `state_out` (0: none) and each row's k, v, g, b saved for a later replay. Element strides.

## chain-segments-args

`glm_kernels_kda.zig`:

kda.chain_segments' arguments: Chain's, with the lanes' states stacked by slot (strides in elements, as kda.cpp passes
tensor strides): rec [slots, 2, layers, H, 128, 128] f32 at this layer (`rec` points at slot 0, parity 0, this layer),
conv [slots, layers, 3, 3 H 128] bf16 at this layer; `seg` the device table of `nseg` rows.

## chain-segments

`glm_kernels_kda.zig`:

kda.chain_segments (kda_chain_wide_segments_cuda, v1.8 kda.cu:604-627): one layer's window cut into the table's
segments, each the bits of a solo wide chain of its rows; conv is read only (the commit shifts it).

## replay-segments

`glm_kernels_kda.zig`:

kda.replay_layers_segments (v1.8 kda.cu:629-634): each segment with keep < rows, every layer's state after its first
`keep` saved rows, rec[slot, parity] into rec[slot, 1 - parity]. rec at slot 0, parity 0, layer 0; strides in
elements; the saves [layers, scratch_rows, H, ...] of the window's rows.

## hc-partial

`glm_kernels_kda.zig`:

hc.hc_partial (v1.8 hc.cu): decode rows' 24 mixing dots and sum of squares of x [rows, 16384] bf16 against fn [24, 16384]
into part [rows, 16, 32] fp32 (glue._hc_partial's bits, 48 blocks a row).

## l2-prefetch

`glm_kernels_kda.zig`:

l2pf.prefetch: `count` (address, bytes) pieces of the table at `table` + 16 `first` into L2 (mode 0: cp.async.bulk.prefetch.L2,
1: prefetch.global.L2::evict_last per line, 2: loads), on the stream given.

## chain-chunked

`glm_kernels_kda.zig`:

kda_chunked.chain: a prompt chunk of `c.rows` rows whose first row sits at absolute position `pos`, in 32-row
sub-chunks at absolute positions (TF_GLM_KDA_CHUNKED=1). Writes `c.out` and `c.state_out` only; the conv state is read.
