# glm_segments.zig

The full text of the comments that the code now carries as one line each, by the anchor the code names.

## overview

The device tables of a shared verify window's DSA rows (v1.10 segments.SegRows, without index rings): several lanes'
rows back to back, each lane's per-token caches in its extent of the layer arenas (fwd.Pool, extents at multiples of
fwd.extent tokens; its index keys and gates in the same extent, so ibase = base). The segmented kernels (latent.seg_*,
sparse.seg_*) read these tables; one CUDA graph a window size then covers every mix of lanes.

`Span` is one lane's rows of a window: its extent's first arena row (`base`, a multiple of `extent`), the position of
its first row, and its rows.

## table

SegRows' one int32 table: per row pos, base, seg, sparse, ibase (max_rows each), then per segment seg_start, seg_rows,
seg_pbase, seg_pools (max_segs each).

## fill

SegRows.fill: the host table of `spans` in window order into `t`; `sparse_from` the first position that attends
sparsely (null: no index, every row dense). Returns the window's rows.
