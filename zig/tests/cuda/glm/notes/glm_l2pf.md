# glm_l2pf.cu notes

## modes

L2 prefetch of weight ranges for GLM-5.3-Flash's decode windows (l2pf.py): a small kernel on a side stream asks
the memory system to bring the next kernels' weights into L2 while the main stream waits on an all-gather or runs
latency-bound glue. It only loads or prefetches (no store, atomic or reduction), so no result can change.

- TABLE [n, 2] int64: (address, bytes) pieces, each 16-byte aligned and at most a few hundred KiB (the host splits).
- MODE 0 ("bulk"): one cp.async.bulk.prefetch.L2 a piece (the TMA unit walks it; sm_90+).
- MODE 1 ("lines"): prefetch.global.L2::evict_last on every 128-byte line of a piece, a warp's lanes on adjacent lines.
- MODE 2 ("touch"): ld.global.cg of every line's first 16 bytes (a real load: the kernel lasts as long as the reads).

Adapted from jayleaton/glm53-tensorfold-spark patches/0460 (l2pf.cu; Apache-2.0, Copyright 2026 Jay Leaton).
Changes: one kernel for the three modes over a table of pre-split pieces (the host cuts ranges into pieces, so no
per-thread chunk loop), lines mode a warp a piece with prefetch.global.L2::evict_last, touch mode as loads here.

# glm_l2pf.zig

## sites

TF_GLM_L2PF (v1.8 glm5_next/cuda/l2pf.py): in decode windows a small kernel on a side stream asks the memory system to
bring the weights the main stream reads next into L2 while it waits on an all-gather or runs latency-bound glue. It only
loads or prefetches, so no result changes. Three sites a layer: `a` at the attention block's all-gather (the FFN's
hyper-connection weights and norm, the router and shared expert, or a dense MLP), `f` at the FFN's all-gather (the next
layer's hyper-connection weights, norm and input projections; after the last layer the final norm and the head), `o` after
the attention block's input projections (its output projection). Each site takes up to TF_GLM_L2PF_MB (default 8) MiB:
small tensors first, then a 4-bit matrix's chunk heads where its decode launch is one wave, else the leading bytes of
the large tensors. The side stream forks from the main stream at a site (an event) and is joined before a window ends,
so it also works inside a captured graph.

## q4-heads

l2pf.q4_heads: the first `left / chunks` bytes of every (64-column tile, K slice) chunk of a 4-bit matrix whose decode
launch is one wave, so each program starts on L2 hits.
