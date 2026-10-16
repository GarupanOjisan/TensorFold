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
