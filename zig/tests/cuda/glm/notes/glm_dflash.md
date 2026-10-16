# glm_dflash.zig notes

Full text of comments shortened in `glm_dflash.zig`, `glm_dflash_chain.zig` and `glm_dflash_batch.zig`.

## overview

DFlash2 drafts for GLM-5.3-Flash on two CUDA ranks (v1.8 glm5_next/cuda/dflash2.py, the single-stream `Drafter`): a five
layer block model with two-tap dynamic convolutions and sliding-window attention reads the means of the target's streams
at five layers (the "taps" of every committed row) and, for [pending token, mask x (block - 1)], proposes candidates for
the next block - 1 positions: each rank's top 16 of its half of the vocabulary, merged, with a selector's rows. A chain is
then picked on the host (the selector's edge scores and the target's keyed noise) and cut by a confidence rule. Drafts only
propose; the verify window keeps the serial sample, so the arithmetic need not equal Python's bit for bit.

Both ranks run `addTaps` and `candidates` with the same all-gathers; rank 0 alone reads the candidates back and picks the
chain, whose tokens ride the next verify command to rank 1.

## drafter-init

`Drafter.init`: the drafter's buffers and caches on this rank (dflash2.Drafter.__init__): the context of `capacity`
positions in a ring of the window and block when that is shorter (TF_GLM_DRAFT_RING=0 keeps a flat one).

## adopt

`Drafter.adopt`: moves the caches to `cache` (cacheBytes), their contents with them. The position stays where it is
(16-aligned: the one-lane _dattn_kernel was captured with POS divisible by 16); the batch copies it into its `meta` each
pass.

## profile

`Drafter.profile`: eager block passes timed by piece, printed (both ranks run it alike): embed and rope, each layer's input
stretch (norm, conv projections, qkv), prep/scatter/attention, the o projection with its all-gather, the MLP up to swiglu,
the down projection with its all-gather, and the head with the candidates' gather and the selector's projection.

## block

`Drafter.block` (dflash2.Drafter.candidates): the block pass for `pending` at the context's end. `read`: also bring the
merged candidates to the host (rank 0 alone); the lists are then valid until the next call.

## patch-0092

v1.10 patch 0092: an id outside the vocabulary is a float's bits read before the copy had landed; the rows are copied
again once the device is idle, and a second failure is an error. `Batch.readBack` does the same as `Drafter.block`.

## policy

`Policy` (glm_dflash_chain.zig): engine.encode_policy's DFlash2 stop rule `fncN:P[:B]` (the product of the noise-aware
confidences holds P) or `fcN:P` (the product of the plain ones); TF_GLM_DFLASH_POLICY, v1.8's production setting
`fnc7:0.3`.

## chain

`chain` (glm_dflash_chain.zig, reached as `Drafter.chain`): dflash2.Drafter.chain: pick candidates with the selector's
edges and the target's keyed noise; stop below the cumulative confidence, always keeping the first draft. `first` is the
position of the first draft. Up to `depth` tokens.

## batch

`Batch` (glm_dflash_batch.zig): block passes of several lanes' drafters in one launch sequence (v1.10
dflash2_multi.MultiDrafter.candidates): the rows of every lane's block [pending, mask x (block - 1)] back to back. The
row-independent work (embedding, norms, the 4-bit projections in pieces of 16 rows, the o and down projections' gathers,
the head, its top-k and the selector's rows) runs on all of them; each lane's rotary, `_prep`, cache writes and attention
read its own drafter's ring and position, and the dynamic convolutions run a block at a time (a block's first row mixes
no row before it). Each drafter is then left with its merged candidates as its own `block(read)` leaves them, so its
`chain` is unchanged. Eager; drafts only propose.

## batch-taps

`Batch.taps`: dflash2_multi.MultiDrafter's context update over several lanes (`_launch_taps`): lane s's first kept[s]
rows of the last shared window (its rows from row0s[s], `buf`'s taps) join drafter ds[s]'s context, every row stacked in
one pass (one graph a row count): fc and its norm, each row's rotary at its own position, and per layer the kv
projection, `_prep` and the cache writes. false (nothing done): no pool, or more rows than tmax; the caller adds them
lane by lane.

## batch-pool

`Batch.adopt`: v1.10 MultiDrafter's pool: moves the caches of `ds` (every lane's drafter, before any graph holds their
addresses) into one buffer, drafter s's at s caches in, so `layer` writes and reads every block's ring in one launch
each. A flat context (no ring) keeps its own caches and the launches per lane.

ponytail: each drafter's own caches stay allocated and unused (about 36 MB a lane); give Drafter.init a no-cache mode if
that memory counts.

## batch-mm

`Batch.mm`: a 4-bit projection of `rows` rows (x rows q.k apart): 17 to 32 rows in one launch on the 32-row tile (the
weights read once, v1.10's qmm bucket; drafts only propose, so these bits need not be one lane's), else pieces of up to
16. `Batch.pass` reads the draft head the same way: one launch on the 32-row tile reads the head once; the drafts are
hints, so these bits need not be the one-lane pass's.

## batch-blocks

`Batch.blocks`: Drafter.block for every drafter of `ds` (each at its context's end, `pend` its pending token), in one
pass. `read`: also bring each one's merged candidates to the host (rank 0 alone), as its own `block(read)` would.
