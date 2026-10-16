# glm_node.zig

## commands

One rank of the two-rank GLM engine and the commands that keep the ranks in step: rank 0 sends each command to
rank 1 over the TCP pair before running it, rank 1 runs the same commands in the same order. Every token the
ranks pick comes from the same all-gather, so they agree without sending results back.

## start-up-check

Both ranks must have been started with the same window and the same TF_GLM_DENSE (their all-gathered partials
would mix 4-bit and BF16 projections) and the same row split (TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP: one rank's
swaps would wait on the other's forever).

## begin

A prompt pass starts: from 0, or from kept state `snap` after `at` tokens (the attention caches' rows below
`at` must still hold the prompt's tokens; the caller checks). The MTP head absorbs the prompt too when `mtp`.

## chunklanes

One prompt chunk over several begun lanes' next rows (multi_prefill.prefill_pieces; rank 0 sends it, both run it):
`items` (lane, at, rows), each piece at the next multiple of 64 rows of the prompt buffers; then per piece its taps
into its lane's DFlash2 context and its lane's commit. Not with the MTP head (its cache is not grouped).

## first

The first token after the prompt's last chunk. k 0: greedy, the token returned; else the last row's k
candidates a rank gathered (Engine.cand_host), or both ranks' whole rows (whole_vocabulary).

## dflashturn

DFlash2's turn (rank 0 sends it, both run it): from .verify the first `keep` rows of the last window join its context
(their taps); then, with `block`, the block pass for `pending`, whose merged candidates rank 0 reads back (`chainOf`).

## dflashtapslanes

DFlash2's context updates of several lanes after a shared round (rank 0 sends it, both run it): lane items[i][0]'s
first items[i][1] rows of the last window join its context, stacked in one pass (dflash.Batch.taps), else lane by lane.

## dflashblocks

The block passes of `lanes` (each at its context's end, `pend` its pending token) in one pass (dflash.Batch.blocks;
rank 0 sends it, both run it, rank 0 reads the candidates back into each lane's drafter).

## whole-vocabulary

A `k` asking for a draw over the vocabulary (sampling.nucleus_rows, or a top_k past the candidates a rank gives): the
forward leaves its rows' logits where they are and the draw's `nucleus` commands follow.

## nucleus

The draw's gather (rank 0 sends it, both run it): `count` > 0: Engine.nucleusGather's candidates; 0: every rank's
whole rows (Engine.fullRows, into Engine.full_host). The words are valid until the next call.

## verifyshared

Several lanes' windows in one forward (Engine.forwardShared): `segs` in order, `ids` their rows back to back. Each
segment's tokens are then picked with `pickSegment` (and drawn with `nucleus` after binding its lane), in order.

## picksegment

Segment `g`'s rows of the last shared window (rank 0 sends it, both run it): each row's greedy token into `out`
(k 0), or its k candidates a rank gathered (Engine.cand_host), as `verify`.

## pickrows

Every row of the last shared window at once (rank 0 sends it, both run it): `pick` over its first `rows` rows of
buf.logits (the segments back to back from row 0), one greedy (or top-k candidates) pass, one gather, one copy back.

## grow

Lanes up to `total` on both ranks, each with its own DFlash2 context when drafting with it: how many both have
(rank 1 answers with its count).

## copy

Lane `src`'s attention-cache rows below `rows` into lane `dst` (Engine.copyLaneRows). DFlash2's context comes
whole with the kept state `begin` restores, so it is not copied.
