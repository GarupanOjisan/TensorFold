# glm_native.zig and its glm_native_*.zig parts

The full text of the comments that the code now carries as one line each, by the anchor the code names.

## overview

GLM-5.3-Flash on two CUDA ranks behind the native family interface (native/cuda.zig): this process is rank 0 and
serves; rank 1 runs `tf-glm follow` on the other Spark. A stream a lane (each lane one sequence's caches and DFlash2
context), the lanes' rounds taking turns; an idle lane keeps its caches' rows, so a later turn of its conversation
resumes there. Windows of up to 16 rows, MTP drafts or DFlash2's (TF_GLM_DRAFTER_DIR; glm_dflash.zig) and v1.8's
prompt-lookup (copy) drafts (glm_copy.zig). TF_GLM_PORT (default 29660) is the port rank 0 listens on for rank 1.

The file is split by method group: glm_native_sample.zig (the draws), glm_native_prompt.zig (prompt passes),
glm_native_round.zig (verify, keep, draft, release), glm_native_timing.zig (start-up timings). `Glm` reaches each
moved method under its old name through an alias.

## reads

Whether this port reads `dir`: an EXL3 checkpoint (config.json's quantization_config names quant_method "exl3");
other glm5_next checkpoints (NVIDIA's NVFP4) go to families/glm/cuda_native.zig.

## weight-share

This rank's share of the checkpoint (the routed experts and most projections halve, the embedding and the replicated
layers do not): what the memory admission compares with the budget before loading.

## copy-rule

glm_copy.zig as the lane core's proposer (lanes/proposer.zig): one a lane, reset by each prompt pass there. The core
asks twice a round with the same context (the head's depth rule, then the next plan): the second answer is the first.

## live-tokens

`Lane.live`: the tokens whose attention-cache rows are committed, from position 0: a kept state resumes only while the
rows below its position still hold its tokens (the caches are positional; only KDA's state is copied).

`liveHolds`: whether lane `ln`'s attention caches' rows still hold kept state `sn`'s tokens: the `at` before it and the
one its MTP rows read past it.

## shared-costs

`Glm.shared`: shared windows' ms by total rows: the lone windows' until `grow` times them over two lanes (the facts
point here, and the lane core reads them after `grow`).

## hidden-rows

In `facts`: several live streams share a forward (verifyShared), up to a window's rows; the lanes `grow` allocated cap
how many streams are live, so one lane never shares.

## grow

Lanes up to `total` on both ranks, allocated now (native/cuda.zig calls this once, after the memory admission and
before serving): how many there are.

## snapshots

Kept prompt states (prompt_cache.Snapshots, wired by native/cuda.zig).

A kept state: KDA's state after `at` prompt tokens on both ranks under `id`, and the hash of the `at + 1` tokens the
attention caches must still hold for it (liveHolds).

## cache-rules

prompt_cache.Rules: the MTP head's row at-1 reads token at; prompt kernels change bits with a chunk's rows, so states
are kept and resumed at the request's planned chunk starts only.

## admit

In `admitPrompt`: a kept state of this prompt's prefix resumes in a lane whose rows hold it, or in a free lane they are
copied to; without one the pass starts at 0 in the free lane used longest ago.

## prefill-many

One prompt-chunk round over `streams` (Backend.prefill_many; v1.10 multi_prefill with FILL's alternation, the host
running a decode round between calls): a stream not on a lane yet is admitted (admitPrompt); then one piece a prompt,
its next cut exactly as its own pass takes it (nextCut), the fewest rows left first, while the prompt buffers hold them
(each piece at a multiple of 64 rows), in one chunk (Node.chunkLanes; one piece: Node.chunk); a prompt that does not
fit waits. The streams whose prompt is in (their first token drawn) or that were cancelled move to the front; their
count is returned. With the MTP head each prompt fills whole, alone. While a lane outside `streams` holds a stream (it
decodes between these calls) a round takes at most busy_rows rows, so its pause stays short; otherwise the whole prompt
buffers.

## shared-pick

In `verifyShared`: one pick over every row when the lanes draw alike (greedy, or the same top-k), as v1.10's packed
all-gather; a whole-vocabulary draw (top_k off) reads the bound lane's rows, so it picks lane by lane.

## dflash-draft

DFlash2 in place of the MTP head (v1.8's dflash_decode): the kept rows' taps join its context, then, unless a copy is
ahead (v1.8 takes the copy first and runs no block), its block pass and chain (its own stop rule, not `r.depth`, up to
TF_GLM_DFLASH_POLICY's most) give the next window's drafts as host tokens (`treeFn`). The first round after a prompt
is taken as `r.depth` held drafts (the core does not ask for the chain there), filled up with the pending token.

## dflash-taps

The bound lane's DFlash2 turn up to its chain: commit, the context update and, with `block`, its own block pass.
`turn` false: no DFlash2 turn here (the caller adds the kept rows to the context, Node.dflashTapsLanes).

## dflash-lanes

A shared round's DFlash2 turns (facts.draft_streams): each lane's context update on its own, then the block passes of
every lane that wants drafts in one (Node.dflashBlocks; one lane alone takes its own captured pass), then the chains.

Every lane's kept rows go in one commit (Node.keepLanes), before the turns: each turn's own commit then finds nothing.

## choose-whole

Each row's draw over the vocabulary (Node.whole_vocabulary): top_k off: sampling.nucleus_rows, its candidates a rank
first (`nucleus_first`, then `nucleus_wider`: TENSORFOLD_NUCLEUS_UNION, v1.10 patch 0034's rule), whole shards when a
row's nucleus runs past them; else choose_rows over every token (a top_k past the candidates a rank gives). Row r keyed
at position0 + r.

Without the union rule (stock v1.8) a row whose nucleus is not inside one rank's candidates goes straight to the shards.

## draw-nucleus

sampling._draw over the ranks' `count` best candidates (`words`, Engine.nucleusGather's layout): false when some row's
nucleus is not inside them (a wider gather or the whole shards decide it). With the union rule
(TENSORFOLD_NUCLEUS_UNION, default on) a row is covered when the merged walk to the cumulative mass ceil(top_p total)
stops above every rank's last candidate: no token left out can be in the nucleus, so the draw is the whole vocabulary's.

## candidate-count

The former comment, both of its parts:

- A stream's candidates a rank (top_k + MARGIN); 0 for greedy. top_k 0 (the full-vocabulary nucleus) is not ported.
- k candidates a rank (top_k + exact_sampling.MARGIN), 0 for greedy, or the whole vocabulary: top_k off
  (sampling.nucleus_rows) or past the candidates a rank gives.

## nucleus

sampling.nucleus_rows for one row over the whole vocabulary `row` (bf16 bits by token id). Python reads each rank's top
1,024 first and every whole shard when a row's nucleus runs past them, so its draw is this one: tokens in (value desc,
id asc) order with v = logit / T in float64, mass floor(exp(v - top) 2^40); keep a token while the mass before it is
below ceil(top_p total) (top_p in (0, 1)), and while v >= top + ln(min_p); the Gumbel race v - ln(-ln u(seed,
position, id)) among them, the first maximum in that order. Both cuts keep a prefix of the order, so the kept set is
found by selection (no sort). Tokens 42 or more below the top are left out: their mass is 0 (below 2^-40) and their
race score stays under the top token's (the race's noise spans 41.1).

## prefix-within

Reorders `a` so its kept tokens come first and returns how many: a token is kept when the mass of the tokens before it
in (value desc, id asc) order is below `need` (quickselect on that order, expected linear).

## timing-ids

v1.10 verify.timing_tokens(vocab, 32): numpy default_rng(1234).choice(150000, 32, replace=False), the distinct ids its
startup timings run (one token repeated, or a run of neighbours, routes to a few experts and times light).

## measure-shared

Shared windows of 2 to max_rows rows over lanes 0 and 1 (half each), timed on whatever their caches hold (both ranks run
them; no commit, so the lanes stay where they were). Serving overwrites the rows they wrote.

## compare-one

v1.10 multi._compare_paths: one lane's window of `rows` rows on the one-stream graphs (verify) and on the shared ones
(verifyShared over a single segment), the same tokens, timed by GPU events (fastest of 5, after a warm-up), each with
its greedy pick; logged only.

## shared-flat

TF_GLM_SHARED_FLAT (default on; 0 off): every shared width costs as 2 rows, so the core keeps every draft in a shared
round, as v1.10 does (its multi-stream rounds take each stream's drafts with no depth policy): +1.7% at four streams,
the same replies.

## dflash-timing

In `measure`: DFlash2's pieces (Engine._calibrate's "block" and "taps8"): the block pass with its candidates read back,
the chain on the host, and a context update of the last window's first 8 rows.
