# glm_decode.zig

The full text of the comments that the code now carries as one line each, by the anchor the code names.

The file header: prefill, serial decoding and MTP-drafted decoding over the forward (v0.6.6 glm5_next/cuda/decode.py):
both ranks run the same calls, and every choice comes from the gathered greedy picks, so they agree without messages.

## prefill

decode.prefill without a resume point: the prompt in chunks, each also absorbed by the MTP head (its rows from the
chunk's second token on), then the first token. The last row's hidden stays in e.st.last_hidden for the first draft.

## prefill-chunk

One prompt chunk, `rows` rows of `prompt` from `start` (the state's position): its caches, the MTP head's rows (each
with its following token, the next chunk's first included), its last row's logits and hidden kept.

## absorb

decode.absorb: kept rows (hidden, next token) into the MTP cache, after trimming the last round's draft entries; the
last row's draft logits in mbuf.logits.

## run

What a drafted decode did: rounds, drafts kept, the rounds that verified copied drafts (and their drafts and kept
ones), and how many windows ran at each row count.

## mtp-decode

decode.mtp_decode, greedy, `depth` drafts a round: verify the pending token and the drafts, keep through the first
mismatch, draft again from the kept rows. With `copies` (v1.8's COPY_DRAFTS, glm_copy.zig; `prompt` is the prompt the
pending token follows) a round whose context repeats earlier text verifies the copied continuation instead, and the
MTP head only absorbs the kept rows.

## dflash-decode

decode.dflash_decode, greedy: each round verifies the pending token and either a copied continuation (v1.8's
COPY_DRAFTS, when `copies` is given and the context repeats) or DFlash2's chain (`policy`), keeps through the first
mismatch, then adds the kept rows' taps to DFlash2's context. `prompt` is the prompt the pending token follows.
