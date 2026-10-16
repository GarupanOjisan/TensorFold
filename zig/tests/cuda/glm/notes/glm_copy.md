# glm_copy.zig

The full text of the comments that the code now carries as one line each, by the anchor the code names.

## overview

v1.8's prompt-lookup ("copy") drafts (glm5_next/cuda/copy_drafts.py, patches 0007, 0013 and 0032): when the context's
last `match` tokens (the pending token included) occurred before, the tokens that followed their latest earlier
occurrence are verified in place of the MTP head's drafts. They only propose; verification keeps the serial sample.

Three rules differ from the lanes core's `SuffixLookup` (a 3-gram index, the longest match of at least 4 tokens, silent
for 16 rounds after four missed ones), so GLM brings its own (glm_native.zig hands it to the stream):

- which occurrence: the latest one with `k` tokens after it, else the earliest (not the longest match);
- an occurrence that starts inside the reply must match `reply_match` tokens (COPY_CODE: 16), not `match` (8): code
  replies repeat short boilerplate whose continuations differ;
- a window of 9 to 15 rows is padded to the CUDA-graph width (16) by repeating its last draft, when the room allows.

There is no silence, and a copy of one token is a copy (the lanes core wants two).

## occurs

Whether `ctx[i .. i + match]` is an earlier occurrence of the context's last `match` tokens (`prompt` leading tokens
are the prompt's: an occurrence starting past them also needs the tokens before it to match).

## propose

CopyDrafts.propose: up to `min(most, room)` drafts, what followed the latest earlier occurrence that has that many
tokens after it, else the most after any occurrence (the earliest); none when the suffix never occurred before. A
window of more than `graphed` rows short of the `pad` width is filled up to it by repeating the last draft, when `room`
allows. `ctx` is the prompt, the reply so far and the pending token; the result lives in `buf`.
