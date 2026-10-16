# glm_hc.cu notes

## order

GLM-5.3-Flash's hyper-connection mixing dots for decode rows (hc_pre's first half) with more blocks than
glue._hc_partial, reproducing its bits: that Triton kernel (4 warps, a [32, 128] bf16 tile in the blocked layout
sizePerThread [1, 8], threadsPerWarp [2, 16]) sums, for each of 8 steps of 128 inputs, a row's 8 products per
thread in input order (products of bf16 values are exact in fp32, so fused or not), then across the 16 lanes of
the row with butterflies 8, 4, 2, 1, and adds the steps to a zero accumulator in order; the sum of squares keeps
one input per thread (4 warps x 32 lanes), adds the 8 steps' squares to zero in order, then butterflies 16 .. 1
within each warp and (w0 + w2) + (w1 + w3) across the warps. PART [rows, 16 blocks, 32]: the 24 dots, then the sum.

Block (K block, part, row group): warps 0..7 the dots of mixing rows 8 part .. 8 part + 7 (their weights read once
for up to RPB rows), warp 8 of part 0 the squares.
