# glm_exl3d.cu notes

The kernels live in `zig/kernels/cuda/glm_exl3d.cu` and its pieces `glm_exl3d_dec.cuh` (decode windows) and
`glm_exl3d_prompt.cuh` (prompt chunks), included inside `namespace tf_glm_exl3d`.

## format

GLM-5.3-Flash's EXL3 routed experts on CUDA: the grouped trellis GEMV and the Hadamard rotations around it.

Format (see exl3.py, after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): a 16x16 tile is 32 little-endian
32-bit words; lane L of a warp decodes the tile's values 8L..8L+7 from words L-1 and L, and those eight values
are exactly the B fragments of two mma.m16n8k16 (columns 0-7 and 8-15 of the tile), so a tile goes from memory
to the tensor cores without a shuffle or a layout change.

Every output depends only on its own row: the rows of a window share an mma tile but mma keeps rows
independent, the K range of every warp and split is fixed by the shape, and warps and splits are summed in a
fixed order (shared memory, then the epilogue kernels), never with atomics.

`decode_tile_p` gives decode_tile's values with fewer integer instructions (the prompt kernels): the byte-aligned
16-bit states taken with one PRMT each and w >> 4 shared by two states. The same states, so the same bits.

## ldg-nc-v4

16 bytes of trellis words, read once a window (no L1 allocation); asm volatile keeps it where it stands relative
to the (volatile) mma, so the ring depth is what the code says (patches/0580 of jayleaton/glm53-tensorfold-spark).

## grouped-kernel

Grid (item, n block, mat * SK + split) -> Z[mat][split][pair][n]; warps sum fixed k ranges, added in warp order.
SEQ (prompt chunks): grid (.., .., mat); one program runs the SK splits in turn and writes Z[mat][0][pair][n] =
((0 + split 0) + split 1) + ..., the epilogue's own sum, so an SK = 1 epilogue gives the same bits without the
SK fp32 partials' round trip through memory.

## rot-in

Program (member row, 128-block of K, matrix): Xh[mat][row][block] = fp16((x[row] * suh[mat][e]) @ H) for the
gate and up projections of every routed slot (row, slot) with slot < slots - 1 (the last slot is the shared
expert). One warp per program. ROWS (prompt chunks of a layer whose experts all share one suh for gate and up):
program (row, block) writes Xh[row][block] from suh0 = that vector, the same arithmetic as every pair's input of
that row, so each pair's fp16 input has the same bits, computed once a row instead of 2 x top_k times.

## gateup-epilogue

Program (member row, 128-block of the rank's intermediate width): gate and up outputs summed over the splits in
order, rotated, scaled by svh; GLM's limited SwiGLU with the grouped kernels' bf16 roundings; then the down
projection's input rotation: Xd[row][block] = fp16((act * suh_d[e]) @ H).

`down_epilogue_kernel`, program (member row, 128-block of the model width): the down projection's output summed
over the splits in order, rotated and scaled by svh: Y[row][slot][:] (fp32, this rank's share of the expert's
output).

## decode-windows

Decode windows: grouped_kernel's arithmetic with its epilogues fused (`dec_kernel`, glm_exl3d_dec.cuh).

Every mma chain, warp sum and split sum is grouped_kernel's (non-SEQ, register prefetch of the next k step): a
warp runs k tiles kt0 .. kt0 + STEPS - 1 of its split in order from zero, the warps are added in warp order, the
splits in split order from 0. What changes:

- FUSE 1 (the down projection, SK = 1): the block's 128 columns are one Hadamard block, so it applies
  down_epilogue_kernel's arithmetic to its own sums and writes Y, no Z round trip, no epilogue launch;
- FUSE 2 (gate and up): each block still writes its Z partial; the last of the 2 * SK blocks of an (item, n block)
  to finish (an atomic count, reset for the next launch) runs gateup_epilogue_kernel's arithmetic on the item's
  rows and 128 columns, reading the partials in split order: the same sums whichever block finishes last.
- XROW: X holds one row a token (rot_rows, a layer whose experts share one suh), pair p reads row p / slots.
- LD (TF_GLM_EXL3_LOADS; data movement only, every chain and sum as above, so the same bits):
  - 0: 32-bit __ldg of word `lane` of each tile, one k step ahead, after the count -> item -> members round trips;
  - 1: adapted from jayleaton/glm53-tensorfold-spark patches/0580 (exl3_ld.cu ld_kernel, LD_NC; Apache-2.0,
    Copyright 2026 Jay Leaton). Changes: in our dec_kernel (its items / counts plan, fused epilogues and XROW
    inputs kept), PD fixed at compile time, no probes / PDL. The count and the item are read together and the
    first PD k steps' trellis words are issued before the member rows reach shared memory; the words come as
    ld.global.nc.L1::no_allocate.v4 (lane l: bytes 16 l .. of each 512 B of a step, NT / 4 loads a lane), PD
    steps in flight, through a 1 KB staging area in the warp's own slice of `red` (two __syncwarp a step), read
    back as word `lane` of each tile.

## prompt-chunks

Prompt chunks: Y^T = W^T X^T (glm_exl3d_prompt.cuh).

A decoded trellis tile is, reordered, an m16n8k16 A fragment of W^T (lane (g, t): a0 = b0[0], a1 = b1[0],
a2 = b0[1], a3 = b1[1]), so each warp decodes its own tiles straight into registers and every decoded fragment
feeds the whole pass's member rows (up to 64 tokens, the prefill plan's item) as B fragments from shared memory.
Each output is one fp32 chain of mma over ascending k tiles (no split K, no partials in memory), so a row's bits
depend on its own values only (any chunking gives the same prompt state), though they are not the decode
kernel's (which sums split and warp partials). The epilogues are gateup/down_epilogue_kernel's formulas.

## prompt-block

A block covers 16 m tiles (gate/up: 8 of each matrix = one 128-column Hadamard block; down: 256 columns) for a
pass of PM members with NW warps: each warp holds MT = 16 / NW m tiles for all PM members (PM / 8 n blocks), so a
decoded tile feeds PM / 8 mma. PM 64 / 8 warps (two blocks an SM) or PM 128 / 16 warps (one): the same per-element
chains, so the same bits. Every stage brings the members' next PBK k values and the 16 tiles' trellis words into
shared memory (cp.async, PSTAGES deep). SHX (gate/up of a layer whose experts share one suh for both): gate and up
read the same member rows (X0 = X1 = the per-row Xh, the members' rows from xrows), so a stage holds them once.

`prompt_spill`: acc (this warp's m tiles, members PM / 2 * hf .. + PM / 2 - 1) -> E[member - PM / 2 * hf][col]
fp32, row stride ES; col0: the first m tile's first column within the block's E columns.

`prompt_smem`: max of: gate/up stages (XM copies of the member rows), down stages, gate/up epilogue half
(2 x PM/2 x GU_ES), down half (PM/2 x DN_ES).

## prompt-order

The launch order of a prompt plan's items (ORDER): the items of experts that take one item (their weights come
from DRAM, few mma) and of experts that take several (their passes run together and share the weights through L2:
mostly mma) merged evenly, each list in plan order, so DRAM and the tensor cores stay busy together instead of in
turns. Only which block runs which item changes: every output keeps its bits. The shared expert's items (id E, the
plan's last) are left out: order[slot] = item for slot < the routed items, -1 after. One block.

## instantiations

The instantiations GLM-5.3-Flash at two ranks launches (glm5_next/cuda/exl3_mm.py routed): a decode window's gate/up
and down projections (NT 8, 4 warps, 16 k steps a warp, 16-byte loads one step ahead: TF_GLM_EXL3_LOADS=nc; the
down projection with its epilogue fused), the row rotation, and a prompt chunk's passes of 64 pairs.
