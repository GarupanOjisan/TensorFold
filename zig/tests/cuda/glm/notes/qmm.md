# qmm.cu notes

## instantiations

The instantiations a decode window launches (rows <= 16, groups of 64; qmm_cfg tile configs 0, 2 and 4: the default
16 x 64 tile and glm5_next/cuda/qmm.py DEC_TILES' 16 x 128 on 8 warps / 4 stages and on 4 warps / 6 stages), bf16 or
fp32 out, K slices in a thread block cluster (SK > 1, sm_90+: qmm_clusters) or one slice.
