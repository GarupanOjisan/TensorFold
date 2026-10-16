// glm_exl3d.cu's decode windows (dec_kernel), included inside namespace tf_glm_exl3d.
// ---- decode windows: grouped_kernel's arithmetic with its epilogues fused -------------------------------------
// Every mma chain, warp sum and split sum is grouped_kernel's (non-SEQ, register prefetch of the next k step): a
// warp runs k tiles kt0 .. kt0 + STEPS - 1 of its split in order from zero, the warps are added in warp order, the
// splits in split order from 0. What changes:
// FUSE 1 (the down projection, SK = 1): the block's 128 columns are one Hadamard block, so it applies
//   down_epilogue_kernel's arithmetic to its own sums and writes Y, no Z round trip, no epilogue launch;
// FUSE 2 (gate and up): each block still writes its Z partial; the last of the 2 * SK blocks of an (item, n block)
//   to finish (an atomic count, reset for the next launch) runs gateup_epilogue_kernel's arithmetic on the item's
//   rows and 128 columns, reading the partials in split order: the same sums whichever block finishes last.
// XROW: X holds one row a token (rot_rows, a layer whose experts share one suh), pair p reads row p / slots.
// LD (TF_GLM_EXL3_LOADS; data movement only, every chain and sum as above, so the same bits):
//   0: 32-bit __ldg of word `lane` of each tile, one k step ahead, after the count -> item -> members round trips;
//   1: adapted from jayleaton/glm53-tensorfold-spark patches/0580 (exl3_ld.cu ld_kernel, LD_NC; Apache-2.0,
//      Copyright 2026 Jay Leaton). Changes: in our dec_kernel (its items / counts plan, fused epilogues and XROW
//      inputs kept), PD fixed at compile time, no probes / PDL. The count and the item are read together and the
//      first PD k steps' trellis words are issued before the member rows reach shared memory; the words come as
//      ld.global.nc.L1::no_allocate.v4 (lane l: bytes 16 l .. of each 512 B of a step, NT / 4 loads a lane), PD
//      steps in flight, through a 1 KB staging area in the warp's own slice of `red` (two __syncwarp a step), read
//      back as word `lane` of each tile.
template <int NT, int W, int STEPS, int FUSE, bool XROW, int LD = 0, int PD = 1, int SWZ = 0, int B = 4>
__global__ void __launch_bounds__(W * 32) dec_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(FUSE == 0 || NT * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(LD == 0 || (NT % 4 == 0 && PD >= 1 && STEPS % PD == 0), "LD 1: NT a multiple of 4, PD | STEPS");
    static_assert(B == 4 || (B == 3 && LD == 1 && tf_exl3::lane_words<2 * B>() == 2), "3-bit tiles: 16-byte loads");
    constexpr int TW = 8 * B;                            // 32-bit words a tile
    int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    if constexpr (SWZ != 0) {                            // launch order only (v1.10 patch 0090): the blocks of one k slice run together
        const unsigned lid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        nblock = (int)(lid % gridDim.y);
        const unsigned r = lid / gridDim.y;
        if (SWZ == 1) { item = (int)(r % gridDim.x); zz = (int)(r / gridDim.x); }
        else { zz = (int)(r % gridDim.z); item = (int)(r / gridDim.z); }
    }
    int e, first, cnt;
    if constexpr (LD == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = zz / SK, split = zz % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[W][16][NT * 16];
    constexpr int NV = (NT * TW + 127) / 128;            // LD 1: 16-byte loads a lane a k step
    uint4 rv[LD == 1 ? PD : 1][LD == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * STEPS) * NTILES + nblock * NT) * TW;
    const size_t kstep = (size_t)NTILES * TW;            // words from one k tile to the next
    if constexpr (LD == 1) {
#pragma unroll
        for (int d = 0; d < PD; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) if (B == 4 || 128 * v + 4 * lane < NT * TW) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (XROW ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (XROW ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * NT;
    const int kt0 = split * (KT / SK) + warp * STEPS;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (LD == 1) {
        uint32_t ra[PD][4];
#pragma unroll
        for (int d = 0; d < PD; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        int lw0 = lane, lw1 = 0, loff = 0;               // 3-bit: the lane's two words of a tile
        if constexpr (B != 4) { tf_exl3::lane_start<2 * B>(lane, lw0, loff); lw1 = (lw0 + 1) % TW; }
        (void)lw1; (void)loff;
        for (int kb = 0; kb < STEPS; kb += PD) {
#pragma unroll
            for (int d = 0; d < PD; ++d) {
                const int s = kb + d;
                const bool more = s + PD < STEPS;
                uint32_t a[4], words[NT], words1[B == 4 ? 1 : NT];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) if (B == 4 || 128 * v + 4 * lane < NT * TW) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        if (B == 4 || 128 * v + 4 * lane < NT * TW) rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + PD) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + PD) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < NT; ++i) words[i] = stage[i * TW + lw0];
                if constexpr (B != 4) {
#pragma unroll
                    for (int i = 0; i < NT; ++i) words1[i] = stage[i * TW + lw1];
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    if constexpr (B == 4) {
                        decode_tile(words[i], lane, b0, b1);
                    } else {
                        const uint32_t w2[2] = {words[i], words1[i]};
                        tf_exl3::decode_lane<2 * B, tf_exl3::CB_MCG>(w2, lane, b0, b1);
                    }
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[NT], a[4];
#pragma unroll
    for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + STEPS; ++kt) {
        uint32_t nw[NT], na[4];
        const bool more = kt + 1 < kt0 + STEPS;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < NT; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < NT; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (FUSE == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += W) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < W; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * NT * 16 / (W * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * W * 32;
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (FUSE == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.y + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += W) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}
