// Device code of src/tensorfold/families/glm5_next/cuda/exl3.cu [v1.8] (lines 1-912, ATen includes dropped), checked by zig/tests/cuda/copies.py.
#include <stdint.h>
// GLM-5.3-Flash's EXL3 routed experts on CUDA: the grouped trellis GEMV and the Hadamard rotations around it.
//
// Format (see exl3.py, after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): a 16x16 tile is 32 little-endian
// 32-bit words; lane L of a warp decodes the tile's values 8L..8L+7 from words L-1 and L, and those eight values
// are exactly the B fragments of two mma.m16n8k16 (columns 0-7 and 8-15 of the tile), so a tile goes from memory
// to the tensor cores without a shuffle or a layout change.
//
// Every output depends only on its own row: the rows of a window share an mma tile but mma keeps rows
// independent, the K range of every warp and split is fixed by the shape, and warps and splits are summed in a
// fixed order (shared memory, then the epilogue kernels), never with atomics.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "exl3_decode.cuh"

namespace tf_glm_exl3d {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Two values of the "mcg" codebook from two 16-bit states, as a half2 (first state in .x).
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// This lane's eight values of a 4-bit tile (word = tile[lane]) as the B fragments of its two n8 halves.
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

// decode_tile's values with fewer integer instructions (the prompt kernels): the byte-aligned 16-bit states taken
// with one PRMT each and w >> 4 shared by two states. The same states, so the same bits.
__device__ __forceinline__ void decode_tile_p(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    const uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    const uint32_t s = __funnelshift_r(w, p, 20);
    const uint32_t t = w >> 4;
    b0[0] = mcg2(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu);          // (s >> 8) & 0xffff, (s >> 4) & 0xffff
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2(__byte_perm(t, 0u, 0x4421), __byte_perm(w, 0u, 0x4421));   // (w >> 12) & 0xffff, (w >> 8) & 0xffff
    b1[1] = mcg2(__byte_perm(t, 0u, 0x4410), w & 0xffffu);                  // (w >> 4) & 0xffff, w & 0xffff
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

// 16 bytes of trellis words, read once a window (no L1 allocation); asm volatile keeps it where it stands relative
// to the (volatile) mma, so the ring depth is what the code says (patches/0580 of jayleaton/glm53-tensorfold-spark).
__device__ __forceinline__ uint4 ldg_nc_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// Grid (item, n block, mat * SK + split) -> Z[mat][split][pair][n]; warps sum fixed k ranges, added in warp order.
// SEQ (prompt chunks): grid (.., .., mat); one program runs the SK splits in turn and writes Z[mat][0][pair][n] =
// ((0 + split 0) + split 1) + ..., the epilogue's own sum, so an SK = 1 epilogue gives the same bits without the
// SK fp32 partials' round trip through memory.
template <int NT, int W, bool NFIRST, bool PF, bool SEQ>
__global__ void __launch_bounds__(W * 32) grouped_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E) {
    // NFIRST: the n blocks of an item are adjacent in launch order, so they read its rows while L2 holds them
    const int item = NFIRST ? blockIdx.y : blockIdx.x;
    const int nblock = NFIRST ? blockIdx.x : blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = SEQ ? blockIdx.z : blockIdx.z / SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int nt0 = nblock * NT;
    constexpr int OUT = 16 * NT * 16 / (W * 32);        // the sums a thread writes (SEQ keeps them in registers)
    float tot[OUT];
    __shared__ float red[W][16][NT * 16];
    for (int si = 0; si < (SEQ ? SK : 1); ++si) {
    const int split = SEQ ? si : blockIdx.z % SK;
    const int kt0 = split * per_split + warp * per_warp;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if (PF) {
        // the next k step's words and A fragments load while this one decodes and multiplies (the same mma order)
        uint32_t words[NT], a[4];
#pragma unroll
        for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);
        {
            const int k = kt0 * 16;
            a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
            a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
            uint32_t nw[NT], na[4];
            const bool more = kt + 1 < kt0 + per_warp;
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
    } else {
        for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
            uint32_t words[NT];
#pragma unroll
            for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);
            const int k = kt * 16;
            uint32_t a[4] = {load_pair(x0 + k, r0 >= 0), load_pair(x1 + k, r1 >= 0),
                             load_pair(x0 + k + 8, r0 >= 0), load_pair(x1 + k + 8, r1 >= 0)};
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                uint32_t b0[2], b1[2];
                decode_tile(words[i], lane, b0, b1);
                mma16816(acc[i][0], a, b0);
                mma16816(acc[i][1], a, b1);
            }
            tile += (size_t)NTILES * 32;
        }
    }

    // warps' partial sums through shared memory, added in warp order
    if (SEQ && si > 0) __syncthreads();                // the previous split's sums are read
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
#pragma unroll
    for (int j = 0; j < OUT; ++j) {
        const int idx = threadIdx.x + j * W * 32;
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        if (SEQ) {
            tot[j] = si == 0 ? 0.f + s : tot[j] + s;
        } else if (r >= 0) {
            Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
    }
    }  // splits
    if (SEQ) {
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * W * 32;
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[row];
            if (r >= 0) Z[((size_t)mat * P + r) * N + nt0 * 16 + col] = tot[j];
        }
    }
}

// Fast Walsh-Hadamard transform of 128 values held 4 per lane (lane L: values 4L..4L+3), natural order, fixed
// butterfly order: strides 1, 2 in registers, 4..64 across lanes.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// Program (member row, 128-block of K, matrix): Xh[mat][row][block] = fp16((x[row] * suh[mat][e]) @ H) for the
// gate and up projections of every routed slot (row, slot) with slot < slots - 1 (the last slot is the shared
// expert). One warp per program. ROWS (prompt chunks of a layer whose experts all share one suh for gate and up):
// program (row, block) writes Xh[row][block] from suh0 = that vector, the same arithmetic as every pair's input of
// that row, so each pair's fp16 input has the same bits, computed once a row instead of 2 x top_k times.
template <bool ROWS>
__global__ void rot_in_kernel(const __nv_bfloat16* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    int row, e;
    if (ROWS) {
        row = p;
        e = 0;
    } else {
        row = p / slots;
        const int slot = p % slots;
        if (slot == slots - 1) return;
        e = pick[row * slots + slot];
    }
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Program (member row, 128-block of the rank's intermediate width): gate and up outputs summed over the splits in
// order, rotated, scaled by svh; GLM's limited SwiGLU with the grouped kernels' bf16 roundings; then the down
// projection's input rotation: Xd[row][block] = fp16((act * suh_d[e]) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int slots, float limit) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x;
    const int e = pick[row * slots + slot];
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
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
        float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit), limit);
        float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): the down projection's output summed over the splits in
// order, rotated and scaled by svh: Y[row][slot][:] (fp32, this rank's share of the expert's output).
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int slots) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x;
    const int e = pick[row * slots + slot];
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

#include "glm_exl3d_dec.cuh"

#include "glm_exl3d_prompt.cuh"

}  // namespace tf_glm_exl3d

// The instantiations GLM-5.3-Flash at two ranks launches (glm5_next/cuda/exl3_mm.py routed): a decode window's gate/up
// and down projections (NT 8, 4 warps, 16 k steps a warp, 16-byte loads one step ahead: TF_GLM_EXL3_LOADS=nc; the
// down projection with its epilogue fused), the row rotation, and a prompt chunk's passes of 64 pairs.
#define TF_DEC(FUSE, XROW, SWZ, B)                                                                                  \
    template __global__ void tf_glm_exl3d::dec_kernel<8, 4, 16, FUSE, XROW, 1, 1, SWZ, B>(                          \
        const half*, const half*, const uint32_t*, const uint32_t*, const int*, const int*, const int*, float*,   \
        int, int, int, int, int, int, const half*, const half*, const half*, void*, float, int*);
TF_DEC(2, true, 0, 4)
TF_DEC(2, true, 0, 3)
TF_DEC(2, false, 0, 4)
TF_DEC(2, false, 0, 3)
TF_DEC(2, true, 1, 4)
TF_DEC(2, true, 1, 3)
TF_DEC(2, false, 1, 4)
TF_DEC(2, false, 1, 3)
TF_DEC(2, true, 2, 4)
TF_DEC(2, true, 2, 3)
TF_DEC(2, false, 2, 4)
TF_DEC(2, false, 2, 3)
TF_DEC(0, true, 0, 4)
TF_DEC(0, true, 0, 3)
TF_DEC(0, false, 0, 4)
TF_DEC(0, false, 0, 3)
TF_DEC(1, false, 0, 4)
TF_DEC(1, false, 0, 3)
TF_DEC(0, true, 1, 4)
TF_DEC(0, true, 1, 3)
TF_DEC(0, false, 1, 4)
TF_DEC(0, false, 1, 3)
TF_DEC(1, false, 1, 4)
TF_DEC(1, false, 1, 3)
TF_DEC(0, true, 2, 4)
TF_DEC(0, true, 2, 3)
TF_DEC(0, false, 2, 4)
TF_DEC(0, false, 2, 3)
TF_DEC(1, false, 2, 4)
TF_DEC(1, false, 2, 3)
#undef TF_DEC

#define TF_ROT(ROWS)                                                                                               \
    template __global__ void tf_glm_exl3d::rot_in_kernel<ROWS>(const __nv_bfloat16*, int, const int*, const half*, \
                                                              const half*, half*, half*, int, int);
TF_ROT(true)
TF_ROT(false)
#undef TF_ROT

template __global__ void tf_glm_exl3d::prompt_gateup_kernel<64, 8, true, 4>(
    const half*, const half*, const uint32_t*, const uint32_t*, const int*, const int*, const int*, const half*,
    const half*, const half*, half*, int, int, int, float, int, const int*);
template __global__ void tf_glm_exl3d::prompt_gateup_kernel<64, 8, false, 4>(
    const half*, const half*, const uint32_t*, const uint32_t*, const int*, const int*, const int*, const half*,
    const half*, const half*, half*, int, int, int, float, int, const int*);
template __global__ void tf_glm_exl3d::prompt_down_kernel<64, 8, 4>(const half*, const uint32_t*, const int*, const int*,
                                                                 const int*, const half*, float*, int, int, int,
                                                                 const int*);
template __global__ void tf_glm_exl3d::prompt_gateup_kernel<64, 8, true, 3>(
    const half*, const half*, const uint32_t*, const uint32_t*, const int*, const int*, const int*, const half*,
    const half*, const half*, half*, int, int, int, float, int, const int*);
template __global__ void tf_glm_exl3d::prompt_gateup_kernel<64, 8, false, 3>(
    const half*, const half*, const uint32_t*, const uint32_t*, const int*, const int*, const int*, const half*,
    const half*, const half*, half*, int, int, int, float, int, const int*);
template __global__ void tf_glm_exl3d::prompt_down_kernel<64, 8, 3>(const half*, const uint32_t*, const int*, const int*,
                                                                 const int*, const half*, float*, int, int, int,
                                                                 const int*);
