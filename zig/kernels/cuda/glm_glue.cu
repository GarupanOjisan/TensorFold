// GLM's torch glue between kernels, as small exact kernels (ours): what the Python engine does with torch ops
// between its Triton and CUDA launches (v0.6.6 families/glm5_next/cuda): copies, the sparse token lists and the
// greedy pick. Every result is a copy or an integer, so bits cannot differ from torch's.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// rows of `bytes` bytes (a multiple of 4) from src (row stride src_ld bytes) to dst (dst_ld): forward.py:407's
// `b.ey[:R, top_k].copy_(b.sy[:R])`. grid (rows), block 256.
extern "C" __global__ void tf_glm_copy_rows(const uint32_t* __restrict__ src, uint64_t src_ld, uint32_t* __restrict__ dst,
                                            uint64_t dst_ld, uint64_t bytes) {
  const uint64_t r = blockIdx.x;
  const uint32_t* s = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(src) + r * src_ld);
  uint32_t* d = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(dst) + r * dst_ld);
  for (uint64_t i = threadIdx.x; i < bytes / 4; i += blockDim.x) d[i] = s[i];
}

// sparse.py:204-215: each row's selected token list and count from its 512 best pools (int64, ascending).
// q = pos + r; npool = (q + 1) / 4; tokens[r, 4i + j] = pools[r, i] * 4 + j; then the 3 tail tokens past the last
// whole pool if visible (else -1); count = npool > 512 ? 2048 + visible tails : 0. grid (rows), block 256.
extern "C" __global__ void tf_glm_tokens(const int64_t* __restrict__ pools, const int* __restrict__ pos, int* __restrict__ tokens,
                                         int* __restrict__ counts, int width, int k) {
  const int r = blockIdx.x;
  const int64_t q = (int64_t)pos[0] + r;
  const int64_t npool = (q + 1) / 4;
  int* row = tokens + (int64_t)r * width;
  for (int i = threadIdx.x; i < k * 4; i += blockDim.x) row[i] = (int)(pools[(int64_t)r * k + i / 4] * 4 + (i % 4));
  if (threadIdx.x == 0) {
    int visible = 0;
    for (int j = 0; j < 3; ++j) {
      const int64_t tail = npool * 4 + j;
      const bool ok = tail <= q;
      row[k * 4 + j] = ok ? (int)tail : -1;
      visible += ok;
    }
    counts[r] = npool > k ? k * 4 + visible : 0;
  }
}

// decode.py:34-49 for greedy rows on one rank: torch.topk(logits.float(), 1) gives the row's maximum and its first
// column; out[r] = (value as fp32 bits, column + offset) for the all-gather. grid (rows), block 1024.
extern "C" __global__ void tf_glm_greedy(const __nv_bfloat16* __restrict__ logits, int vocab, int ld, int offset,
                                         uint32_t* __restrict__ out) {
  __shared__ float best_v[1024];
  __shared__ int best_i[1024];
  const int r = blockIdx.x;
  const __nv_bfloat16* row = logits + (int64_t)r * ld;
  float v = -INFINITY;
  int at = 0x7fffffff;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float x = __bfloat162float(row[i]);
    if (x > v || (x == v && i < at)) {
      v = x;
      at = i;
    }
  }
  best_v[threadIdx.x] = v;
  best_i[threadIdx.x] = at;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = best_v[threadIdx.x + s];
      const int oi = best_i[threadIdx.x + s];
      if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
        best_v[threadIdx.x] = ov;
        best_i[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[2 * r] = __float_as_uint(best_v[0]);
    out[2 * r + 1] = (uint32_t)(best_i[0] + offset);
  }
}

// decode.py:31-37 for sampled rows on one rank: torch.topk(logits.float(), k) of the rank's vocabulary slice, as
// (value as fp32 bits, column + offset) pairs; which pairs, not their order, matters (the host chooser sorts them).
// The k-th largest bf16 key is found in two 8-bit radix passes; columns above it are all taken, columns equal to it
// in column order until k. grid (rows), block 1024, k <= 1024.
__device__ __forceinline__ uint32_t tf_glm_key(unsigned short b) {
  return (b & 0x8000) ? (uint32_t)(~b & 0xffff) : (uint32_t)(b | 0x8000);
}

extern "C" __global__ void tf_glm_topk(const unsigned short* __restrict__ logits, int vocab, int ld, int k, int offset,
                                       uint32_t* __restrict__ out) {
  __shared__ int hist[256];
  __shared__ int sel_hi, sel_lo, above_hi, need_eq, written;
  const int r = blockIdx.x;
  const unsigned short* row = logits + (int64_t)r * ld;
  uint32_t* o = out + (int64_t)r * 2 * k;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) atomicAdd(&hist[tf_glm_key(row[i]) >> 8], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    int above = 0, b = 255;
    while (b > 0 && above + hist[b] < k) above += hist[b--];
    sel_hi = b;
    above_hi = above;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const uint32_t key = tf_glm_key(row[i]);
    if ((int)(key >> 8) == sel_hi) atomicAdd(&hist[key & 0xff], 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int above = above_hi, b = 255;
    while (b > 0 && above + hist[b] < k) above += hist[b--];
    sel_lo = b;
    need_eq = k - above;
    written = 0;
  }
  __syncthreads();
  const uint32_t threshold = ((uint32_t)sel_hi << 8) | (uint32_t)sel_lo;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const uint32_t key = tf_glm_key(row[i]);
    if (key > threshold) {
      const int at = atomicAdd(&written, 1);
      o[2 * at] = __float_as_uint(__uint_as_float((uint32_t)row[i] << 16));
      o[2 * at + 1] = (uint32_t)(i + offset);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {   // ties at the threshold, lowest columns first
    int at = written, left = need_eq;
    for (int i = 0; i < vocab && left > 0; ++i) {
      if (tf_glm_key(row[i]) == threshold) {
        o[2 * at] = __float_as_uint(__uint_as_float((uint32_t)row[i] << 16));
        o[2 * at + 1] = (uint32_t)(i + offset);
        ++at;
        --left;
      }
    }
  }
}

// qmm.quantize4 (v0.6.6 glm5_next/cuda/qmm.py:158): bf16 (n, k) -> MLX affine 4-bit groups of 64, one thread a group.
// scale = bf16(max((hi - lo) / 15, 1e-8)), bias = bf16(lo), q = round-half-even((w - bias) / scale) in 0..15, input
// 8 i + j of a group at bits 4 j of its word i; fp32 arithmetic as torch's. torch's CUDA tensor / Python scalar multiplies
// by the fp32 reciprocal (BinaryDivTrueKernel.cu), tensor / tensor divides correctly rounded.
extern "C" __global__ void tf_glm_quantize4(const __nv_bfloat16* __restrict__ w, int n, int k, uint32_t* __restrict__ words,
                                            __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ biases) {
  const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int kg = k / 64;
  if (g >= (long long)n * kg) return;
  const __nv_bfloat16* x = w + g * 64;   // row g / kg, group g % kg
  float lo = __bfloat162float(x[0]), hi = lo;
  for (int i = 1; i < 64; ++i) {
    const float v = __bfloat162float(x[i]);
    lo = fminf(lo, v);
    hi = fmaxf(hi, v);
  }
  const __nv_bfloat16 sb = __float2bfloat16_rn(fmaxf(__fmul_rn(hi - lo, 1.0f / 15.0f), 1e-8f));
  const __nv_bfloat16 bb = __float2bfloat16_rn(lo);
  const float s = __bfloat162float(sb), b = __bfloat162float(bb);
  for (int wi = 0; wi < 8; ++wi) {
    uint32_t word = 0;
    for (int j = 0; j < 8; ++j) {
      const float q = fminf(fmaxf(rintf(__fdiv_rn(__fsub_rn(__bfloat162float(x[wi * 8 + j]), b), s)), 0.0f), 15.0f);
      word |= (uint32_t)q << (4 * j);
    }
    words[g * 8 + wi] = word;
  }
  scales[g] = sb;
  biases[g] = bb;
}

// The squared error of one group of 64 under scale s and bias b, summed in torch's order for `(...).sum(-1)` over 64
// contiguous fp32 values (Reduce.cuh: 32 lanes, lane t adds x[t] and x[t + 32], then shfl_down sums at offsets
// 16, 8, 4, 2, 1; checked bit for bit against torch 2.13 on this checkpoint's groups): every op rounds, as torch's
// separate kernels do (q * s, + b, - g, squared).
__device__ __forceinline__ float group_error(const float (&g)[64], float s, float b) {
  float l[32];
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    float e[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float v = g[t + 32 * h];
      const float q = fminf(fmaxf(rintf(__fdiv_rn(__fsub_rn(v, b), s)), 0.0f), 15.0f);
      const float d = __fsub_rn(__fadd_rn(__fmul_rn(q, s), b), v);
      e[h] = __fmul_rn(d, d);
    }
    l[t] = __fadd_rn(e[0], e[1]);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
#pragma unroll
    for (int t = 0; t < off; ++t) l[t] = __fadd_rn(l[t], l[t + off]);
  return l[0];
}

// qmm.quantize4_mse_raw (v1.8 glm5_next/cuda/qmm.py:179), TF_GLM_DENSE=q4: groups of 64 as quantize4, each group's range
// the one of 16 clippings a = 1 - 0.45 i / 15 of its min and max with the least squared error (the first on a tie),
// scale and bias stored as bf16. a is a Python float, cast to fp32 where torch multiplies the fp32 tensor by it.
// MLX layout out, as tf_glm_quantize4.
extern "C" __global__ void tf_glm_quantize4_mse(const __nv_bfloat16* __restrict__ w, int n, int k, uint32_t* __restrict__ words,
                                                __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ biases) {
  const long long gi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int kg = k / 64;
  if (gi >= (long long)n * kg) return;
  const __nv_bfloat16* x = w + gi * 64;
  float g[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) g[i] = __bfloat162float(x[i]);
  float lo0 = g[0], hi0 = g[0];
#pragma unroll
  for (int i = 1; i < 64; ++i) {
    lo0 = fminf(lo0, g[i]);
    hi0 = fmaxf(hi0, g[i]);
  }
  float best_err = 0.0f, best_s = 0.0f, best_b = 0.0f;
  __nv_bfloat16 best_sb, best_bb;
  for (int i = 0; i < 16; ++i) {
    const float a = (float)(1.0 - 0.45 * (double)i / 15.0);
    const float lo = __fmul_rn(lo0, a), hi = __fmul_rn(hi0, a);
    const __nv_bfloat16 sb = __float2bfloat16_rn(fmaxf(__fmul_rn(__fsub_rn(hi, lo), 1.0f / 15.0f), 1e-8f));
    const __nv_bfloat16 bb = __float2bfloat16_rn(lo);
    const float s = __bfloat162float(sb), b = __bfloat162float(bb);
    const float err = group_error(g, s, b);
    if (i == 0 || err < best_err) {
      best_err = err;
      best_s = s;
      best_b = b;
      best_sb = sb;
      best_bb = bb;
    }
  }
#pragma unroll
  for (int wi = 0; wi < 8; ++wi) {
    uint32_t word = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float q = fminf(fmaxf(rintf(__fdiv_rn(__fsub_rn(g[wi * 8 + j], best_b), best_s)), 0.0f), 15.0f);
      word |= (uint32_t)q << (4 * j);
    }
    words[gi * 8 + wi] = word;
  }
  scales[gi] = best_sb;
  biases[gi] = best_bb;
}

// qmm.make_f8 (v1.8 qmm.py:412): bf16 (n, k) -> e4m3 (n, k) and an fp32 scale per row and 128 columns, one thread a block:
// scale = max(absmax * (1 / 448), 1e-30) (torch multiplies by the reciprocal of a Python scalar), value =
// e4m3(clamp(w / scale, -448, 448)), round to nearest even.
extern "C" __global__ void tf_glm_make_f8(const __nv_bfloat16* __restrict__ w, int n, int k, uint8_t* __restrict__ q8,
                                          float* __restrict__ scale) {
  const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int kb = k / 128;
  if (b >= (long long)n * kb) return;
  const __nv_bfloat16* x = w + b * 128;
  float amax = 0.0f;
  for (int i = 0; i < 128; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(x[i])));
  const float s = fmaxf(__fmul_rn(amax, 1.0f / 448.0f), 1e-30f);
  for (int i = 0; i < 128; ++i) {
    const float v = fminf(fmaxf(__fdiv_rn(__bfloat162float(x[i]), s), -448.0f), 448.0f);
    q8[b * 128 + i] = (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  }
  scale[b] = s;
}
