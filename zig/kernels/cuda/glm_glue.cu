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
  // ties at the threshold, lowest columns first: a tile of blockDim.x columns at a time, each tie ranked in column order by a
  // warp ballot and the warps' counts (the same pairs as a serial scan from column 0, without one thread walking the row)
  __shared__ int warp_ties[32];
  __shared__ int tie_at, tie_left;
  if (threadIdx.x == 0) {
    tie_at = written;
    tie_left = need_eq;
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int base = 0; base < vocab && tie_left > 0; base += blockDim.x) {
    const int i = base + threadIdx.x;
    const bool tie = i < vocab && tf_glm_key(row[i]) == threshold;
    const unsigned mask = __ballot_sync(0xffffffffu, tie);
    if (lane == 0) warp_ties[warp] = __popc(mask);
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) before += warp_ties[w];
    const int rank = before + __popc(mask & ((1u << lane) - 1u));
    if (tie && rank < tie_left) {
      o[2 * (tie_at + rank)] = __float_as_uint(__uint_as_float((uint32_t)row[i] << 16));
      o[2 * (tie_at + rank) + 1] = (uint32_t)(i + offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int total = 0;
      for (int w = 0; w < (int)(blockDim.x >> 5); ++w) total += warp_ties[w];
      const int took = total < tie_left ? total : tie_left;
      tie_at += took;
      tie_left -= took;
    }
    __syncthreads();
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

// -- DFlash2's glue (glm5_next/cuda/dflash2.py): the torch ops between its Triton launches --------------------------

// `self.kc[i].index_copy_(1, idx, k)`: rows [kvh, rows, hd] into a cache [kvh, cap, hd] at the positions pos + r (modulo the
// ring). grid (rows, kvh), block 128.
extern "C" __global__ void tf_glm_df_scatter(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ cache,
                                             const int64_t* __restrict__ pos, int rows, int cap, int ring, int hd) {
  const int r = blockIdx.x, h = blockIdx.y;
  int64_t at = pos[0] + r;
  if (ring) at %= ring;
  for (int d = threadIdx.x; d < hd; d += blockDim.x)
    cache[((int64_t)h * cap + at) * hd + d] = src[((int64_t)h * rows + r) * hd + d];
}

// dflash2_multi's `self.kc[i].index_copy_(1, idx, k)`: the rows [kvh, rows, hd] of blocks of n rows side by side, block s
// into its slot of the pooled caches (`cache` is slot 0; slot[s] counts rows of cap, the kv heads cap apart) at the positions
// pos[s] + r (modulo the ring). grid (rows, kvh), block 128.
extern "C" __global__ void tf_glm_df_scatter_seg(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ cache,
                                                 const int64_t* __restrict__ pos, const int64_t* __restrict__ slot, int rows, int n,
                                                 int cap, int ring, int hd) {
  const int r = blockIdx.x, h = blockIdx.y, s = r / n;
  int64_t at = pos[s] + r % n;
  if (ring) at %= ring;
  const int64_t row = slot[s] * cap + (int64_t)h * cap + at;
  for (int d = threadIdx.x; d < hd; d += blockDim.x) cache[row * hd + d] = src[((int64_t)h * rows + r) * hd + d];
}

// `_rotary`: phase = float(pos + r) * inv_freq, cos and sin in fp32, [rows, half]. grid (ceil(rows half / 128)), block 128.
extern "C" __global__ void tf_glm_df_rope(const int64_t* __restrict__ pos, const float* __restrict__ inv_freq, int rows, int half,
                                          float* __restrict__ cos_out, float* __restrict__ sin_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * half) return;
  const float phase = (float)(pos[0] + i / half) * inv_freq[i % half];
  cos_out[i] = cosf(phase);
  sin_out[i] = sinf(phase);
}

// `self.pos_dev += n`. one thread.
extern "C" __global__ void tf_glm_df_pos_add(int64_t* __restrict__ pos, int n) { pos[0] += n; }

// `_row`: the ranks' fp32 partials [world, count] added in rank order, then bf16. grid (ceil(count / 256)), block 256.
extern "C" __global__ void tf_glm_df_rowsum(const float* __restrict__ g, int world, int count, __nv_bfloat16* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float acc = g[i];
  for (int w = 1; w < world; ++w) acc += g[(int64_t)w * count + i];
  out[i] = __float2bfloat16(acc);
}

// `tap_rows`: five taps [n, d] side by side, [n, 5 d]. grid (n), block 256.
extern "C" __global__ void tf_glm_df_cat(const __nv_bfloat16* __restrict__ t0, const __nv_bfloat16* __restrict__ t1,
                                         const __nv_bfloat16* __restrict__ t2, const __nv_bfloat16* __restrict__ t3,
                                         const __nv_bfloat16* __restrict__ t4, int d, __nv_bfloat16* __restrict__ out) {
  const __nv_bfloat16* t[5] = {t0, t1, t2, t3, t4};
  const int r = blockIdx.x;
  for (int c = threadIdx.x; c < 5 * d; c += blockDim.x) out[(int64_t)r * 5 * d + c] = t[c / d][(int64_t)r * d + c % d];
}

// `F.linear(h, hproj).float()`: h [rows, k] times w [n, k] transposed, fp32 sums, rounded to bf16 and back. One warp an
// output. grid (ceil(n / 8), rows), block 256.
extern "C" __global__ void tf_glm_df_proj(const __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ w,
                                          float* __restrict__ out, int n, int k) {
  const int col = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32, row = blockIdx.y;
  if (col >= n) return;
  float acc = 0.f;
  for (int i = lane; i < k; i += 32) acc += __bfloat162float(h[(int64_t)row * k + i]) * __bfloat162float(w[(int64_t)col * k + i]);
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) out[(int64_t)row * n + col] = __bfloat162float(__float2bfloat16(acc));
}

// -- the top_p nucleus draw's device side (cuda/sampling.py nucleus_rows, TENSORFOLD_NUCLEUS_UNION): per row, the mass floor(exp(v - top)
// 2^40) of every token in fixed point (v = logit * (1 / max(T, 1e-6)) in float64 as torch's tensor / scalar computes it, top the
// maximum over the ranks' row maxima, which `all` holds as tf_glm_greedy wrote them, gathered: [world][rows][value bits, id]),
// the shard's sum, and the top `count` candidates' (value, id, mass). No fused multiply-add (the module's flags), the same
// double exp as torch's CUDA kernel.
__device__ __forceinline__ double tf_nuc_top(const uint32_t* all, int world, int rows, int r, double inv_t) {
  double top = -INFINITY;
  for (int k = 0; k < world; ++k) {
    const double x = (double)__uint_as_float(all[((int64_t)k * rows + r) * 2]) * inv_t;
    top = x > top ? x : top;
  }
  return top;
}

// grid (rows), block 1024: sums[r] = the shard's mass total (an integer: the order of the sum does not matter).
extern "C" __global__ void tf_glm_nuc_stats(const unsigned short* __restrict__ logits, int vocab, int ld, double inv_t,
                                            const uint32_t* __restrict__ all, int world, int rows,
                                            unsigned long long* __restrict__ sums) {
  __shared__ unsigned long long red[1024];
  const int r = blockIdx.x;
  const unsigned short* row = logits + (int64_t)r * ld;
  const double top = tf_nuc_top(all, world, rows, r, inv_t);
  unsigned long long acc = 0;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const double x = (double)__uint_as_float((uint32_t)row[i] << 16) * inv_t;
    acc += (unsigned long long)floor(exp(x - top) * 1099511627776.0);
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) sums[r] = red[0];
}

// grid (ceil(count / 256), rows), block 256: out[r] = [count value bits (float64)][count ids][count masses][sum][width] as 64-bit
// words, from tf_glm_topk's (value bits, id) pairs `cand` [rows][count][2].
extern "C" __global__ void tf_glm_nuc_pack(const uint32_t* __restrict__ cand, int count, double inv_t, const uint32_t* __restrict__ all,
                                           int world, int rows, const unsigned long long* __restrict__ sums, int width,
                                           unsigned long long* __restrict__ out) {
  const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long* o = out + (int64_t)r * (3 * count + 2);
  if (i < count) {
    const uint32_t* c = cand + ((int64_t)r * count + i) * 2;
    const double x = (double)__uint_as_float(c[0]) * inv_t;
    const double top = tf_nuc_top(all, world, rows, r, inv_t);
    o[i] = (unsigned long long)__double_as_longlong(x);
    o[count + i] = (unsigned long long)c[1];
    o[2 * count + i] = (unsigned long long)floor(exp(x - top) * 1099511627776.0);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    o[3 * count] = sums[r];
    o[3 * count + 1] = (unsigned long long)width;
  }
}
