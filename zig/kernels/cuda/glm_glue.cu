// GLM's torch glue between kernels, as small exact kernels (ours): what the Python engine does with torch ops
// between its Triton and CUDA launches (v0.6.6 families/glm5_next/cuda): copies, the sparse token lists and the
// greedy pick. Every result is a copy or an integer, so bits cannot differ from torch's.

#include <cuda_bf16.h>
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
