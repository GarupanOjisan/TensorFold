// Prompt chunks of glm_exl3d.cu: Y^T = W^T X^T, one fp32 mma chain an output; notes/glm_exl3d.md#prompt-chunks
constexpr int PBK = 32;              // k values a shared-memory stage (2 k tiles)
constexpr int PKT = PBK / 16;
constexpr int PSTAGES = 3;
constexpr int PXS = PBK + 8;         // stage row stride in halves (+16 B: ldmatrix rows on distinct banks)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

// A block: 16 m tiles for a pass of PM members, NW warps, PSTAGES cp.async stages; notes/glm_exl3d.md#prompt-block
constexpr int PWORDS = 16 * PKT * 32;          // a stage's trellis words (16 m tiles x PKT k tiles x 32)

template <int MATS, int PM, int NW, bool SHX = false>
struct PromptCfg {
    static constexpr int MT = 16 / NW, NB = PM / 8, T = NW * 32;
    static constexpr int XM = SHX ? 1 : MATS;               // member-row copies a stage holds
    static constexpr size_t STAGE_HALVES = (size_t)XM * PM * PXS + 2 * PWORDS;
};

// Which (matrix, m tile) warp-tile q (0..15) of block cb is.
template <int MATS>
__device__ __forceinline__ void prompt_tile(int q, int cb, int& mat, int& nt) {
    mat = MATS == 2 ? q >> 3 : 0;
    nt = MATS == 2 ? cb * 8 + (q & 7) : cb * 16 + q;
}

template <int MATS, int PM, int NW, bool SHX = false, int B = 4>
__device__ __forceinline__ void prompt_mainloop(const half* __restrict__ X0, const half* __restrict__ X1,
                                                const uint32_t* __restrict__ T0, const uint32_t* __restrict__ T1,
                                                const int* rows_sh, int cnt, int e, int K, int NTILES, int cb,
                                                half* stage, float (&acc)[16 / NW][PM / 8][4]) {
    using C = PromptCfg<MATS, PM, NW, SHX>;
    constexpr int TW = 8 * B, TP = TW / 4;           // words and 16-byte pieces a tile (stage tiles stay 32 words apart)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int lw0 = lane, lw1 = 0, loff = 0;
    if constexpr (B != 4) { tf_exl3::lane_start<2 * B>(lane, lw0, loff); lw1 = (lw0 + 1) % TW; }
    (void)lw1; (void)loff;
    int mat, nt_unused;
    prompt_tile<MATS>(warp * C::MT, cb, mat, nt_unused);
    const int KT = K >> 4, chunks = K / PBK;
    const int nblocks = (cnt + 15) >> 4;                                            // 16-member blocks in use

#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;

    auto load = [&](int chunk) {
        half* dst = stage + (size_t)(chunk % PSTAGES) * C::STAGE_HALVES;
        const int k0 = chunk * PBK;
        for (int idx = threadIdx.x; idx < C::XM * PM * (PBK / 8); idx += C::T) {
            const int m = idx / (PM * (PBK / 8)), rem = idx % (PM * (PBK / 8)), i = rem / (PBK / 8),
                      j = rem % (PBK / 8);
            const int r = rows_sh[i];
            const half* src = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : r) * K + k0 + j * 8;
            cp_async16(dst + ((size_t)m * PM + i) * PXS + j * 8, src, r >= 0);
        }
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + (size_t)C::XM * PM * PXS);
        for (int idx = threadIdx.x; idx < 16 * PKT * TP; idx += C::T) {      // 16 B pieces of the tiles' words
            const int q = idx / (PKT * TP), kk = (idx / TP) % PKT, part = idx % TP;
            int wm, nt;
            prompt_tile<MATS>(q, cb, wm, nt);
            const uint32_t* src = (wm ? T1 : T0) + (((size_t)e * KT + chunk * PKT + kk) * NTILES + nt) * TW + part * 4;
            cp_async16(wdst + ((size_t)q * PKT + kk) * 32 + part * 4, src, true);
        }
    };
#pragma unroll
    for (int c = 0; c < PSTAGES - 1; ++c) {
        if (c < chunks) load(c);
        cp_async_commit();
    }
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<PSTAGES - 2>();
        __syncthreads();
        if (c + PSTAGES - 1 < chunks) load(c + PSTAGES - 1);
        cp_async_commit();
        const half* base = stage + (size_t)(c % PSTAGES) * C::STAGE_HALVES;
        const half* xs = base + (size_t)(SHX ? 0 : mat) * PM * PXS;
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(base + (size_t)C::XM * PM * PXS);
#pragma unroll
        for (int kk = 0; kk < PKT; ++kk) {
            uint32_t a[C::MT][4];
#pragma unroll
            for (int i = 0; i < C::MT; ++i) {
                uint32_t b0[2], b1[2];
                const uint32_t* tw = ws + ((size_t)(warp * C::MT + i) * PKT + kk) * 32;
                if constexpr (B == 4) {
                    decode_tile_p(tw[lane], lane, b0, b1);
                } else {
                    const uint32_t w2[2] = {tw[lw0], tw[lw1]};
                    tf_exl3::decode_lane<2 * B, tf_exl3::CB_MCG>(w2, lane, b0, b1);
                }
                a[i][0] = b0[0]; a[i][1] = b1[0]; a[i][2] = b0[1]; a[i][3] = b1[1];
            }
#pragma unroll
            for (int jb = 0; jb < PM / 16; ++jb) {
                if (jb < nblocks) {
                    uint32_t r[4];
                    const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    ldsm_x4(r, xs + (size_t)row * PXS + kk * 16 + ((lane >> 3) & 1) * 8);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
                    const bool upper = jb * 16 + 8 < cnt;         // members 8..15 of the block (else zero rows)
#pragma unroll
                    for (int i = 0; i < C::MT; ++i) {
                        mma16816(acc[i][2 * jb], a[i], bl);
                        if (upper) mma16816(acc[i][2 * jb + 1], a[i], bh);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();                                   // the stages become the epilogue's rows
}

// acc (this warp's m tiles, members PM / 2 * hf ..) -> E fp32 (row stride ES) from column col0 of the block's E
template <int MT, int PM>
__device__ __forceinline__ void prompt_spill(float* E, int ES, int col0, const float (&acc)[MT][PM / 8][4], int hf) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int q = 0; q < PM / 16; ++q) {
            const int nb = hf * (PM / 16) + q;
            const int tok = q * 8 + 2 * t, col = col0 + i * 16 + g;
            E[(size_t)tok * ES + col] = acc[i][nb][0];
            E[(size_t)(tok + 1) * ES + col] = acc[i][nb][1];
            E[(size_t)tok * ES + col + 8] = acc[i][nb][2];
            E[(size_t)(tok + 1) * ES + col + 8] = acc[i][nb][3];
        }
}

constexpr int GU_ES = 128 + 4, DN_ES = 256 + 4;
template <int PM, int XM = 2>
constexpr size_t prompt_smem() {
    // max of the gate/up stages, the down stages, the gate/up epilogue half and the down epilogue half
    const size_t gu = (size_t)PSTAGES * (XM * PM * PXS + 2 * PWORDS) * 2;
    const size_t ep = (size_t)2 * (PM / 2) * GU_ES * 4;
    const size_t dn = (size_t)(PM / 2) * DN_ES * 4;
    return gu > ep ? (gu > dn ? gu : dn) : (ep > dn ? ep : dn);
}

// ORDER: one-item and several-item experts' items merged evenly (launch order only); notes/glm_exl3d.md#prompt-order
__global__ void __launch_bounds__(1024) prompt_order_kernel(const int* __restrict__ items,
                                                            const int* __restrict__ counts, int* __restrict__ order,
                                                            int cap, int E) {
    __shared__ int wsum[32];
    __shared__ int routed;
    const int all = min(counts[0], cap);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) routed = 0;
    __syncthreads();
    for (int i = tid; i < all; i += 1024)               // items are in expert order: the routed ones come first
        if (items[3 * i] < E && (i + 1 == all || items[3 * (i + 1)] >= E)) routed = i + 1;
    __syncthreads();
    const int n = routed;
    const int per = (n + 1023) / 1024;
    const int lo = min(n, tid * per), hi = min(n, lo + per);
    auto heavy = [&](int i) {
        const int e = items[3 * i];
        return (i > 0 && items[3 * (i - 1)] == e) || (i + 1 < n && items[3 * (i + 1)] == e);
    };
    int h = 0;
    for (int i = lo; i < hi; ++i) h += heavy(i);
    int v = h;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += u;
    }
    if (lane == 31) wsum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = wsum[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += u;
        }
        wsum[lane] = w;
    }
    __syncthreads();
    const int nh = wsum[31], nl = n - nh;
    int rh = (warp ? wsum[warp - 1] : 0) + v - h;       // heavy items before lo
    for (int i = lo; i < hi; ++i) {
        if (heavy(i)) {                                // heavy rank r sits at (r + 1/2) / nh; lights before it: strictly
            const int r = rh++;
            const long long X = (2LL * r + 1) * nl;
            const int cl = X > nh ? (int)min((long long)nl, (X - nh + 2LL * nh - 1) / (2LL * nh)) : 0;
            order[r + cl] = i;
        } else {                                       // light rank l at (l + 1/2) / nl; heavies at or before it
            const int l = i - rh;
            const long long Y = (2LL * l + 1) * nh;
            const int ch = Y >= nl ? (int)min((long long)nh, (Y - nl) / (2LL * nl) + 1) : 0;
            order[l + ch] = i;
        }
    }
    for (int i = n + tid; i < all; i += 1024) order[i] = -1;
}

// SHX: Xg = Xu = Xh [rows, K], one row a token (pair p reads row p / slots); otherwise Xg, Xu [pairs, K].
template <int PM, int NW, bool SHX, int B = 4>
__global__ void __launch_bounds__(NW * 32, 16 / NW) prompt_gateup_kernel(
    const half* __restrict__ Xg, const half* __restrict__ Xu, const uint32_t* __restrict__ Tg,
    const uint32_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u,
    const half* __restrict__ suh_d, half* __restrict__ xd, int K, int N, int E, float limit, int slots,
    const int* __restrict__ order) {
    constexpr int MT = 16 / NW, PH = PM / 2;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rows_sh[PM];
    __shared__ int xrows_sh[SHX ? PM : 1];              // SHX: the members' token rows
    const int cb = blockIdx.x;
    if ((int)blockIdx.y >= counts[0]) return;
    const int item = order ? order[blockIdx.y] : blockIdx.y;
    if (item < 0) return;                                // ORDER leaves the shared expert's items out
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;                                  // the shared expert
    if (threadIdx.x < PM) {
        const int p = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
        rows_sh[threadIdx.x] = p;
        if (SHX) xrows_sh[threadIdx.x] = p < 0 ? -1 : p / slots;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[MT][PM / 8][4];
    prompt_mainloop<2, PM, NW, SHX, B>(Xg, Xu, Tg, Tu, SHX ? xrows_sh : rows_sh, cnt, e, K, N >> 4, cb,
                                    reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    prompt_tile<2>(warp * MT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * PH < cnt; ++hf) {
    if (hf) __syncthreads();                             // the first half's rows are read
    prompt_spill<MT, PM>(Eg + (size_t)mat * PH * GU_ES, GU_ES, col0, acc, hf);
    __syncthreads();
    for (int i = warp; i < PH && hf * PH + i < cnt; i += NW) {
        const int p = rows_sh[hf * PH + i];
        float gv[4], uv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
            uv[j] = Eg[(size_t)(PH + i) * GU_ES + 4 * lane + j];
        }
        fwht128(gv, lane);
        fwht128(uv, lane);
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
            v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
        }
        fwht128(v, lane);
        const half2 o01 = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
        const half2 o23 = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
        uint2 packed;
        packed.x = *reinterpret_cast<const uint32_t*>(&o01);
        packed.y = *reinterpret_cast<const uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(xd + (size_t)p * N + n) = packed;
    }
    }  // halves
}

template <int PM, int NW, int B = 4>
__global__ void __launch_bounds__(NW * 32, 16 / NW) prompt_down_kernel(
    const half* __restrict__ Xd, const uint32_t* __restrict__ Td, const int* __restrict__ items,
    const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
    float* __restrict__ y, int K, int D, int E, const int* __restrict__ order) {
    constexpr int MT = 16 / NW, PH = PM / 2;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rows_sh[PM];
    const int cb = blockIdx.x;
    if ((int)blockIdx.y >= counts[0]) return;
    const int item = order ? order[blockIdx.y] : blockIdx.y;
    if (item < 0) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;
    if (threadIdx.x < PM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[MT][PM / 8][4];
    prompt_mainloop<1, PM, NW, false, B>(Xd, Xd, Td, Td, rows_sh, cnt, e, K, D >> 4, cb, reinterpret_cast<half*>(smem_raw),
                               acc);
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int hf = 0; hf < 2 && hf * PH < cnt; ++hf) {
    if (hf) __syncthreads();
    prompt_spill<MT, PM>(Ed, DN_ES, warp * MT * 16, acc, hf);
    __syncthreads();
    for (int i = warp; i < PH && hf * PH + i < cnt; i += NW) {
        const int p = rows_sh[hf * PH + i];
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
            const int n = cb * 256 + blk * 128 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j];
            fwht128(v, lane);
            const uint2 sv = *reinterpret_cast<const uint2*>(svh_d + (size_t)e * D + n);
            const half2 s01 = *reinterpret_cast<const half2*>(&sv.x), s23 = *reinterpret_cast<const half2*>(&sv.y);
            *reinterpret_cast<float4*>(y + (size_t)p * D + n) =
                make_float4(v[0] * HAD_SCALE * __low2float(s01), v[1] * HAD_SCALE * __high2float(s01),
                            v[2] * HAD_SCALE * __low2float(s23), v[3] * HAD_SCALE * __high2float(s23));
        }
    }
    }  // halves
}
