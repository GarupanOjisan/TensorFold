//! Mangled names of the GLM-5.3-Flash kernels in the glm_kda and glm_exl3 fatbins (cuobjdump -symbols, nvcc 13.0,
//! sm_121), and the host launch geometry the Python engine v0.6.6 uses for each. File:line references are into
//! ~/tf-native/py066/src/tensorfold/. Every launch is on the current stream; "smem" is the dynamic shared memory
//! argument; "static" is the kernel's own shared memory (cuobjdump -res-usage SHARED minus the 1 KiB the sm_121 ELF
//! reserves per CTA). Pointer arguments are device addresses (u64); `int` is i32, `size_t` u64, `float` f32;
//! bf16 is __nv_bfloat16, f16 is half. Sizes below are GLM-5.3-Flash at world 2: 64 KDA heads (32 a rank),
//! D = hidden 4096, I = moe_intermediate 2048 / world = 1024 a rank (glm5_next/cuda/forward.py:96), E = 288 routed
//! experts on every rank, slots = top_k + 1 = 9 (forward.py:95; the last slot is the shared expert, id E,
//! glm5_next/cuda/glue.py:385), all experts 4-bit mcg (config quantization_config bits 4, codebook mcg).
//!
//! KDA (glm5_next/cuda/kda.cu, built -O3 --fmad=false, kda.py:18-19)
//!
//! A null pointer stands for an undefined or empty tensor (ptr<T>, kda.cu:357-358). DK = DV = 128, TAPS = 4
//! (kda.cu:19). H = heads a rank = a_log.numel() = 32. GLM calls kda.chain from forward.py:273-282 and
//! kda.replay_layers from forward.py:545; kda.chain takes the wide path when rows >= WIDE_ROWS = 64 (kda.py:58, 79).
//!
//! chain_kernel (kda.py:85-87 -> kda.cpp:21-46 -> kda.cu:360-375, launch kda.cu:368):
//!   grid (H, 1, 1), block 1024, smem 0 (static 2568 B).
//!   args (kda.cu:48-56): int H, bf16* P, int p_stride, int b_off, bf16* A, int a_stride, bf16* G, int g_stride,
//!   bf16* cs, bf16* cw, float* state_in, float* a_log, float* dt_bias, bf16* norm_w, float eps, float lower,
//!   int rows, bf16* out, float* state_out, float* k_save, bf16* v_save, float* g_save, float* b_save.
//!   P = the KDA projection rows [rows, width] (forward.py:264), p_stride = P.stride(0) in elements, b_off = k.b_off;
//!   A = b.ka[:R], G = b.kg[:R] bf16 with their row strides; cs = conv state [3, 3*H*128] bf16 (forward.py:167);
//!   cw = conv weight [3*H*128, 4] bf16; state_in/state_out [H, 128, 128] f32 (st.rec[cur|1-cur, li]); a_log [H],
//!   dt_bias [H*128] f32; norm_w [128] bf16; eps = c.eps, lower = gate_lower_bound (-5.0, weights.py:94), both
//!   double in Python cast to float; out = scratch.out [rows, H*128] bf16; k_save/g_save [rows, H, 128] f32,
//!   v_save [rows, H, 128] bf16, b_save [rows, H] f32 (KDAScratch, kda.py:22-34). state_out may be null.
//!
//! chain_wide = three launches in order (kda.py:80-83 -> kda.cpp:48-79 -> kda.cu:397-419), same args as chain
//! plus q_tmp [rows, H, 128] f32 and y_tmp [rows, H, 128] bf16 (kda.py:62-70, shared by all layers):
//!   prep_kernel (launch kda.cu:406): grid (rows, H, 1), block 512, smem 0 (static 1024 B).
//!     args (kda.cu:221-226): int H, bf16* P, int p_stride, int b_off, bf16* A, int a_stride, bf16* cs, bf16* cw,
//!     float* a_log, float* dt_bias, float lower, float* q_out (= q_tmp), float* k_save, bf16* v_save,
//!     float* g_save, float* b_save.
//!   step_kernel<4, 16> (launch kda.cu:411-414): grid (H, DV / 4 / WARPS = 8, 1), block 128, smem 0
//!     (static 25664 B). args (kda.cu:272-276): int H, float* state_in, float* q_in (= q_tmp), float* k_save,
//!     bf16* v_save, float* g_save, float* b_save, int rows, bf16* y_out (= y_tmp), float* state_out (may be null).
//!   out_kernel (launch kda.cu:416-417): grid (rows, H, 1), block DV = 128, smem 0 (static 4 B).
//!     args (kda.cu:334-336): int H, bf16* y_in (= y_tmp), bf16* G, int g_stride, bf16* norm_w, float eps, bf16* out.
//!
//! replay_kernel (kda.py:91-92 -> kda.cpp:81-88 -> kda.cu:377-384, launch kda.cu:381): grid (H, 1, 1) with
//!   H = b_save.size(1), block 1024, smem 0 (static 512 B). args (kda.cu:153-156): int H, float* state_in,
//!   float* k_save, bf16* v_save, float* g_save, float* b_save, int rows, float* state_out.
//!
//! replay_layers_kernel (kda.py:50-55 -> kda.cpp:90-98 -> kda.cu:386-395, launch kda.cu:391):
//!   grid (layers * H, 1, 1), block 1024, smem 0 (static 512 B). Block b is layer b / H, head b % H.
//!   args (kda.cu:183-186): int H, float* state_in, size_t state_stride, float* k_save, bf16* v_save, float* g_save,
//!   float* b_save, size_t kv_stride, size_t b_stride, int rows, float* state_out.
//!   kda.py:54 passes state_stride = H*128*128, kv_stride = scratch.rows*H*128, b_stride = scratch.rows*H (elements),
//!   layers = scratch.layers (KDA layers), rows = the kept rows; state_in/out = st.rec[cur] / st.rec[1-cur]
//!   [layers, H, 128, 128] f32; k/v/g/b = the KDAScratchSet's [layers, rows, ...] buffers (kda.py:37-47).
//!   Note the Python host signature orders (.., layers, heads, rows, ..); the kernel takes only H and rows.
//!
//! EXL3 routed experts (cuda/exl3/experts.cu + experts_grouped.cuh, built -O3 -lineinfo, experts.py:28-30)
//!
//! GLM calls experts.routed(x, pick, wts=None, ex, s, out=None, R, limit=c.limit, act_mode=ACT_BF16 = 0)
//! (glm5_next/cuda/exl3_generic.py:9, from forward.py:402); with wts None the launches are, in order
//! (experts.py:184-197): group, rot_in, grouped (gate+up), gateup_epilogue, grouped (down), down_epilogue.
//! P = R * slots (experts.py:181). Scratch (experts.py:149-170): xg, xu [rows*slots, D] f16, xd [rows*slots, I] f16,
//! z f32 of max(2*SK_gu*I, SK_d*D) * rows*slots, y [rows*slots, D] f32; window(R): maxu = min(R*slots, E),
//! ids = ids[:maxu], members = members_buf[:maxu*R] viewed [maxu, R], so maxm = members.size(1) = R.
//! pick is int32 [rows, slots] contiguous; picks outside [0, E) (the shared slot, id E) are skipped by every kernel.
//!
//! group_kernel (experts.py:186 -> experts.cpp:58-69 -> experts.cu:290-310, launch experts.cu:306-308):
//!   grid (1, 1, 1), block GROUP_THREADS = 1024, smem R*slots*4 B (static 128 B). Host checks E <= 4096 and
//!   slots <= 32; when smem + 128 > 48 KiB it reads cudaFuncGetAttributes, fails if smem > sharedMemPerBlockOptin -
//!   sharedSizeBytes, and sets MaxDynamicSharedMemorySize to that limit if smem > maxDynamicSharedSizeBytes
//!   (experts.cu:292-305). args (experts.cu:19-21): int* pick, int* uids (= ids), int* ucount (= s.count, [1]),
//!   int* members, int R, int slots, int E, int maxm (= members.size(1) = R).
//!
//! rot_in_kernel<bf16> (experts.py:187 -> experts.cpp:71-83 -> experts.cu:312-330, launch experts.cu:322-324):
//!   grid (rows*slots, K/128, 2), block 32, smem 0. args (experts.cu:94-96): bf16* x, int x_stride, int* pick,
//!   f16* suh0, f16* suh1, f16* out0, f16* out1, int K, int slots, int E.
//!   x = b.normed[:R] bf16, x_stride = x.stride(0); suh0/suh1 = ex.suh_g/ex.suh_u [E, D]; out0/out1 = s.xg/s.xu;
//!   K = D = 4096 (host checks K % 128 == 0), rows = R.
//!
//! grouped_kernel<CB 1, NT 8, W 4, PF 1, LO 8, HI 8> (experts.py:189-190 and 193-194 -> experts.cpp:26-45 ->
//!   experts.cu:252-278 -> grouped_launch<1>, experts_grouped.cuh:324-342, launch :329):
//!   grid (maxu = uids.size(0), N / (16*NT) = N/128, mats * SK * MT) with MT = (maxm + 15) / 16, block W*32 = 128,
//!   smem 0 (static 32832 B: red[4][16][128] f32 and rows_sh[16]). blockIdx.z = (mat * SK + split) * MT + mtile.
//!   args (experts_grouped.cuh:190-195, in GroupedArgs order experts.cu:258-271): f16* X0, f16* X1, int64* TP0,
//!   int64* TP1, int* K2_0, int* K2_1, int* uids, int* ucount, int* members, float* Z, int K, int N, int P, int SK,
//!   int maxm, int slots. Z is [mats][SK][P][N] f32 (experts_grouped.cuh:282).
//!   Dispatch: cb = ex.cb = 1 (experts.cu:273-276); the setting nt 8, warps 4, pf 1 (experts_grouped.cuh:336) and
//!   lo = hi = 8 (TF_RANGES, :333) select this instance; other bit ranges need <.., 2, 10> or <.., 2, 16>, which this
//!   fatbin does not contain. Host check K % (16*SK*warps) == 0 and N % (16*nt) == 0 (experts.cu:257).
//!   gate+up: X0 = s.xg, X1 = s.xu, TP0/TP1 = ex.gate_ptr/ex.up_ptr, K2_0/K2_1 = ex.gate_k2/ex.up_k2, mats 2,
//!     K = D = 4096, N = I = 1024, SK = 4 (GLM_GATEUP, experts.py:19, 136-143) -> grid (maxu, 8, 8*MT).
//!   down: X0 = X1 = s.xd, TP0 = TP1 = ex.down_ptr, K2_0 = K2_1 = ex.down_k2, mats 1, K = I = 1024, N = D = 4096,
//!     SK = 1 (GLM_DOWN, experts.py:20) -> grid (maxu, 32, MT).
//!
//! gateup_epilogue_kernel (experts.py:191 -> experts.cpp:85-97 -> experts.cu:332-342, launch experts.cu:337):
//!   grid (rows*slots, N/128), block 32, smem 0. args (experts.cu:116-119): float* Z, int* pick, f16* svh_g,
//!   f16* svh_u, f16* suh_d, f16* xd, int P, int N, int SK, int E, float limit, int act_mode.
//!   svh_g/svh_u = ex.svh_g/ex.svh_u [E, I], suh_d = ex.suh_d [E, I], xd = s.xd; N = I = 1024 (host checks
//!   N % 128 == 0), SK = 4, limit = c.limit (swiglu_limit, 10.0, weights.py:103) as float, act_mode = 0 (GLM's
//!   bf16 roundings, not ACT_F32).
//!
//! down_epilogue_kernel (experts.py:196 -> experts.cpp:99-108 -> experts.cu:344-351, launch experts.cu:347):
//!   grid (rows*slots, D/128), block 32, smem 0. args (experts.cu:161-163): float* Z, int* pick, f16* svh_d,
//!   float* y, int P, int D, int SK, int E. svh_d = ex.svh_d [E, D], y = s.y; D = 4096, SK = 1.
//!
//! down_combine_kernel (experts.py:201 -> experts.cpp:118-130 -> experts.cu:361-371, launch experts.cu:366): only
//!   when wts is given, which GLM never does. grid (rows, D/128), block 32*slots (host checks slots <= 32), smem 0
//!   (static 16384 B). args (experts.cu:194-197): float* Z, int* pick, f16* svh_d, float* y, float* wts,
//!   float* out, int P, int D, int SK, int E, int slots.
//!
//! combine_kernel (experts.cpp:110-116 -> experts.cu:353-359, launch experts.cu:356): not called by routed().
//!   grid (rows, (D + 255)/256), block 256, smem 0. args (experts.cu:183-184): float* y, float* wts, float* out,
//!   int D, int slots.

// glm_kda.fatbin
/// kda.chain, rows < 64: one block a head runs every row (kda.cu:368).
pub const kda_chain = "_ZN10tf_glm_kda12chain_kernelEiPK13__nv_bfloat16iiS2_iS2_iS2_S2_PKfS4_S4_S2_ffiPS0_PfS6_S5_S6_S6_";
/// kda.chain, rows >= 64, launch 1 of 3: conv, decay, beta and the q/k norms of every row (kda.cu:406).
pub const kda_prep = "_ZN10tf_glm_kda11prep_kernelEiPK13__nv_bfloat16iiS2_iS2_S2_PKfS4_fPfS5_PS0_S5_S5_";
/// kda.chain, rows >= 64, launch 2 of 3: the delta rule row by row, step_kernel<4, 16> (kda.cu:412).
pub const kda_step = "_ZN10tf_glm_kda11step_kernelILi4ELi16EEEviPKfS2_S2_PK13__nv_bfloat16S2_S2_iPS3_Pf";
/// kda.chain, rows >= 64, launch 3 of 3: the gated RMSNorm of the read-out (kda.cu:416).
pub const kda_out = "_ZN10tf_glm_kda10out_kernelEiPK13__nv_bfloat16S2_iS2_fPS0_";
/// kda.replay: one layer's state after the kept rows (kda.cu:381); not called by GLM's forward.
pub const kda_replay = "_ZN10tf_glm_kda13replay_kernelEiPKfS1_PK13__nv_bfloat16S1_S1_iPf";
/// kda.replay_layers: every KDA layer's state after the kept rows, from commit (kda.cu:391, forward.py:545).
pub const kda_replay_layers = "_ZN10tf_glm_kda20replay_layers_kernelEiPKfmS1_PK13__nv_bfloat16S1_S1_mmiPf";

// glm_exl3.fatbin
/// ext.group: distinct experts and their members (experts.cu:306).
pub const exl3_group = "_ZN11tf_glm_exl312group_kernelEPKiPiS2_S2_iiii";
/// ext.rot_in on bf16 rows: x * suh, Hadamard, fp16 for gate and up (experts.cu:322).
pub const exl3_rot_in = "_ZN11tf_glm_exl313rot_in_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iii";
/// ext.grouped, gate+up (mats 2, SK 4) and down (mats 1, SK 1): mcg, 4-bit, NT 8, 4 warps, PF 1 (experts_grouped.cuh:329).
pub const exl3_grouped = "_ZN8tf_exl3x14grouped_kernelILi1ELi8ELi4ELi1ELi8ELi8EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiii";
/// ext.gateup_epilogue: splits summed, SwiGLU (act_mode 0 for GLM), rotated into xd (experts.cu:337).
pub const exl3_gateup_epilogue = "_ZN11tf_glm_exl322gateup_epilogue_kernelEPKfPKiPK6__halfS6_S6_PS4_iiiifi";
/// ext.down_epilogue: GLM's last expert launch, Y per slot in fp32 (experts.cu:347).
pub const exl3_down_epilogue = "_ZN11tf_glm_exl320down_epilogue_kernelEPKfPKiPK6__halfPfiiii";
/// ext.down_combine: down epilogue plus the weighted slot sum (experts.cu:366); emitted, not on GLM's path (wts is None).
pub const exl3_down_combine = "_ZN11tf_glm_exl319down_combine_kernelEPKfPKiPK6__halfPfS1_S7_iiiii";
/// ext.combine: the weighted slot sum alone (experts.cu:356); emitted, not called by routed().
pub const exl3_combine = "_ZN11tf_glm_exl314combine_kernelEPKfS1_Pfii";
