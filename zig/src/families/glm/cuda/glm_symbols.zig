//! Mangled GLM-5.3-Flash kernel names in the glm_port_kda and glm_exl3 fatbins; notes/glm_symbols.md#launch-geometry

// glm_port_kda.fatbin
pub const kda_chain = "_ZN10tf_glm_kda12chain_kernelEiPK13__nv_bfloat16iiS2_iS2_iS2_S2_PKfS4_S4_S2_ffiPS0_PfS6_S5_S6_S6_"; // kda.chain, rows < 64: one block a head runs every row (kda.cu:368).
pub const kda_prep = "_ZN10tf_glm_kda11prep_kernelEiPK13__nv_bfloat16iiS2_iS2_S2_PKfS4_fPfS5_PS0_S5_S5_"; // kda.chain, rows >= 64, launch 1 of 3: conv, decay, beta and the q/k norms of every row (kda.cu:406).
pub const kda_step = "_ZN10tf_glm_kda11step_kernelILi4ELi16EEEviPKfS2_S2_PK13__nv_bfloat16S2_S2_iPS3_Pf"; // kda.chain, rows >= 64, launch 2 of 3: the delta rule row by row, step_kernel<4, 16> (kda.cu:412).
pub const kda_out = "_ZN10tf_glm_kda10out_kernelEiPK13__nv_bfloat16S2_iS2_fPS0_"; // kda.chain, rows >= 64, launch 3 of 3: the gated RMSNorm of the read-out (kda.cu:416).
pub const kda_replay = "_ZN10tf_glm_kda13replay_kernelEiPKfS1_PK13__nv_bfloat16S1_S1_iPf"; // kda.replay: one layer's state after the kept rows (kda.cu:381); not called by GLM's forward.
pub const kda_replay_layers = "_ZN10tf_glm_kda20replay_layers_kernelEiPKfmS1_PK13__nv_bfloat16S1_S1_mmiPf"; // kda.replay_layers: every KDA layer's state after the kept rows, from commit (kda.cu:391, forward.py:545).
pub const kda_prep_seg = "_ZN10tf_glm_kda15prep_seg_kernelEiPKiiPK13__nv_bfloat16iiS4_iS4_mS4_PKfS6_fPfS7_PS2_S7_S7_"; // kda.chain_segments (v1.8 kda.cu:613-627), launch 1 of 3: prep_kernel with each row's conv taps from its segment's slot.
pub const kda_step_seg = "_ZN10tf_glm_kda15step_seg_kernelILi4ELi16EEEviPKiPfmmPKfS5_PK13__nv_bfloat16S5_S5_PS6_"; // kda.chain_segments, launch 2 of 3: step_seg_kernel<4, 16>, block (segment x head, part).
pub const kda_replay_layers_seg = "_ZN10tf_glm_kda24replay_layers_seg_kernelEiiPKiPfmmmPKfPK13__nv_bfloat16S4_S4_mm"; // kda.replay_layers_segments (v1.8 kda.cu:629-634): block (segment, layer, head).

// glm_exl3.fatbin
pub const exl3_group = "_ZN11tf_glm_exl312group_kernelEPKiPiS2_S2_iiii"; // ext.group: distinct experts and their members (experts.cu:306).
pub const exl3_rot_in = "_ZN11tf_glm_exl313rot_in_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iii"; // ext.rot_in on bf16 rows: x * suh, Hadamard, fp16 for gate and up (experts.cu:322).
pub const exl3_grouped = "_ZN8tf_exl3x14grouped_kernelILi1ELi8ELi4ELi1ELi8ELi8EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiii"; // ext.grouped, gate+up (mats 2, SK 4) and down (mats 1, SK 1): mcg, 4-bit, NT 8, 4 warps, PF 1 (experts_grouped.cuh:329).
pub const exl3_gateup_epilogue = "_ZN11tf_glm_exl322gateup_epilogue_kernelEPKfPKiPK6__halfS6_S6_PS4_iiiifi"; // ext.gateup_epilogue: splits summed, SwiGLU (act_mode 0 for GLM), rotated into xd (experts.cu:337).
pub const exl3_down_epilogue = "_ZN11tf_glm_exl320down_epilogue_kernelEPKfPKiPK6__halfPfiiii"; // ext.down_epilogue: GLM's last expert launch, Y per slot in fp32 (experts.cu:347).
pub const exl3_down_combine = "_ZN11tf_glm_exl319down_combine_kernelEPKfPKiPK6__halfPfS1_S7_iiiii"; // ext.down_combine: down epilogue plus the weighted slot sum (experts.cu:366); emitted, not on GLM's path (wts is None).
pub const exl3_combine = "_ZN11tf_glm_exl314combine_kernelEPKfS1_Pfii"; // ext.combine: the weighted slot sum alone (experts.cu:356); emitted, not called by routed().
