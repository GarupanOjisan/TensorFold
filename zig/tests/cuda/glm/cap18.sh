#!/bin/bash
# A v1.8 Python capture with the not-yet-ported features off: cap18.sh RUN DENSE [capture_glm.py options]
# (DENSE bf16 or q4). The toggles that pick v0.6.6's kernels over v1.8's new ones are the OFF list below.
RUN=$1 DENSE=$2; shift 2
OFF="-e TF_GLM_DENSE=$DENSE -e TF_GLM_KV=bf16 -e TF_GLM_KDA_CHUNKED=0 -e TF_GLM_EXL3_LOADS=0 -e TF_GLM_L2PF=0 \
 -e TF_GLM_EXL3_PROMPT=0 -e TF_GLM_EXL3_DEC=0 -e TF_GLM_EXL3_FUSE=0 -e TF_GLM_EXL3_XROW=0 -e TF_GLM_EXL3_SEQ=0 \
 -e TF_GLM_EXL3_PF=0 -e TF_GLM_EXL3_NFIRST=0 -e TF_GLM_EXL3_ORDER=0 -e TF_GLM_EXL3_ROWROT=0 \
 -e TF_GLM_HC_DEC=0 -e TF_GLM_HC_MMA=0 -e TF_GLM_LATENT_MMA=0 -e TF_GLM_SPARSE_ONEPASS=0 -e TF_GLM_KDA_WIDE_ROWS=64 \
 -e TF_GLM_SEQ_ROWS=1000000 -e TF_GLM_COMBINE_SY=0 -e TF_GLM_SCORE_RB=1 -e TF_GLM_WIDE_GRAPHS=0 ${OFF_EXTRA:-}"
export EXTRA_ENV="$OFF"
exec ~/tf-native/bin/pycap18.sh $RUN "$@"
