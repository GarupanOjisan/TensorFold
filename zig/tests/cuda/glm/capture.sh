#!/bin/bash
# The Python v0.6.6 engine's capture on two Sparks: capture.sh RUN [capture_glm.py options] (README.md)
# Rank 0 (here) records; rank 1 (WORKER) mirrors and is stopped when rank 0 exits. Outputs: ~/tf-native/cap/RUN.
set -u -o pipefail
RUN=${1:?run id}; shift
WORKER=${WORKER:?rank 1 ssh target}
MASTER=${MASTER:?rank 0 address on the RoCE link}
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}
SNAP=${SNAP:?the EXL3 snapshot under ~/.cache/huggingface/hub}
for c in $(docker ps -q --filter name=tf-) ; do echo "PREFLIGHT: containers running on spark1"; docker ps; exit 3; done
[ -z "$(ssh "$WORKER" docker ps -q --filter name=tf-)" ] || { echo "PREFLIGHT: containers running on spark2"; exit 3; }
rsync -a --delete ${PY066:-$HOME/tf-native/py066}/ "$WORKER:tf-native/py066/" --exclude .git
rsync -a ~/tf-native/bin/ "$WORKER:tf-native/bin/"
rsync -a $(dirname "$0")/../../../../tools/zig/triton_aot_manifest.py "$WORKER:tf-native/py066/tools-zig/"
mkdir -p ${PY066:-$HOME/tf-native/py066}/tools-zig && cp $(dirname "$0")/../../../../tools/zig/triton_aot_manifest.py ${PY066:-$HOME/tf-native/py066}/tools-zig/
mkdir -p ~/tf-native/cap/$RUN/{triton,torch_ext,cuda_cache}
more=""; (( $# )) && more=$(printf '%q ' "$@")
ssh "$WORKER" mkdir -p tf-native/cap/$RUN/{triton,torch_ext,cuda_cache}
env_args="-e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:?} -e NCCL_IB_HCA=${NCCL_IB_HCA:?} -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} \
  -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e HOME=/tmp -e PYTHONPATH=/py/src -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 -e TORCH_CUDA_ARCH_LIST=12.1 -e TRITON_CACHE_DIR=/cap/triton -e TORCH_EXTENSIONS_DIR=/cap/torch_ext \
  -e CUDA_CACHE_PATH=/cap/cuda_cache -e TENSORFOLD_NO_UPDATE_CHECK=1 ${EXTRA_ENV:-}"
run_args="--rm --gpus all --network host --ipc host --ulimit memlock=-1 --device /dev/infiniband \
  -v \${PY066:-$HOME/tf-native/py066}:/py:ro -v \$HOME/tf-native/cap/$RUN:/cap -v \$HOME/.cache/huggingface/hub:/hub:ro \
  $env_args -w /py --entrypoint python $IMAGE -B /py/capture_glm.py --model /hub/$SNAP --out /cap --master $MASTER"
ssh "$WORKER" "docker run --name tf-cap1 $run_args --rank 1 $more" > ~/tf-native/cap/$RUN/rank1.log 2>&1 &
eval timeout 7200 docker run --name tf-cap0 $run_args --rank 0 --tools /py/tools-zig $more 2>&1 \
  | tee ~/tf-native/cap/$RUN/rank0.log | grep -E "capture|Error|error|Traceback"
r0=$?
ssh "$WORKER" "docker stop --time 5 tf-cap1" >/dev/null 2>&1
wait
echo "--- rank 1 tail"; tail -5 ~/tf-native/cap/$RUN/rank1.log
ls ~/tf-native/cap/$RUN
