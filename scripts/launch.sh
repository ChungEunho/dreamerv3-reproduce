#!/usr/bin/env bash
# Launcher for DreamerV3 runs on the shared 4x RTX 5090 box.
#
# Safety rules (from the team):
#   * Only use a GPU that nobody else is using. The target GPU is checked with
#     nvidia-smi right before launch and the run is aborted if it is busy.
#   * Never preallocate the whole GPU memory (XLA_PYTHON_CLIENT_PREALLOCATE=false).
#
# Usage:
#   scripts/launch.sh <gpu_index|cpu> <logdir_name> <main.py args...>
# Example:
#   scripts/launch.sh 2 atari100k_pong_s0 --configs atari100k --task atari100k_pong --seed 0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/mnt/data8tb/eunho/miniforge3/envs/dreamerv3/bin/python"
GPU="$1"; shift
NAME="$1"; shift
LOGDIR="$ROOT/logdir/$NAME"
mkdir -p "$ROOT/logdir"

if [[ "$GPU" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  PLATFORM_ARGS="--jax.platform cpu"
else
  # Abort if the GPU has any compute process or more than 1 GB used.
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" | wc -l)
  if [[ "$PROCS" -gt 0 || "$USED" -gt 1024 ]]; then
    echo "GPU $GPU is in use (procs=$PROCS, mem=${USED}MiB). Refusing to launch." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$GPU"
  PLATFORM_ARGS="--jax.prealloc False"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

cd "$ROOT/dreamerv3"
echo "[launch] gpu=$GPU logdir=$LOGDIR args: $* $PLATFORM_ARGS"
exec "$PY" dreamerv3/main.py --logdir "$LOGDIR" "$@" $PLATFORM_ARGS
