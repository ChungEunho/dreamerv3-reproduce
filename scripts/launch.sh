#!/usr/bin/env bash
# Launcher for DreamerV3 runs on the shared 4x RTX 5090 box.
#
# Safety rules (from the team):
#   * Only use a GPU that nobody else is using (compute OR graphics processes).
#     The target GPU is checked right before launch and the run is aborted if busy.
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
  # Abort unless the GPU is completely free (no compute or graphics process of
  # anyone else, see scripts/gpu_free.sh).
  if ! "$ROOT/scripts/gpu_free.sh" "$GPU"; then
    echo "GPU $GPU is in use:" >&2; "$ROOT/scripts/gpu_free.sh" "$GPU" -v >&2
    echo "Refusing to launch." >&2
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
