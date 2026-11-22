#!/usr/bin/env bash
# Sequential job queue for the shared 4x RTX 5090 box.
#
# Reads jobs from scripts/queue.txt, one per line:
#   <logdir_name> <main.py args...>
# Lines starting with '#' are skipped. The file is re-read after every job so
# new jobs can be appended while the queue is running. A job whose logdir
# already contains a 'DONE' marker is skipped, so the queue can be restarted.
#
# GPU policy (team rules):
#   * GPUs are tried in the order given on the command line (default "2 0 1 3").
#   * A GPU is only used when it is idle: no compute processes and <= 1 GB used.
#     Other people's jobs are never touched; the queue waits instead.
#   * If a job dies with a CUDA / unknown error, that GPU is put on a blacklist
#     (logdir/bad_gpus) and the job is resumed from its checkpoint on another
#     idle GPU. Up to MAX_RETRIES attempts per job.
#
# Usage: scripts/queue.sh [gpu_pref ...]      e.g. scripts/queue.sh 2 0 1 3
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE="$ROOT/scripts/queue.txt"
BAD="$ROOT/logdir/bad_gpus"
MAX_RETRIES=3
PREF=("$@"); [[ ${#PREF[@]} -eq 0 ]] && PREF=(2 0 1 3)
mkdir -p "$ROOT/logdir"; touch "$BAD"

gpu_idle() {
  local procs used
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | wc -l)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1")
  [[ "$procs" -eq 0 && "$used" -le 1024 ]]
}

pick_gpu() {  # prints the first idle, non-blacklisted GPU, or nothing
  local g
  for g in "${PREF[@]}"; do
    grep -qx "$g" "$BAD" && continue
    gpu_idle "$g" && { echo "$g"; return 0; }
  done
  return 1
}

cuda_error() {  # did this log die from a GPU-side error?
  grep -qiE "CUDA_ERROR|cuda error|unknown error|XlaRuntimeError|INTERNAL: |RESOURCE_EXHAUSTED|out of memory|cudnn|cublas|NCCL|device-side|illegal memory" "$1"
}

while true; do
  job=""
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    name="${line%% *}"
    [[ -f "$ROOT/logdir/$name/DONE" ]] && continue
    [[ -f "$ROOT/logdir/$name.GAVEUP" ]] && continue
    job="$line"; break
  done < "$QUEUE"
  [[ -z "$job" ]] && { echo "[queue] $(date '+%F %T') no pending jobs, exiting"; break; }

  name="${job%% *}"; args="${job#* }"
  attempt=0; done_ok=0
  while [[ $attempt -lt $MAX_RETRIES ]]; do
    attempt=$((attempt+1))
    gpu=""
    until gpu=$(pick_gpu); do
      if [[ $(grep -c . "$BAD") -ge 4 ]]; then echo "[queue] all GPUs blacklisted, abort"; exit 2; fi
      sleep 60
    done
    log="$ROOT/logdir/$name.log"
    echo "[queue] $(date '+%F %T') start $name attempt=$attempt gpu=$gpu :: $args"
    echo "=== attempt $attempt gpu=$gpu $(date) ===" >> "$log"
    # shellcheck disable=SC2086
    "$ROOT/scripts/launch.sh" "$gpu" "$name" $args >> "$log" 2>&1
    code=$?
    echo "[queue] $(date '+%F %T') end   $name exit=$code"
    if [[ $code -eq 0 ]]; then date > "$ROOT/logdir/$name/DONE"; done_ok=1; break; fi
    if tail -n 200 "$log" | cuda_error /dev/stdin; then
      echo "[queue] GPU $gpu hit a CUDA/unknown error -> blacklisting it"
      echo "$gpu" >> "$BAD"
    fi
    sleep 30   # resume from checkpoint on the next attempt
  done
  [[ $done_ok -eq 0 ]] && { echo "$(date) gave up after $attempt attempts" > "$ROOT/logdir/$name.GAVEUP"; }
done
