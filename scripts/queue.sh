#!/usr/bin/env bash
# Job queue worker for the shared 4x RTX 5090 box. Run up to TWO workers
# (team rule: at most 2 GPUs); they coordinate through lock directories.
#
# Jobs come from scripts/queue.txt, one per line:  <logdir_name> <main.py args...>
# '#' lines are skipped; the file is re-read before every job.
#
# Per job:
#   1. Wait until one of this worker's preferred GPUs is free (scripts/gpu_free.sh:
#      no compute OR graphics process except Xorg/gnome-shell, <= 1 GB used)
#      and claim it (logdir/locks/gpu_<i>).
#   2. Claim the first pending job (logdir/locks/job_<name>). A job is pending
#      unless it has a DONE / GAVEUP marker or is claimed by a live process.
#      A run that already reached its step budget is just marked DONE.
#   3. Run it via launch.sh; release both locks afterwards.
# Locks hold the owner's PID; a lock whose PID is dead is stale and reclaimed.
# A GPU that dies with a CUDA / unknown error is blacklisted (logdir/bad_gpus)
# and the job resumes from its checkpoint on the next attempt (max 3).
#
# Usage: scripts/queue.sh <worker_name> <gpu_pref ...>
#   tmux new -d -s dreamer_wA 'scripts/queue.sh A 2 0 1 3'
#   tmux new -d -s dreamer_wB 'scripts/queue.sh B 0 1 3 2'
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/mnt/data8tb/eunho/miniforge3/envs/dreamerv3/bin/python"
QUEUE="$ROOT/scripts/queue.txt"
LOCKS="$ROOT/logdir/locks"
BAD="$ROOT/logdir/bad_gpus"
MAX_RETRIES=3
WORKER="$1"; shift
PREF=("$@"); [[ ${#PREF[@]} -eq 0 ]] && PREF=(2 0 1 3)
mkdir -p "$LOCKS"; touch "$BAD"

log() { echo "[queue $WORKER] $(date '+%F %T') $*"; }

# claim <lockdir> : atomically create a lock owned by this shell; reclaim stale ones.
claim() {
  local d="$1" pid
  if mkdir "$d" 2>/dev/null; then echo $$ > "$d/pid"; return 0; fi
  pid=$(cat "$d/pid" 2>/dev/null || true)
  if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$d/pid"; rmdir "$d" 2>/dev/null
    if mkdir "$d" 2>/dev/null; then echo $$ > "$d/pid"; return 0; fi
  fi
  return 1
}
release() { [[ -n "$1" ]] && { rm -f "$1/pid"; rmdir "$1" 2>/dev/null; }; return 0; }

# A GPU must stay free for 5 minutes straight (10 checks, 30 s apart), so a
# renderer that briefly restarts is not mistaken for a free GPU.
gpu_idle() {
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    "$ROOT/scripts/gpu_free.sh" "$1" || return 1
    [[ $i -lt 10 ]] && sleep 30
  done
  return 0
}

pick_gpu() {  # claims and prints an idle, non-blacklisted GPU
  local g
  for g in "${PREF[@]}"; do
    grep -qx "$g" "$BAD" && continue
    claim "$LOCKS/gpu_$g" || continue
    if gpu_idle "$g"; then echo "$g"; return 0; fi
    release "$LOCKS/gpu_$g"
  done
  return 1
}

pick_job() {  # claims and prints the first pending job line
  local line name
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    name="${line%% *}"
    [[ -f "$ROOT/logdir/$name/DONE" || -f "$ROOT/logdir/$name.GAVEUP" ]] && continue
    claim "$LOCKS/job_$name" || continue
    if "$PY" "$ROOT/scripts/is_complete.py" "$ROOT/logdir/$name"; then
      date > "$ROOT/logdir/$name/DONE"; log "$name already complete -> DONE"
      release "$LOCKS/job_$name"; continue
    fi
    echo "$line"; return 0
  done < "$QUEUE"
  return 1
}

has_pending() {
  local line name
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    name="${line%% *}"
    [[ -f "$ROOT/logdir/$name/DONE" || -f "$ROOT/logdir/$name.GAVEUP" ]] && continue
    return 0
  done < "$QUEUE"
  return 1
}

cuda_error() {
  grep -qiE "CUDA_ERROR|cuda error|unknown error|XlaRuntimeError|INTERNAL: |RESOURCE_EXHAUSTED|out of memory|cudnn|cublas|NCCL|device-side|illegal memory"
}

log "started, gpu preference: ${PREF[*]}"
while has_pending; do
  # 1. GPU first, so a waiting worker never sits on a job another worker could run.
  gpu=""
  until gpu=$(pick_gpu); do
    has_pending || break 2
    if [[ $(sort -u "$BAD" | grep -c .) -ge 4 ]]; then log "all GPUs blacklisted, abort"; exit 2; fi
    sleep 60
  done
  # 2. Job.
  if ! job=$(pick_job); then
    release "$LOCKS/gpu_$gpu"
    sleep 60   # remaining jobs are running in the other worker
    continue
  fi
  name="${job%% *}"; args="${job#* }"
  attempt=0; ok=0
  while true; do
    attempt=$((attempt+1))
    logf="$ROOT/logdir/$name.log"
    log "start $name attempt=$attempt gpu=$gpu"
    echo "=== worker $WORKER attempt $attempt gpu=$gpu $(date) ===" >> "$logf"
    # shellcheck disable=SC2086
    "$ROOT/scripts/launch.sh" "$gpu" "$name" $args >> "$logf" 2>&1
    code=$?
    log "end   $name exit=$code"
    if [[ $code -eq 0 ]]; then date > "$ROOT/logdir/$name/DONE"; ok=1; break; fi
    if tail -n 200 "$logf" | cuda_error; then
      log "GPU $gpu hit a CUDA/unknown error -> blacklisting it"
      echo "$gpu" >> "$BAD"
    fi
    release "$LOCKS/gpu_$gpu"
    [[ $attempt -ge $MAX_RETRIES ]] && break
    sleep 30
    gpu=""
    until gpu=$(pick_gpu); do sleep 60; done
  done
  [[ $ok -eq 0 ]] && echo "$(date) gave up after $attempt attempts" > "$ROOT/logdir/$name.GAVEUP"
  release "$LOCKS/gpu_$gpu"
  release "$LOCKS/job_$name"
done
log "no pending jobs, exiting"
