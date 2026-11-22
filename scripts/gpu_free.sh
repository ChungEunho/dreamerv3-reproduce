#!/usr/bin/env bash
# Exit 0 only if GPU <index> is completely free for us to use.
#
# "Free" means: no process at all on the GPU except the display server
# (Xorg, gnome-shell), counting BOTH compute (C) and graphics (G) processes,
# and at most 1 GB memory in use. Graphics processes matter: other people's
# simulators render on a GPU without showing up as compute apps.
# Usage: scripts/gpu_free.sh <gpu_index> [-v]
set -uo pipefail
GPU="$1"
foreign=$(nvidia-smi -i "$GPU" | sed -n '/Processes:/,$p' \
  | grep -E '^\|\s+[0-9]+\s+\S+\s+\S+\s+[0-9]+\s+(C|G|C\+G)\s' \
  | grep -vE '/usr/lib/xorg/Xorg|/usr/bin/gnome-shell')
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [[ "${2:-}" == "-v" ]]; then
  echo "gpu=$GPU used=${used}MiB foreign_processes=$(grep -c . <<< "$foreign")"
  [[ -n "$foreign" ]] && echo "$foreign"
fi
[[ -z "$foreign" && "$used" -le 1024 ]]
