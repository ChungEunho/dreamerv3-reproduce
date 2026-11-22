#!/usr/bin/env bash
# Rebuild the results page from logdir/ and commit it locally. Pushes to GitHub
# only with --push (results are published once the reproduction is verified).
# Usage: scripts/publish.sh [--push] ["commit message"]
set -euo pipefail
PUSH=0; if [[ "${1:-}" == "--push" ]]; then PUSH=1; shift; fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/mnt/data8tb/eunho/miniforge3/envs/dreamerv3/bin/python"
cd "$ROOT"
"$PY" scripts/analyze.py --logdir logdir --out docs > /dev/null
# Keep only the small per-run artifacts (metrics, scores, config, DONE) in git.
git add docs scripts README.md env
for d in logdir/*/; do
  for f in scores.jsonl metrics.jsonl config.yaml DONE; do
    [[ -f "$d$f" ]] && git add -f "$d$f"
  done
done
if git diff --cached --quiet; then echo "[publish] nothing to commit"; exit 0; fi
git commit -q -m "${1:-results: update $(date '+%Y-%m-%d %H:%M')}

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
echo "[publish] committed $(git rev-parse --short HEAD)"
if [[ $PUSH -eq 1 ]]; then git push -q origin main; echo "[publish] pushed"; fi
