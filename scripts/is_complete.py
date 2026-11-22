"""Exit 0 if the run in <logdir> has reached its configured step budget.

The repo's logger writes `step` multiplied by the env action repeat (main.py),
while `run.steps` counts agent steps, so compare against steps * repeat.
Usage: python scripts/is_complete.py <logdir>
"""
import json
import pathlib
import sys

import yaml

d = pathlib.Path(sys.argv[1])
cfg, met = d / 'config.yaml', d / 'metrics.jsonl'
if not cfg.exists() or not met.exists():
  sys.exit(1)
config = yaml.safe_load(cfg.read_text())
suite = config['task'].split('_')[0]
repeat = config.get('env', {}).get(suite, {}).get('repeat', 1)
target = float(config['run']['steps']) * repeat
last = 0
for line in met.read_text().splitlines():
  try:
    last = max(last, json.loads(line).get('step', 0))
  except json.JSONDecodeError:
    pass
# Metrics are written every log_every seconds, so the last logged step can be a
# few thousand frames short of the budget even for a finished run. 99% of the
# budget is past the 400k-frame evaluation window (runs go to 440k), so this
# never cuts data that the analysis uses.
sys.exit(0 if last >= 0.99 * target else 1)
