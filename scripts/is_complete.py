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
# The driver advances in chunks of 10 agent steps; allow one chunk of slack.
sys.exit(0 if last >= target - 10 * repeat else 1)
