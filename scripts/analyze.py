"""Compare our DreamerV3 runs against the official scores shipped in the repo.

Usage (from the project root, inside the `dreamerv3` conda env):
    python scripts/analyze.py --logdir logdir --out docs

Produces
    docs/assets/<suite>_<task>.png   learning curve, ours vs official (5 seeds)
    docs/assets/<suite>_overview.png all tasks in one figure
    docs/results.md                  markdown table + per-task verdict
    docs/results.json                machine readable summary

Conventions
  * Atari100k x-axis is environment frames (the repo logs `step` already
    multiplied by the action repeat of 4), matching the official score files
    (budget 400k frames; a run ends at 110k agent steps = 440k frames).
  * "Final score" = mean episode return over the last 10% of the frame budget
    (360k-400k frames), computed identically for our runs and official seeds.
  * Verdict "reproduced" if our final score is within the official
    [min, max] over seeds, widened by one official std.
"""
import argparse
import collections
import gzip
import json
import pathlib
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCORES = ROOT / 'dreamerv3' / 'scores'

SUITES = {
    # suite: official file, x budget, multiplier applied to the logged 'step'.
    # NOTE: the repo's logger already multiplies the step counter by the action
    # repeat (main.py: elements.Logger(step, outputs, multiplier)), so the
    # 'step' in metrics.jsonl / scores.jsonl is in env FRAMES for Atari.
    'atari100k': dict(file='atari100k-dreamerv3.json.gz', budget=400_000, repeat=1),
    'dmc_vision': dict(file='dmc_vision-dreamerv3.json.gz', budget=1_000_000, repeat=1),
    'dmc_proprio': dict(file='dmc_proprio-dreamerv3.json.gz', budget=1_000_000, repeat=1),
}


def load_official(suite):
  runs = json.load(gzip.open(SCORES / SUITES[suite]['file']))
  by = collections.defaultdict(list)
  for r in runs:
    by[r['task']].append((np.asarray(r['xs'], float), np.asarray(r['ys'], float)))
  return by


def load_ours(logdir):
  """Yield (name, suite, task, seed, xs_frames, ys) for every run with scores."""
  for d in sorted(pathlib.Path(logdir).glob('*/')):
    f = d / 'scores.jsonl'
    cfg = d / 'config.yaml'
    if not f.exists() or not cfg.exists():
      continue
    config = yaml.safe_load(cfg.read_text())
    task = config['task']                  # e.g. atari100k_pong
    suite = task.split('_')[0]
    if suite not in SUITES:
      continue
    official_task = task.replace('atari100k_', 'atari_', 1)
    df = pd.read_json(f, lines=True)
    if not len(df):
      continue
    xs = df['step'].to_numpy(float) * SUITES[suite]['repeat']
    ys = df['episode/score'].to_numpy(float)
    yield dict(name=d.name, suite=suite, task=official_task, seed=config['seed'],
               xs=xs, ys=ys, done=(d / 'DONE').exists(),
               steps=float(config['run']['steps']))


def final_score(xs, ys, budget, frac=0.1):
  lo = budget * (1 - frac)
  m = (xs >= lo) & (xs <= budget * 1.02)
  if m.sum() == 0:
    return np.nan
  return float(ys[m].mean())


def binned(xs, ys, edges):
  idx = np.digitize(xs, edges) - 1
  out = np.full(len(edges) - 1, np.nan)
  for i in range(len(edges) - 1):
    sel = ys[idx == i]
    if len(sel):
      out[i] = sel.mean()
  return out


def smooth(y, k=5):
  y = pd.Series(y)
  return y.rolling(k, min_periods=1, center=True).mean().to_numpy()


def plot_task(ax, suite, task, ours, official, budget):
  edges = np.linspace(0, budget, 41)
  mid = (edges[:-1] + edges[1:]) / 2
  if official:
    curves = np.stack([binned(xs, ys, edges) for xs, ys in official])
    mean = np.nanmean(curves, 0)
    std = np.nanstd(curves, 0)
    ax.plot(mid, mean, color='#888', lw=2, label=f'official ({len(official)} seeds)')
    ax.fill_between(mid, mean - std, mean + std, color='#888', alpha=0.25, lw=0)
  colors = ['#0022ff', '#ff0011', '#33aa00', '#ddaa00']
  for i, r in enumerate(ours):
    c = binned(r['xs'], r['ys'], edges)
    ok = ~np.isnan(c)
    ax.plot(mid[ok], smooth(c[ok], 3), color=colors[i % len(colors)], lw=2,
            label=f"ours seed {r['seed']}" + ('' if r['done'] else ' (running)'))
  ax.set_title(task.replace('atari_', '').replace('dmc_', ''))
  ax.set_xlabel('env frames (k)' if suite == 'atari100k' else 'env steps (k)')
  ax.set_ylabel('episode return')
  ax.set_xlim(0, budget)
  ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}'))
  ax.grid(alpha=0.3)
  ax.legend(fontsize=7, loc='best')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--logdir', default=str(ROOT / 'logdir'))
  p.add_argument('--out', default=str(ROOT / 'docs'))
  args = p.parse_args()
  out = pathlib.Path(args.out)
  (out / 'assets').mkdir(parents=True, exist_ok=True)

  ours = list(load_ours(args.logdir))
  if not ours:
    print('no runs found'); return
  summary = []
  by_suite = collections.defaultdict(lambda: collections.defaultdict(list))
  for r in ours:
    by_suite[r['suite']][r['task']].append(r)

  for suite, tasks in by_suite.items():
    official_all = load_official(suite)
    budget = SUITES[suite]['budget']
    n = len(tasks)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax, (task, runs) in zip(axes.flat, sorted(tasks.items())):
      official = official_all.get(task, [])
      plot_task(ax, suite, task, runs, official, budget)
      # single task figure too
      f1, a1 = plt.subplots(figsize=(4.8, 3.6))
      plot_task(a1, suite, task, runs, official, budget)
      f1.tight_layout(); f1.savefig(out / 'assets' / f'{suite}_{task}.png', dpi=130); plt.close(f1)

      off_final = [final_score(xs, ys, budget) for xs, ys in official]
      for r in runs:
        ours_final = final_score(r['xs'], r['ys'], budget)
        progress = min(1.0, r['xs'].max() / budget) if len(r['xs']) else 0.0
        lo, hi = (min(off_final), max(off_final)) if off_final else (np.nan, np.nan)
        std = float(np.std(off_final)) if off_final else np.nan
        if np.isnan(ours_final):
          verdict = 'running' if not r['done'] else 'no data'
        elif lo - std <= ours_final <= hi + std:
          verdict = 'reproduced'
        else:
          verdict = 'mismatch'
        summary.append(dict(
            suite=suite, task=task, seed=r['seed'], run=r['name'], done=r['done'],
            progress=round(progress, 3), ours_final=None if np.isnan(ours_final) else round(ours_final, 1),
            ours_last_episode=float(r['ys'][-1]), ours_episodes=int(len(r['ys'])),
            official_mean=round(float(np.mean(off_final)), 1) if off_final else None,
            official_std=round(std, 1) if off_final else None,
            official_min=round(float(lo), 1) if off_final else None,
            official_max=round(float(hi), 1) if off_final else None,
            official_seeds=len(off_final), verdict=verdict))
    for ax in list(axes.flat)[n:]:
      ax.axis('off')
    fig.tight_layout(); fig.savefig(out / 'assets' / f'{suite}_overview.png', dpi=130); plt.close(fig)

  (out / 'results.json').write_text(json.dumps(summary, indent=1))
  lines = ['| suite | task | seed | progress | ours (last 10%) | official mean ± std (min–max, n seeds) | verdict |',
           '|---|---|---|---|---|---|---|']
  for s in sorted(summary, key=lambda s: (s['suite'], s['task'], s['seed'])):
    off = (f"{s['official_mean']} ± {s['official_std']} ({s['official_min']}–{s['official_max']}, n={s['official_seeds']})"
           if s['official_mean'] is not None else 'n/a')
    lines.append(f"| {s['suite']} | {s['task']} | {s['seed']} | {int(s['progress']*100)}% | "
                 f"{s['ours_final'] if s['ours_final'] is not None else '–'} | {off} | **{s['verdict']}** |")
  table = '\n'.join(lines) + '\n'
  (out / 'results.md').write_text(table)
  print(table)

  # Render the GitHub Pages index from the template.
  import datetime
  tpl = out / '_template.md'
  if tpl.exists():
    n_done = sum(s['done'] for s in summary)
    n_rep = sum(s['verdict'] == 'reproduced' for s in summary)
    figs = '\n'.join(
        f"![{s}]({f'assets/{s}_overview.png'})" for s in by_suite)
    repo_file = out / '.repo'   # "owner/name" of the GitHub repo, used for links
    repo = repo_file.read_text().strip() if repo_file.exists() else 'OWNER/REPO'
    page = (tpl.read_text()
            .replace('{{GITHUB_REPO}}', repo)
            .replace('{{RESULTS_TABLE}}', table)
            .replace('{{OVERVIEW_FIGURES}}', figs)
            .replace('{{N_RUNS}}', str(len(summary)))
            .replace('{{N_DONE}}', str(n_done))
            .replace('{{N_REPRODUCED}}', str(n_rep))
            .replace('{{UPDATED}}', datetime.datetime.now().strftime('%Y-%m-%d %H:%M KST')))
    (out / 'index.md').write_text(page)


if __name__ == '__main__':
  main()
