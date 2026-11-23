---
layout: default
title: DreamerV3 Reproduction (Week 1)
---

# DreamerV3 Reproduction — Week 1 (Reproduce)

_Last updated: 2026-11-23 12:38 KST · runs: 6 (finished 5, reproduced 0/6 games)_

**Paper:** Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3), [arXiv 2301.04104](https://arxiv.org/abs/2301.04104), Nature 2025.
**Code:** [danijar/dreamerv3](https://github.com/danijar/dreamerv3) at commit `e01491f` (Oct 2026), unmodified.

## 1. Goal

Confirm that the public DreamerV3 implementation, run with its **default (paper) hyperparameters**
on our own hardware, reaches the same scores as the official result files that ship with the repo
(`dreamerv3/scores/*.json.gz`, 5 seeds per task). We use the **Atari100k** benchmark because one
run finishes in ~2 hours on a single GPU, which makes a multi-game, multi-seed check possible within
a week.

## 2. What DreamerV3 does (short version)

DreamerV3 is a model-based RL agent with a fixed set of hyperparameters across domains:

1. **World model (RSSM).** An encoder maps pixels (64×64) to discrete latent codes (32 categoricals × 64 classes);
   a recurrent model predicts the next latent, the reward and the episode-continuation flag.
   Loss = reconstruction + reward + continue + dynamics/representation KL with free bits (1 nat) and KL balancing.
2. **Actor–critic trained in imagination.** From replayed states the world model dreams 15-step trajectories;
   the critic regresses λ-returns and the actor maximises them with an entropy bonus (3e-4).
3. **Robustness tricks that remove per-domain tuning:** `symlog` transform of inputs/targets, two-hot
   discrete regression for reward and value heads, percentile-based return normalisation,
   unimix (1%) on categoricals, and adaptive gradient clipping.

The repo defaults (`dreamerv3/configs.yaml`) correspond to the paper's **200M-parameter** model;
for Atari100k it uses `train_ratio=256`, one env, 110k agent steps (= 440k frames incl. the 400k budget).

## 3. Setup used here

| item | value |
|---|---|
| GPU | 1× NVIDIA RTX 5090 (32 GB, Blackwell), ~17 GB used per run |
| OS / driver | Ubuntu 20.04, driver 570.181 (CUDA 12.8) |
| Python | 3.11 (dedicated conda env `dreamerv3`) |
| JAX | `jax[cuda12]==0.6.2` (see note below) |
| Env packages | `ale_py==0.9.0`, `autorom`, `crafter`, `dm_control` |
| Throughput | ≈3.1 gradient updates/s (batch 16×64), ≈13 env steps/s at steady state |
| Wall-clock | ≈2.2 h per Atari100k run |

Exact package versions: [`env/requirements-lock.txt`](https://github.com/ChungEunho/dreamerv3-reproduce/blob/main/env/requirements-lock.txt).

### Install notes / pitfalls we hit

- The pinned `jax[cuda12]==0.4.33` in `requirements.txt` does not support Blackwell GPUs (sm_120).
  The latest JAX (0.10) breaks the repo (`jax.jit` positional arguments were removed:
  `TypeError: jit() takes from 0 to 1 positional arguments but 5 were given`).
  **`jax[cuda12]==0.6.2` + `chex<0.1.90`** works without touching the code.
- `nvidia-cuda-nvcc-cu12<=12.2` from `requirements.txt` must be dropped (Blackwell needs ≥12.8).
- Python output is block-buffered when piped to a log; set `PYTHONUNBUFFERED=1`.
- JAX preallocates ~75% of GPU memory by default. We run with `--jax.prealloc False`
  (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) so the shared GPU is not held hostage.

## 4. Results

**Metric.** Episode return of the training episodes, as logged by the repo (`scores.jsonl`).
"Final score" = mean return over the **last 10 % of the 400k-frame budget**, computed the same way for our
runs and for each official seed. A run counts as *reproduced* if its final score lies inside the official
seed range `[min, max]` widened by one std.

#### Per game (verdict over our finished seeds)

| task | seeds done | ours mean ± std | paper Table 9 | official file mean (min–max, 5 seeds) | gamer-norm ours / paper | curve in band | verdict |
|---|---|---|---|---|---|---|---|
| boxing | 1/3 | 78.8 ± 0.0 | 82 | 79.4 (70.2–85.0) | 656.9% / 683.3% | 88% | **reproduced (partial 1/3)** |
| hero | 1/3 | 13372.8 ± 0.0 | 13354 | 13344.3 (13190.7–13465.0) | 41.4% / 41.4% | 90% | **reproduced (partial 1/3)** |
| krull | 1/3 | 6143.3 ± 0.0 | 8171 | 7683.5 (7365.0–8154.3) | 425.6% / 615.4% | 70% | **below official range (partial 1/3)** |
| kung_fu_master | 1/3 | 23492.0 ± 0.0 | 25900 | 23301.7 (18766.7–29533.3) | 103.4% / 114.1% | 88% | **reproduced (partial 1/3)** |
| ms_pacman | 0/3 | – | 1521 | 1525.8 (1206.7–1807.5) | – / 18.3% | – | **running** |
| pong | 1/3 | -12.0 ± 0.0 | -4 | -4.2 (-9.2–-1.3) | 25.0% / 47.2% | 74% | **below official range (partial 1/3)** |

#### Per seed

| task | seed | progress | ours (last 10%) | official mean ± std (min–max) | curve in band | verdict |
|---|---|---|---|---|---|---|
| boxing | 0 | 100% | 78.8 | 79.4 ± 5.7 (70.2–85.0) | 88% | **reproduced** |
| hero | 0 | 100% | 13372.8 | 13344.3 ± 96.5 (13190.7–13465.0) | 90% | **reproduced** |
| krull | 0 | 100% | 6143.3 | 7683.5 ± 356.4 (7365.0–8154.3) | 70% | **below official range** |
| kung_fu_master | 0 | 100% | 23492.0 | 23301.7 ± 4398.5 (18766.7–29533.3) | 88% | **reproduced** |
| ms_pacman | 0 | 87% | – | 1525.8 ± 240.7 (1206.7–1807.5) | 40% | **running** |
| pong | 0 | 100% | -12.0 | -4.2 ± 2.7 (-9.2–-1.3) | 74% | **below official range** |


![atari100k](assets/atari100k_overview.png)

Per-task figures are in [`docs/assets/`](https://github.com/ChungEunho/dreamerv3-reproduce/tree/main/docs/assets)
(grey = official mean ± std over 5 seeds, colored = our seeds).

## 5. How to reproduce

```bash
# 1. environment
conda create -n dreamerv3 python=3.11 -y
conda activate dreamerv3
pip install "jax[cuda12]==0.6.2" "chex<0.1.90"
pip install -r dreamerv3/requirements.txt --no-deps   # or env/requirements-lock.txt for exact versions
pip install crafter dm_control matplotlib pandas pyyaml

# 2. one run (paper defaults), on GPU 2, logs to logdir/atari100k_pong_s0
scripts/launch.sh 2 atari100k_pong_s0 --configs atari100k --task atari100k_pong --seed 0

# 3. (optional) run the whole job list in scripts/queue.txt sequentially
tmux new -d -s wA "scripts/queue.sh A 2 0 1 3"   # worker A, GPU preference order
tmux new -d -s wB "scripts/queue.sh B 0 1 3 2"   # worker B (max 2 workers = max 2 GPUs)

# 4. compare with the official scores and rebuild this page
python scripts/analyze.py --logdir logdir --out docs
```

`scripts/launch.sh` refuses to start if the chosen GPU is busy. `scripts/queue.sh` workers only take a GPU that is completely idle, coordinate through lock files, blacklist a GPU that
dies with a CUDA error and resumes the run from its checkpoint on another idle GPU.

## 6. Notes & limitations

- Only Atari100k was run end-to-end: Crafter (1M steps, train_ratio 512) and DMC-vision (1M steps) need
  roughly 50 h and 25 h per run on one GPU, which did not fit the one-week budget.
- Scores are from training episodes (sticky actions off, 1 env), matching how the official files were produced.
- Pong on Atari100k is a weak, high-variance task for DreamerV3 (official seeds range −9 … −1); the more
  deterministic games (Boxing, Hero, Krull, Kung-Fu Master) are the informative ones.
