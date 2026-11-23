# DreamerV3 reproduction (team project, week 1)

Reproduction of **DreamerV3** ([arXiv 2301.04104](https://arxiv.org/abs/2301.04104)) using the
official implementation [danijar/dreamerv3](https://github.com/danijar/dreamerv3) on Atari100k.

**Results page:** see `docs/` (published with GitHub Pages) — `docs/index.md`.

```
.
├── dreamerv3/        upstream code, commit e01491f, unmodified
├── env/              requirements-lock.txt (exact package versions that worked on RTX 5090)
├── scripts/
│   ├── launch.sh     run one job on an idle GPU (refuses busy GPUs, no memory preallocation)
│   ├── queue.sh      queue worker (run up to 2): idle-GPU only, lock files, CUDA-crash fallback
│   ├── is_complete.py  has a run reached its step budget?
│   ├── queue.txt     job list
│   └── analyze.py    compare logdir/* with official scores -> docs/results.md, figures, index.md
├── logdir/           metrics.jsonl / scores.jsonl per run (checkpoints and replay are git-ignored)
└── docs/             GitHub Pages site
```

Quick start:

```bash
conda create -n dreamerv3 python=3.11 -y && conda activate dreamerv3
pip install "jax[cuda12]==0.6.2" "chex<0.1.90"
pip install -r env/requirements-lock.txt
scripts/launch.sh 2 atari100k_pong_s0 --configs atari100k --task atari100k_pong --seed 0 \
    --env.atari100k.actions all --env.atari100k.noops 0   # env protocol of the official scores
python scripts/analyze.py
```
