# DreamerV3 reproduction (team project, week 1)

Reproduction of **DreamerV3** ([arXiv 2301.04104](https://arxiv.org/abs/2301.04104)) using the
official implementation [danijar/dreamerv3](https://github.com/danijar/dreamerv3) on Atari100k.

**Results page:** https://chungeunho.github.io/dreamerv3-reproduce/ (built from `docs/` with GitHub Pages).

The official code is included as a plain folder (not a submodule), so a normal `git clone` gets everything,
and changes to `dreamerv3/` are tracked in this repository.

```
.
├── dreamerv3/        official code copied from danijar/dreamerv3 @ e01491f (MIT, see dreamerv3/LICENSE); unmodified so far
├── env/              requirements-lock.txt (exact package versions that worked on RTX 5090)
├── scripts/
│   ├── is_complete.py  has a run reached its step budget?
│   └── analyze.py    compare logdir/* with official scores -> docs/results.md, figures, index.md
├── logdir/           metrics.jsonl / scores.jsonl per run (checkpoints and replay are git-ignored)
└── docs/             GitHub Pages site
```

## Setup

```bash
git clone https://github.com/ChungEunho/dreamerv3-reproduce.git && cd dreamerv3-reproduce
conda create -n dreamerv3 python=3.11 -y && conda activate dreamerv3
pip install "jax[cuda12]==0.6.2" "chex<0.1.90"
pip install -r env/requirements-lock.txt
```

## Run one game

The official scores were produced with all 18 Atari actions and no start no-ops, so pass both flags:

```bash
cd dreamerv3
python dreamerv3/main.py --logdir ../logdir/atari100k_pong_s0 \
    --configs atari100k --task atari100k_pong --seed 0 \
    --env.atari100k.actions all --env.atari100k.noops 0
cd ..
python scripts/is_complete.py logdir/atari100k_pong_s0 && date > logdir/atari100k_pong_s0/DONE
python scripts/analyze.py    # compare logdir/* with the official scores -> docs/
```

`analyze.py` reads the game and seed from each run's `config.yaml`, and only counts a run as finished
once it has a `DONE` file. One run takes about 2h20m and
~17 GB of GPU memory on an RTX 5090.
