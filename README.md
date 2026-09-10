# world-model-audits

A test suite for probing what a learned world model has actually represented — whether
its latent state carries physical quantities, whether its predictor does anything beyond
passing its input through, and whether its response to actions looks like the dynamics of
the environment it was trained on.

Two published world models are wired up here, both on Lunar Lander:

- **[LeWorldModel (LeWM)](https://le-wm.github.io/)** — a JEPA trained end-to-end from pixels.
  Most of the work in this repo targets this model.
- **[R2-Dreamer](https://github.com/NM512/r2dreamer)** — an RSSM world model, trained here
  from state vectors. A smaller, earlier pass; see [Status](#status).

Neither model is mine. Both are vendored as forks — see [Provenance](#provenance).

## Status

This is a working repo, not a package. It contains no results, reports, plots or numbers —
those live outside the repo, in run directories. What is here is the machinery that produced
them and the tests that cover it.

Paths and configuration come from `env.sh` (see `env.sh.example`), which is machine-local.
Running any of it needs the models, the datasets and a GPU. If something here is useful to
you, take it and adapt it; it is MIT licensed.

`dreamerv3/` is a thin slice — an env wrapper, a training launcher and an eval script. The
audit suite proper is under `lewm/`.

## What the suite checks

`lewm/scripts/eval_suite.py` runs the tests, caches each one's JSON output, and renders a
single report. What it measures:

- linear probes from encoder-z and from predicted-z to the six kinematic dims (x, y, vx, vy,
  angle, angular velocity), as per-dim R²
- how much the predictor edits z, against frame-to-frame encoder drift as a control
- predictor MSE against an identity baseline
- response in z to each action, decoded to kinematics through both a freshly fit state head
  and the model's own auxiliary head
- left/right symmetry and reverse-main sign of the thrust response
- action-magnitude linearity, with per-magnitude standard error
- multi-step rollout error growth
- response to out-of-distribution actions
- a state head trained on one checkpoint, applied to another
- sample sizes, and which tests ran, are unimplemented, or were skipped for missing
  prerequisites

Tests are grouped into clusters A–E, selectable with `--include-clusters`.

## Layout

```
lewm/
  eval/          suite runner, probes, rollout, planner eval, kinematic cost, video
  scripts/       CLI entry points (eval_suite.py is the orchestrator), data prep
  env/           Lunar Lander gym env returning a 15-dim state + synthetic render
  utils/         triangle-frame renderer matching the JEPA's training data
  ARCHITECTURE.md   code map: what is this repo's, what is the library's, what is patched upstream
  vendor/le-wm/     submodule — fork of LeWorldModel
dreamerv3/
  scripts/       training launcher + eval
  vendor/r2dreamer/ submodule — fork of R2-Dreamer
tests/           pytest suite over the eval code
```

## Setup

```bash
git clone --recurse-submodules https://github.com/saptaxis/world-model-audits.git
cd world-model-audits
cp env.sh.example env.sh   # edit paths for your machine
pip install -r requirements.txt
pip install -e .
```

Tests that need a GPU, a checkpoint or a dataset skip themselves when those are absent:

```bash
pytest                        # unit tests
pytest -m integration         # the rest, where the data is present
```

## Provenance

Both vendored models are other people's work, tracked as submodules pointing at forks:

| Submodule | Upstream | Licence | What is changed in the fork |
|---|---|---|---|
| `lewm/vendor/le-wm` | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) — Maes, Le Lidec, Scieur, LeCun, Balestriero | MIT, © Lucas Maes | Lunar Lander train/eval hydra configs; `eval.py` routing for a replay eval loop and a kinematic cost model; an optional auxiliary kinematic loss and a dedicated `z_kin` subspace in `module.py`/`train.py` |
| `dreamerv3/vendor/r2dreamer` | [NM512/r2dreamer](https://github.com/NM512/r2dreamer) — Naoki Morihira | MIT, © Naoki Morihira | A Lunar Lander env wrapper and config for state-vector training |

Each fork keeps its upstream commit history and its upstream `LICENSE` file. The changes are
listed in each fork's README. `lewm/ARCHITECTURE.md` breaks the boundary down file by file.

## Licence

MIT — see [LICENSE](LICENSE). This covers the code in this repository. The vendored
submodules are separately licensed by their authors, as listed above.
