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

`lewm/scripts/eval_suite.py` orchestrates the tests, caches each one's JSON output, and
renders a single report. Tests are grouped into five clusters, which map onto four questions:

| | Question | Cluster |
|---|---|---|
| **Q1** | Does the latent carry state at all? | **A** — linear probes from encoder-z and predicted-z to the six kinematic dims, per-dim R² |
| **Q2** | Is the predictor doing work beyond identity? | **B** — predictor-induced Δz, natural encoder drift as a control, and predictor MSE against an identity baseline |
| **Q3** | Is there a real action pathway? | **C** — z-space response to each action, decoded through both a freshly fit state head and the model's own aux head |
| **Q4** | Does the implied physics hold up? | **D** — action-magnitude linearity, multi-step rollout error growth, out-of-distribution actions |
| | Does the latent geometry transfer across runs? | **E** — a state head trained on one checkpoint, applied to another |

Some things the suite does deliberately:

- **Every claim has a null next to it.** "The predictor moved z" is only meaningful against
  how much z drifts on its own between adjacent frames, so both are measured and the report
  prints the ratio. Predictor MSE is scored against an identity baseline.
- **Symmetry and sign checks.** Left and right side thrust should mirror each other; reverse
  main thrust should flip sign. These catch a model that produces a response of the right
  magnitude in the wrong direction.
- **The report states its own coverage.** It prints which tests ran, which are unimplemented,
  and which were skipped for missing prerequisites, rather than rendering a partial run as a
  complete one.
- **Sample sizes are reported**, with per-magnitude standard error on the action sweep so
  noise is distinguishable from signal.
- **Scenarios with no eligible clips are skipped rather than averaged in.**

Results are cached per test; `--force` re-runs a selected subset, and `--report-only`
re-renders the report from JSONs already on disk.

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

If you use LeWorldModel, cite the authors:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Licence

MIT — see [LICENSE](LICENSE). This covers the code in this repository. The vendored
submodules are separately licensed by their authors, as listed above.
