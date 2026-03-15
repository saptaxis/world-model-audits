# CLAUDE.md — world-model-audits

## Project Overview

Repo for auditing external world models using our physics evaluation suite. Each subdirectory under the root corresponds to one external model being audited.

Current audits:
- `dreamerv3/` — DreamerV3 via r2dreamer on Lunar Lander

## Development Environment

**Working directory:** This repo is the working directory. Never `cd` away from it.

**Virtualenv:**
```
if [ -f /opt/venv/bin/activate ]; then
  echo "scad container — venv already active"
else
  source ~/virtual_envs/r2dreamer/bin/activate
fi
```

**GPU:** 4090 primary, 2080ti secondary.

## Repos

- **This repo:** External model audit code, env wrappers, adapters, eval scripts
- **lwg:** Lunar Lander env, physics tests, probing tools
- **wm-ladder:** Our own world models (ladder), adapter interfaces
- **r2dreamer:** Vendored at `dreamerv3/vendor/r2dreamer/` as git submodule (pinned at `1fadce4`)

## Run Storage

`/media/hdd1/physics-priors-latent-space/world-model-audit-runs/dreamerv3/`

## Training

Train DreamerV3 on LunarLander from state vectors:
```bash
cd dreamerv3/vendor/r2dreamer
python train.py env=lunar_lander model.rep_loss=dreamer seed=42 logdir=/media/hdd1/physics-priors-latent-space/world-model-audit-runs/dreamerv3/s42
```

Or use the convenience script:
```bash
bash dreamerv3/scripts/train.sh 42
```

## Commands

All commands run from repo root. Exception: `train.py` runs from `dreamerv3/vendor/r2dreamer/` because hydra resolves configs relative to that directory.
