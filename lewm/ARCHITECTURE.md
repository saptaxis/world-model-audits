# LeWM Planning Architecture — Code Map

> How our CEM planning pipeline integrates with the LeWM / stable_worldmodel library.

## Overview

```
eval_planning.sh                    ← entry point (bash launcher)
  └→ lewm/vendor/le-wm/eval.py     ← hydra-driven eval script (MODIFIED from upstream)
       ├→ swm.World                 ← library: vectorized gym env wrapper
       │    └→ LunarLanderSynthetic ← OURS: custom gym env (lewm/env/)
       ├→ WorldModelPolicy          ← library: CEM solver wrapper
       │    └→ CEMSolver            ← library: cross-entropy method
       │         └→ model.get_cost  ← OURS or library (see below)
       └→ planner_eval.evaluate_replay  ← OURS: custom eval loop (lewm/eval/)
            └→ viz_planning_rollout ← OURS: post-hoc video renderer
```

## What's ours vs. library

### Library (pip-installed, don't modify)

Located in `~/virtual_envs/lewm/lib/.../stable_worldmodel/`:

| Component | What it does | Key API |
|---|---|---|
| `swm.World` | Wraps N gym envs as vectorized batch. Handles reset, step, info dict management. | `world.reset()`, `world.step()`, `world.infos` |
| `swm.policy.WorldModelPolicy` | Wraps a CEM solver + model. Called by `world.step()` to get actions. Manages action buffer (action_block unrolling). | `policy.get_action(info_dict)` |
| `swm.solver.CEMSolver` | Cross-entropy method. Samples action candidates, rolls out via `model.get_cost()`, refits Gaussian. **Iterates envs with batch_size=1 by default.** | `solver(info_dict, init_action)` |
| `swm.policy.AutoCostModel` | Loads a LeWM checkpoint and returns the model with `get_cost` method. | `model = AutoCostModel("path/to/ckpt")` |
| `swm.data.HDF5Dataset` | Loads our HDF5 datasets. | `dataset[idx]` |
| MegaWrapper chain | `AddPixelsWrapper` → `EverythingToInfoWrapper` → `StackedWrapper` → env. Adds pixels via `env.render()`, stacks history, moves obs to info dict. | Transparent — wraps our env automatically |

**Key behavior:** CEM solver calls `model.get_cost(info_dict, action_candidates)` inside its inner loop. `info_dict` is sliced per-env (batch_size=1), so `get_cost` sees B=1. This is why per-env data must go through `info_dict`, not module buffers.

### Ours — custom code

Located in `~/Dropbox/code/world-model-audits/lewm/`:

| File | What it does | Why custom |
|---|---|---|
| **`env/lunarlander_env.py`** | Gym env wrapper. Returns `{"state": (15,)}` obs (not pixels — MegaWrapper adds those via `render()`). `render()` draws synthetic triangle frame matching JEPA training data. Has `_set_state()` for replay-forward and `_set_goal_state()` for goal storage. | Library envs (pusht, cube) return different obs formats. Lunar Lander needs synthetic triangle rendering + 15-dim state. |
| **`eval/lewm_kinematic.py`** | `LeWMKinematic` — subclass of `LeWM`. Overrides `get_cost()` to decode z via state head → weighted kinematic MSE, instead of raw z-distance. Reads `kin_target` from `info_dict` (per-env). | Library's `LeWM.get_cost()` uses image-goal (z-distance). We need kinematic cost for explicit x/y/angle targeting. |
| **`eval/planner_eval.py`** | `evaluate_replay()` — custom eval loop replacing `world.evaluate_from_dataset()`. Seeds env with dataset's `ep_seed`, injects goal pixels + kinematic target into `world.infos`, runs MPC loop, logs per-step state/action/cost, handles per-env termination. | Library's `evaluate_from_dataset` had start-state mismatch (broken `_set_state` teleport). Our loop uses seed-based deterministic reset. |
| **`eval/rollout_viz.py`** | `render_planner_trajectory_video()` — PIL-based schematic renderer. Left panel: world view with planner (red) + heuristic (blue) triangles, trails, goal star. Right panel: kinematics, actions, cost. | Library's video output is minimal (4 raw frames in a grid, no labels). |
| **`eval/state_head.py`** | `StateHead` MLP/linear probe. Maps z (192-dim) → kinematics (6-dim). Used by `LeWMKinematic` for cost computation. | No equivalent in library. |
| **`scripts/eval_planning.sh`** | Bash launcher. Sets STABLEWM_HOME, PYTHONPATH, resolves repo root dynamically (works on host + scad). | Convenience wrapper around `python eval.py`. |
| **`scripts/viz_planning_rollout.py`** | CLI for post-hoc video rendering from saved npz logs. | Separate from eval — can re-render without re-running planning. |
| **`scripts/augment_hdf5_seeds.py`** | One-shot: adds `ep_seed` column to HDF5 from source .npz metadata. | Library datasets don't have per-episode seeds. |

### Modified upstream (in submodule `lewm/vendor/le-wm/`)

| File | What we changed | Why |
|---|---|---|
| **`eval.py`** | Added `import lewm.env` (env registration). Added branch on `cfg.eval.h5_path` → routes replay mode to `planner_eval.evaluate_replay`. Added branch on `state_head_path` → loads `LeWMKinematic` instead of `AutoCostModel`. Removed inner `from pathlib import Path` that shadowed module-level import. | Library's eval.py only supported `evaluate_from_dataset`. We need custom replay loop + kinematic cost model. |
| **`config/eval/lunarlander_replay.yaml`** | New config for replay mode. Adds `h5_path`, `logs_path`, `z_success_threshold`, `kin_success_threshold`, `kin_weights`, `kinematic_target`, `kinematic_weights`, `state_head_path`. | No Lunar Lander config existed in upstream. |
| **`config/eval/lunarlander_synthetic.yaml`** | New config for synthetic (kinematic target) mode. | Same reason. |

## Data flow during a CEM planning step

```
1. world.step() called by planner_eval loop
   └→ policy.get_action(world.infos)
       ├→ world.infos contains:
       │   "pixels": (n_envs, history, H, W, 3)  ← from MegaWrapper/AddPixelsWrapper
       │   "state":  (n_envs, history, 15)        ← from EverythingToInfoWrapper
       │   "goal":   (n_envs, history, H, W, 3)   ← injected by planner_eval
       │   "kin_target": (n_envs, 6)               ← injected by planner_eval (kinematic mode)
       │
       └→ solver(info_dict)  [CEMSolver]
           ├→ for env_idx in range(n_envs):        ← CEM iterates envs, batch_size=1
           │   ├→ slice info_dict to B=1
           │   ├→ sample 300 action candidates: (1, 300, horizon, action_dim=20)
           │   ├→ model.get_cost(info_slice, candidates)
           │   │   ├→ [image-goal mode] LeWM.get_cost:
           │   │   │   encode goal pixels → z_goal
           │   │   │   rollout(z_current, candidates) → predicted z trajectories
           │   │   │   cost = ||z_final - z_goal||  per candidate
           │   │   │
           │   │   └→ [kinematic mode] LeWMKinematic.get_cost:
           │   │       rollout(z_current, candidates) → predicted z trajectories
           │   │       decode z_final via state_head → predicted kinematics
           │   │       read kin_target from info_dict (1, 6)
           │   │       cost = weighted_MSE(predicted_kin, kin_target)
           │   │
           │   ├→ rank candidates by cost, keep top 30 elites
           │   ├→ refit Gaussian to elites
           │   └→ repeat 30 iterations
           │
           └→ return best first action per env

2. world.envs.step(action)  ← 10 raw env steps (action_block=10)
   └→ each env.step returns new obs, reward, terminated, truncated, info
       └→ MegaWrapper adds pixels via env.render() (synthetic triangle)

3. planner_eval logs: planner_states, planner_actions, done_at
```

## Action shape at frameskip=10

- JEPA trained with `frameskip=10, action_dim=2` → `action_encoder` input is 20-dim (10 raw actions concatenated)
- CEM samples candidates of shape `(B, S, horizon, 20)` — each "action" is 20-dim
- `WorldModelPolicy` reshapes the chosen plan into individual 2D actions via its internal buffer
- `world.step()` calls `env.step(2D_action)` 10 times per outer step
- Each of the 10 env steps CAN have a different action (not repeated)

## Two cost modes

| Mode | Model class | Cost function | When to use |
|---|---|---|---|
| **Image-goal** | `LeWM` (upstream) | `\|z_pred - z_goal\|` over 192 dims | Default replay mode (no state_head_path) |
| **Kinematic** | `LeWMKinematic` (ours) | `Σ w_i * (sh(z_pred)_i - target_i)²` over 6 kinematic dims | Pass `state_head_path` on CLI |

Image-goal cost is dominated by non-position z-dimensions (appearance features). Kinematic cost explicitly weights x, y, vx, vy, angle.

## Per-env termination handling

`AutoresetMode.NEXT_STEP` — terminated envs auto-reset on next step. `done_at[i]` records when each env terminated. The eval loop continues until all envs are done or `eval_budget` is reached. Video renderer trims each episode to its `done_at`.

## File locations recap

```
world-model-audits/                         ← parent repo (main branch)
├── lewm/
│   ├── env/
│   │   └── lunarlander_env.py              ← gym env wrapper
│   ├── eval/
│   │   ├── planner_eval.py                 ← custom replay eval loop
│   │   ├── lewm_kinematic.py               ← kinematic cost subclass
│   │   ├── state_head.py                   ← MLP/linear probe z→kinematics
│   │   ├── rollout_viz.py                  ← video renderer (schematic + info panel)
│   │   ├── rollout.py                      ← z-space rollout (for non-planning viz)
│   │   └── encode_dataset.py               ← bulk HDF5→z encoding
│   ├── scripts/
│   │   ├── eval_planning.sh                ← bash launcher
│   │   ├── viz_planning_rollout.py         ← post-hoc video CLI
│   │   ├── augment_hdf5_seeds.py           ← one-shot seed augmentation
│   │   ├── train_state_head.py             ← train z→kinematic probe
│   │   └── viz_tsne.py                     ← t-SNE visualization
│   ├── utils/
│   │   └── synthetic_render.py             ← triangle-on-black frame renderer
│   └── vendor/le-wm/                       ← submodule (lunar-lander branch)
│       ├── eval.py                         ← MODIFIED: replay routing + kinematic model
│       ├── config/eval/
│       │   ├── lunarlander_replay.yaml     ← NEW
│       │   └── lunarlander_synthetic.yaml  ← NEW
│       └── (rest is upstream LeWM code)
├── tests/
│   ├── test_augment_hdf5_seeds.py
│   ├── test_seed_determinism.py
│   ├── test_planner_eval.py
│   └── test_lunarlander_env.py
└── scad-wm-audits.yml                      ← scad container config
```
