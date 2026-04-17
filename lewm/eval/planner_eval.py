"""Custom replay-mode eval for JEPA CEM planning.

Fixes the start-state mismatch in the original replay implementation by
seeding env.reset() with the dataset's per-episode seed. Logs per-step
planner state/action/cost and the dataset's reference trajectory.

No video rendering here — see lewm/scripts/viz_planning_rollout.py.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch


RESULT_KEYS = (
    "ep_seeds",
    "planner_states",
    "planner_actions",
    "planner_costs",
    "dataset_states",
    "dataset_goal_states",
    "dataset_goal_pixels",
    "success_z",
    "success_kin",
    "final_z_distance",
    "final_kin_distance",
)


@dataclass
class ReplayConfig:
    """Configuration for evaluate_replay."""
    h5_path: str
    n_episodes: int
    goal_offset_steps: int   # N output steps ahead
    eval_budget: int         # total output steps to run planner for
    horizon: int             # CEM planning horizon
    frameskip: int
    z_success_threshold: float  # ||z_planner - z_goal|| success bound
    kin_success_threshold: float
    kin_weights: np.ndarray  # (6,) weighting for kinematic distance
    seed: int  # required — caller must pass (typically from cfg.seed)


def evaluate_replay(
    world: Any,
    policy: Any,
    model: Any,
    cfg: ReplayConfig,
    device: str = "cuda",
) -> dict:
    """Run replay-mode evaluation with seed-based start.

    Args:
        world: swm.World instance already configured with num_envs = n_episodes.
        policy: swm.policy.WorldModelPolicy wrapping the CEM solver + model.
        model: LeWM instance with .encode() and access to projector.
        cfg: ReplayConfig.
        device: torch device for encoding.

    **Env contract:** Assumes `world.envs` is wrapped by `MegaWrapper` with
    `history_size >= 1`. The function injects `dataset_goal_pixels` into
    `world.infos["goal"]` each step, broadcasting to shape
    `(n_envs, history_size, H, W, 3)` to match the history dim.

    **Action capture:** The wrapper accumulates all calls to
    `policy.get_action` within each `world.step()` — so we log per-step
    actions of shape `(n_envs, action_block, action_dim)` rather than only
    the last action. If the capture buffer length after a step does not
    equal `cfg.frameskip`, the logged row falls back to the last captured
    action broadcast across the block dim (and a warning is emitted once).

    Returns:
        dict with keys per RESULT_KEYS. Also saved externally by caller if needed.
    """
    # --- load dataset slice for each selected episode ---
    with h5py.File(cfg.h5_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        ep_seed = f["ep_seed"][:]
        state_all = f["state"]  # keep as h5 handle; read slices below
        pixels_all = f["pixels"]

        # Only need enough dataset for the goal frame (goal_offset_steps ahead).
        # The planner runs eval_budget output steps in the real env; past the
        # episode's end there's simply no dataset reference — dataset_states
        # gets NaN-padded beyond the episode's actual length.
        min_span = (cfg.goal_offset_steps + 1) * cfg.frameskip
        valid = np.where(ep_len >= min_span)[0]
        if len(valid) < cfg.n_episodes:
            raise ValueError(
                f"Only {len(valid)} episodes long enough to reach goal_offset "
                f"(need {min_span} raw steps); requested {cfg.n_episodes}"
            )
        rng = np.random.default_rng(cfg.seed)
        picks = np.sort(rng.choice(valid, size=cfg.n_episodes, replace=False))
        ep_seeds_selected = ep_seed[picks].astype(np.int64)

        # Build reference arrays: dataset state at steps 0..eval_budget (NaN-padded
        # beyond each episode's length) and goal frame at goal_offset_steps.
        T = cfg.eval_budget
        state_dim = state_all.shape[1]
        dataset_states = np.full(
            (cfg.n_episodes, T + 1, state_dim), np.nan, dtype=np.float32
        )
        dataset_goal_states = np.zeros((cfg.n_episodes, state_dim), dtype=np.float32)
        dataset_goal_pixels = np.zeros(
            (cfg.n_episodes, pixels_all.shape[1], pixels_all.shape[2], 3),
            dtype=np.uint8,
        )
        for i, ep_idx in enumerate(picks):
            base = int(ep_offset[ep_idx])
            raw_len = int(ep_len[ep_idx])
            # Fill dataset_states up to the episode's actual length (in output
            # steps). Beyond that, leave NaN so the video renderer / downstream
            # consumers can detect missing reference.
            max_output_steps = raw_len // cfg.frameskip
            n_ref = min(T + 1, max_output_steps)
            indices = np.arange(n_ref) * cfg.frameskip + base
            dataset_states[i, :n_ref] = state_all[indices]
            goal_idx = base + cfg.goal_offset_steps * cfg.frameskip
            dataset_goal_states[i] = state_all[goal_idx]
            dataset_goal_pixels[i] = pixels_all[goal_idx]

    # --- override max_episode_steps so the env doesn't truncate before eval_budget ---
    # eval.py sets max_episode_steps = 2 * eval_budget (treating it as raw steps).
    # We need eval_budget * frameskip raw steps. Set generously.
    needed_raw_steps = cfg.eval_budget * cfg.frameskip + 100
    for env in world.envs.envs:
        if hasattr(env, '_max_episode_steps'):
            env._max_episode_steps = needed_raw_steps
        if hasattr(env, 'spec') and env.spec is not None:
            env.spec.max_episode_steps = needed_raw_steps

    # --- reset env with per-episode seeds (array of seeds, one per env) ---
    # Use world.reset (not world.envs.reset) so world.infos/states are populated.
    world.reset(seed=ep_seeds_selected.tolist())

    # Enable autoreset so terminated envs reset automatically on next step.
    # Without this, SyncVectorEnv asserts on stepping a terminated env.
    # Terminated envs' post-reset data is ignored (masked by done_at).
    import gymnasium as gym
    world.envs.unwrapped.autoreset_mode = gym.vector.AutoresetMode.NEXT_STEP

    # --- encode goal frames once (batched) ---
    goal_pixels_t = torch.from_numpy(dataset_goal_pixels).float().to(device)
    goal_pixels_t = goal_pixels_t.permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    goal_pixels_t = (goal_pixels_t - mean) / std
    with torch.no_grad():
        enc = model.encoder(goal_pixels_t, interpolate_pos_encoding=True)
        z_goals = model.projector(enc.last_hidden_state[:, 0])  # (n_episodes, D)

    # --- allocate per-step logs ---
    planner_states = np.zeros(
        (cfg.n_episodes, T + 1, state_dim), dtype=np.float32
    )
    # Log all action_block actions per outer step. world.step() calls
    # policy.get_action `action_block` times internally; the wrapper below
    # accumulates every call into action_buffer.
    act_dim = int(np.prod(world.envs.single_action_space.shape))
    planner_actions = np.zeros(
        (cfg.n_episodes, T, cfg.frameskip, act_dim),
        dtype=np.float32,
    )
    # NaN placeholder: distinguishes "no signal" from "actually zero".
    planner_costs = np.full((cfg.n_episodes, T), np.nan, dtype=np.float32)
    _cost_available = hasattr(policy, "solver") and hasattr(
        policy.solver, "last_cost"
    )
    if not _cost_available:
        warnings.warn(
            "planner_costs will be NaN — swm solver does not expose last_cost",
            stacklevel=2,
        )

    # Record state 0 from env infos
    initial_info = world.infos
    planner_states[:, 0, :] = _extract_state(initial_info, cfg.n_episodes, state_dim)

    # Prepare goal to inject into info each step. MegaWrapper yields
    # info['pixels'] with shape (n_envs, history, H, W, 3); goal should
    # broadcast to the same shape so WorldModelPolicy's transform works.
    pixels_shape = np.asarray(world.infos["pixels"]).shape  # (n, H_hist, H, W, 3)
    history = pixels_shape[1]
    goal_broadcast = np.broadcast_to(
        dataset_goal_pixels[:, None, ...], (cfg.n_episodes, history) + dataset_goal_pixels.shape[1:]
    ).copy()

    # --- run planner loop ---
    world.set_policy(policy)
    # Wrap get_action to accumulate every per-inner-step action emitted by
    # the planner within a given world.step().
    action_buffer: list[np.ndarray] = []
    _orig_get_action = policy.get_action

    def _capture_get_action(info):
        a = _orig_get_action(info)
        action_buffer.append(np.asarray(a))
        return a

    _warned_block_mismatch = False
    try:
        policy.get_action = _capture_get_action  # type: ignore[assignment]

        # Per-env tracking: done_at[i] = output step at which env i terminated.
        # T means it ran the full budget without terminating.
        done_at = np.full(cfg.n_episodes, T, dtype=np.int32)
        all_done = np.zeros(cfg.n_episodes, dtype=bool)

        for t in range(T):
            # Inject goal into infos so WorldModelPolicy's get_action sees it.
            world.infos["goal"] = goal_broadcast
            action_buffer.clear()
            # world.step() triggers policy.get_action `action_block` times
            world.step()

            info = world.infos
            new_states = _extract_state(info, cfg.n_episodes, state_dim)

            # Check per-env termination. Record final state for newly-done envs.
            termed = np.asarray(world.terminateds) if world.terminateds is not None else np.zeros(cfg.n_episodes, dtype=bool)
            trunc = np.asarray(world.truncateds) if world.truncateds is not None else np.zeros(cfg.n_episodes, dtype=bool)
            newly_done = (termed | trunc) & ~all_done

            # For still-active envs, log their state. For newly-done envs,
            # log their LAST valid state (before autoreset replaced it).
            # With NEXT_STEP autoreset, the observation at this step is
            # still the terminal observation — autoreset happens on next step().
            planner_states[:, t + 1, :] = new_states

            if np.any(newly_done):
                done_at[newly_done] = t + 1
                all_done |= newly_done

            if np.all(all_done):
                break
            # Stack captured actions. Each buffered action has shape
            # (n_envs, act_dim); buffer length should equal cfg.frameskip.
            if len(action_buffer) == cfg.frameskip:
                arr = np.stack(action_buffer, axis=0)  # (block, n_envs, act_dim)
                arr = arr.reshape(cfg.frameskip, cfg.n_episodes, act_dim)
                planner_actions[:, t, :, :] = np.transpose(arr, (1, 0, 2))
            elif len(action_buffer) > 0:
                if not _warned_block_mismatch:
                    warnings.warn(
                        f"action_buffer has {len(action_buffer)} entries, "
                        f"expected {cfg.frameskip}; broadcasting last action "
                        f"across block dim",
                        stacklevel=2,
                    )
                    _warned_block_mismatch = True
                last = np.asarray(action_buffer[-1]).reshape(
                    cfg.n_episodes, act_dim
                )
                planner_actions[:, t, :, :] = last[:, None, :]
            if _cost_available:
                planner_costs[:, t] = np.asarray(policy.solver.last_cost)
    finally:
        policy.get_action = _orig_get_action  # type: ignore[assignment]

    # --- compute final distances + success flags ---
    # z_planner at final step: re-encode final observation
    final_pixels = np.array(
        [_render_from_env(world, i) for i in range(cfg.n_episodes)]
    )
    final_pixels_t = torch.from_numpy(final_pixels).float().to(device)
    final_pixels_t = final_pixels_t.permute(0, 3, 1, 2) / 255.0
    final_pixels_t = (final_pixels_t - mean) / std
    with torch.no_grad():
        enc_final = model.encoder(final_pixels_t, interpolate_pos_encoding=True)
        z_final = model.projector(enc_final.last_hidden_state[:, 0])

    final_z_distance = torch.norm(z_final - z_goals, dim=-1).cpu().numpy()

    # Per-env final state at each env's actual termination step
    planner_final_state = np.array([
        planner_states[i, done_at[i], :6] for i in range(cfg.n_episodes)
    ])
    dataset_goal_kin = dataset_goal_states[:, :6]
    kin_diff = (planner_final_state - dataset_goal_kin) * cfg.kin_weights[None, :]
    final_kin_distance = np.linalg.norm(kin_diff, axis=-1).astype(np.float32)

    success_z = final_z_distance < cfg.z_success_threshold
    success_kin = final_kin_distance < cfg.kin_success_threshold

    return {
        "ep_seeds": ep_seeds_selected,
        "done_at": done_at,  # (n_episodes,) int32 — per-env termination step
        "planner_states": planner_states,
        "planner_actions": planner_actions,
        "planner_costs": planner_costs,
        "dataset_states": dataset_states,
        "dataset_goal_states": dataset_goal_states,
        "dataset_goal_pixels": dataset_goal_pixels,
        "success_z": success_z,
        "success_kin": success_kin,
        "final_z_distance": final_z_distance.astype(np.float32),
        "final_kin_distance": final_kin_distance,
    }


def _extract_state(info: dict, n_envs: int, state_dim: int) -> np.ndarray:
    """Pull per-env state from wrapped `info` dict. Shape (n_envs, state_dim)."""
    # MegaWrapper stacks state history into info['state'] with shape
    # (n_envs, history_size, state_dim) — take the most recent.
    arr = info.get("state")
    if arr is None:
        raise KeyError("info dict has no 'state' key")
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[:, -1, :]
    return arr.astype(np.float32)


def _render_from_env(world: Any, env_idx: int) -> np.ndarray:
    """Grab the latest pixel frame for a specific env index. Shape (H, W, 3) uint8."""
    info = world.infos
    pix = info.get("pixels")
    if pix is None:
        raise KeyError("info dict has no 'pixels' key")
    pix = np.asarray(pix)
    if pix.ndim == 5:  # (n_envs, history, H, W, 3)
        pix = pix[:, -1]
    return pix[env_idx]
