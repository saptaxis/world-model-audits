"""Custom replay-mode eval for JEPA CEM planning.

Fixes the start-state mismatch in the original replay implementation by
seeding env.reset() with the dataset's per-episode seed. Logs per-step
planner state/action/cost and the dataset's reference trajectory.

No video rendering here — see lewm/scripts/viz_planning_rollout.py.
"""
from __future__ import annotations

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
    seed: int = 42


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

        # Sample episodes that are long enough
        span = (cfg.goal_offset_steps + cfg.eval_budget + 1) * cfg.frameskip
        valid = np.where(ep_len >= span)[0]
        if len(valid) < cfg.n_episodes:
            raise ValueError(
                f"Only {len(valid)} episodes long enough for span={span}; "
                f"need {cfg.n_episodes}"
            )
        rng = np.random.default_rng(cfg.seed)
        picks = np.sort(rng.choice(valid, size=cfg.n_episodes, replace=False))
        ep_seeds_selected = ep_seed[picks].astype(np.int64)

        # Build reference arrays: dataset state at steps 0..eval_budget and goal
        T = cfg.eval_budget
        state_dim = state_all.shape[1]
        dataset_states = np.zeros((cfg.n_episodes, T + 1, state_dim), dtype=np.float32)
        dataset_goal_states = np.zeros((cfg.n_episodes, state_dim), dtype=np.float32)
        dataset_goal_pixels = np.zeros(
            (cfg.n_episodes, pixels_all.shape[1], pixels_all.shape[2], 3),
            dtype=np.uint8,
        )
        for i, ep_idx in enumerate(picks):
            base = int(ep_offset[ep_idx])
            for t in range(T + 1):
                dataset_states[i, t] = state_all[base + t * cfg.frameskip]
            goal_idx = base + cfg.goal_offset_steps * cfg.frameskip
            dataset_goal_states[i] = state_all[goal_idx]
            dataset_goal_pixels[i] = pixels_all[goal_idx]

    # --- reset env with per-episode seeds (array of seeds, one per env) ---
    # Use world.reset (not world.envs.reset) so world.infos/states are populated.
    world.reset(seed=ep_seeds_selected.tolist())

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
    # WorldModelPolicy returns one action per world.step() call (shape
    # (n_envs, act_dim)); the action_block expansion is already unrolled by
    # its internal buffer. Log one action per outer step.
    act_dim = int(np.prod(world.envs.single_action_space.shape))
    planner_actions = np.zeros(
        (cfg.n_episodes, T, act_dim),
        dtype=np.float32,
    )
    planner_costs = np.zeros((cfg.n_episodes, T), dtype=np.float32)

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
    # Wrap get_action to capture the last action emitted by the planner.
    last_action_box = {"val": None}
    _orig_get_action = policy.get_action

    def _capture_get_action(info):
        a = _orig_get_action(info)
        last_action_box["val"] = np.asarray(a)
        return a

    policy.get_action = _capture_get_action  # type: ignore[assignment]

    for t in range(T):
        # Inject goal into infos so WorldModelPolicy's get_action sees it.
        world.infos["goal"] = goal_broadcast
        # At each step, world.step() triggers policy.get_action(info) → solver.plan → env.step
        world.step()  # advances env by action_block env steps
        info = world.infos
        planner_states[:, t + 1, :] = _extract_state(info, cfg.n_episodes, state_dim)
        # Capture action from wrapper. Shape may be (n_envs, action_block, act_dim)
        # or (n_envs, act_dim*action_block); flatten to match log layout.
        la = last_action_box["val"]
        if la is not None:
            planner_actions[:, t, :] = la.reshape(cfg.n_episodes, -1)
        # planner_costs: solver usually does not expose last_cost; leave zeros.
        if hasattr(policy, "solver") and hasattr(policy.solver, "last_cost"):
            planner_costs[:, t] = np.asarray(policy.solver.last_cost)

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

    planner_final_state = planner_states[:, -1, :6]
    dataset_goal_kin = dataset_goal_states[:, :6]
    kin_diff = (planner_final_state - dataset_goal_kin) * cfg.kin_weights[None, :]
    final_kin_distance = np.linalg.norm(kin_diff, axis=-1).astype(np.float32)

    success_z = final_z_distance < cfg.z_success_threshold
    success_kin = final_kin_distance < cfg.kin_success_threshold

    return {
        "ep_seeds": ep_seeds_selected,
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
