"""Tests for planner_eval replay loop."""
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import torch


def test_module_imports():
    from lewm.eval.planner_eval import evaluate_replay
    assert callable(evaluate_replay)


def test_result_schema(tmp_path):
    """evaluate_replay returns a dict with the expected per-episode arrays."""
    from lewm.eval.planner_eval import RESULT_KEYS
    assert set(RESULT_KEYS) == {
        "ep_seeds",           # (n_episodes,) int64
        "planner_states",     # (n_episodes, T+1, state_dim) float32
        "planner_actions",    # (n_episodes, T, frameskip, action_dim) float32
        "planner_costs",      # (n_episodes, T) float32 (NaN if solver lacks last_cost)
        "dataset_states",     # (n_episodes, T+1, state_dim) float32 (reference)
        "dataset_goal_states",# (n_episodes, state_dim) float32 (goal at step N)
        "dataset_goal_pixels",# (n_episodes, H, W, 3) uint8 (goal frame)
        "success_z",          # (n_episodes,) bool
        "success_kin",        # (n_episodes,) bool
        "final_z_distance",   # (n_episodes,) float32
        "final_kin_distance", # (n_episodes,) float32
    }


def _make_fake_h5(path, n_eps=3, ep_len=40, state_dim=8, H=32, W=32):
    """Write a minimal fake dataset file matching the evaluate_replay schema."""
    total = n_eps * ep_len
    with h5py.File(path, "w") as f:
        f.create_dataset("ep_len", data=np.full(n_eps, ep_len, dtype=np.int64))
        f.create_dataset(
            "ep_offset",
            data=(np.arange(n_eps) * ep_len).astype(np.int64),
        )
        f.create_dataset(
            "ep_seed", data=np.arange(n_eps, dtype=np.int64) + 1000
        )
        rng = np.random.default_rng(0)
        f.create_dataset(
            "state",
            data=rng.standard_normal((total, state_dim)).astype(np.float32),
        )
        f.create_dataset(
            "pixels",
            data=rng.integers(0, 255, (total, H, W, 3), dtype=np.uint8),
        )


def test_evaluate_replay_end_to_end(tmp_path):
    """Smoke-test evaluate_replay with fully mocked world/policy/model."""
    from lewm.eval.planner_eval import (
        RESULT_KEYS,
        ReplayConfig,
        evaluate_replay,
    )

    h5_path = tmp_path / "fake.h5"
    n_eps = 3
    state_dim = 8
    H, W = 32, 32
    frameskip = 4
    T = 2  # eval_budget
    _make_fake_h5(str(h5_path), n_eps=n_eps, ep_len=40, state_dim=state_dim,
                  H=H, W=W)

    # --- Mock world (MegaWrapper-like infos) ---
    world = MagicMock()
    world.envs.single_action_space.shape = (2,)

    # info dict that mutates on step
    info_state = np.zeros((n_eps, 1, state_dim), dtype=np.float32)
    info_pixels = np.zeros((n_eps, 1, H, W, 3), dtype=np.uint8)
    world.infos = {"state": info_state, "pixels": info_pixels}

    def _step():
        # Simulate frameskip calls to policy.get_action per step
        for _ in range(frameskip):
            policy.get_action(world.infos)
        world.infos["state"] = world.infos["state"] + 0.01

    world.step.side_effect = _step
    world.reset = MagicMock()
    world.set_policy = MagicMock()

    # --- Mock policy (no solver.last_cost → NaN fallback) ---
    policy = MagicMock(spec=["get_action"])  # no .solver attr
    policy.get_action = MagicMock(
        return_value=np.zeros((n_eps, 2), dtype=np.float32)
    )

    # --- Mock model ---
    model = MagicMock()
    fake_enc = MagicMock()
    fake_enc.last_hidden_state = torch.zeros(n_eps, 1, 16)
    model.encoder.return_value = fake_enc
    model.projector.return_value = torch.zeros(n_eps, 16)

    cfg = ReplayConfig(
        h5_path=str(h5_path),
        n_episodes=n_eps,
        goal_offset_steps=T,
        eval_budget=T,
        horizon=5,
        frameskip=frameskip,
        z_success_threshold=1.0,
        kin_success_threshold=10.0,
        kin_weights=np.ones(6, dtype=np.float32),
        seed=0,
    )

    with pytest.warns(UserWarning, match="planner_costs will be NaN"):
        out = evaluate_replay(world, policy, model, cfg, device="cpu")

    # Shape / key checks
    assert set(out.keys()) == set(RESULT_KEYS)
    assert out["ep_seeds"].shape == (n_eps,)
    assert out["planner_states"].shape == (n_eps, T + 1, state_dim)
    assert out["planner_actions"].shape == (n_eps, T, frameskip, 2)
    assert out["planner_costs"].shape == (n_eps, T)
    assert np.all(np.isnan(out["planner_costs"]))
    assert out["dataset_states"].shape == (n_eps, T + 1, state_dim)
    assert out["dataset_goal_states"].shape == (n_eps, state_dim)
    assert out["dataset_goal_pixels"].shape == (n_eps, H, W, 3)
    assert out["success_z"].shape == (n_eps,)
    assert out["success_kin"].shape == (n_eps,)

    # Verify policy.get_action was restored
    assert policy.get_action is not None
