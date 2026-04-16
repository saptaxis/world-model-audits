"""Tests for planner_eval replay loop."""
import numpy as np
import pytest


def test_module_imports():
    from lewm.eval.planner_eval import evaluate_replay
    assert callable(evaluate_replay)


def test_result_schema(tmp_path):
    """evaluate_replay returns a dict with the expected per-episode arrays."""
    from lewm.eval.planner_eval import RESULT_KEYS
    assert set(RESULT_KEYS) == {
        "ep_seeds",           # (n_episodes,) int64
        "planner_states",     # (n_episodes, T+1, state_dim) float32
        "planner_actions",    # (n_episodes, T, action_dim) float32
        "planner_costs",      # (n_episodes, T) float32 (CEM cost at each step)
        "dataset_states",     # (n_episodes, T+1, state_dim) float32 (reference)
        "dataset_goal_states",# (n_episodes, state_dim) float32 (goal at step N)
        "dataset_goal_pixels",# (n_episodes, H, W, 3) uint8 (goal frame)
        "success_z",          # (n_episodes,) bool
        "success_kin",        # (n_episodes,) bool
        "final_z_distance",   # (n_episodes,) float32
        "final_kin_distance", # (n_episodes,) float32
    }
