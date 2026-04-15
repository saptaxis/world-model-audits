"""Tests for Lunar Lander env wrapper with synthetic rendering."""
import gymnasium as gym
import numpy as np
import pytest

import lewm.env  # registers LunarLanderSynthetic-v0


def test_env_registered():
    """The synthetic env is registered with gymnasium."""
    env = gym.make("LunarLanderSynthetic-v0")
    assert env is not None
    env.close()


def test_env_observation_shape():
    """Env returns 224x224x3 uint8 pixels matching training data format."""
    env = gym.make("LunarLanderSynthetic-v0")
    obs, info = env.reset(seed=42)
    assert "pixels" in obs, f"Expected 'pixels' in obs dict, got {obs.keys()}"
    assert obs["pixels"].shape == (224, 224, 3)
    assert obs["pixels"].dtype == np.uint8
    env.close()


def test_env_state_in_info():
    """Info dict exposes 6-dim kinematic state for evaluation metrics."""
    env = gym.make("LunarLanderSynthetic-v0")
    obs, info = env.reset(seed=42)
    assert "state" in info, f"Expected 'state' in info, got {info.keys()}"
    assert info["state"].shape == (6,)
    env.close()


def test_env_action_space():
    """Continuous action space Box(-1, 1, (2,))."""
    env = gym.make("LunarLanderSynthetic-v0")
    assert env.action_space.shape == (2,)
    assert env.action_space.low.tolist() == [-1, -1]
    assert env.action_space.high.tolist() == [1, 1]
    env.close()


def test_env_step():
    """Env steps produce valid obs/reward/done tuples."""
    env = gym.make("LunarLanderSynthetic-v0")
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    assert obs["pixels"].shape == (224, 224, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    env.close()


def test_wrapper_render_matches_direct_render():
    """Frames from LunarLanderSynthetic-v0 must match render_synthetic_frame exactly."""
    from lewm.utils.synthetic_render import render_synthetic_frame

    env = gym.make("LunarLanderSynthetic-v0")
    obs, info = env.reset(seed=42)

    state = info["state"]
    direct_frame = render_synthetic_frame(
        x=float(state[0]),
        y=float(state[1]),
        angle=float(state[4]),
        size=224,
        triangle_radius=35,
    )
    np.testing.assert_array_equal(obs["pixels"], direct_frame)
    env.close()


def test_set_state_via_replay():
    """_set_state replays episode's first K actions to reach target state."""
    # Compute reference state by running a trajectory
    ref_env = gym.make("LunarLanderSynthetic-v0")
    ref_env.reset(seed=42)
    prefix_actions = np.array([[0.0, 0.0]] * 5, dtype=np.float32)
    for a in prefix_actions:
        _, _, _, _, ref_info = ref_env.step(a)
    target_state = ref_info["state"].copy()
    ref_env.close()

    # Now use _set_state with same prefix + seed
    env = gym.make("LunarLanderSynthetic-v0")
    env.unwrapped._set_state({
        "actions_prefix": prefix_actions,
        "reset_seed": 42,
    })
    # State after _set_state should match reference trajectory
    np.testing.assert_allclose(
        env.unwrapped._last_raw_state[:6], target_state, atol=1e-5,
        err_msg="Replay-forward did not arrive at the reference state"
    )
    env.close()
