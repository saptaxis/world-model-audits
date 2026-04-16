"""Verify ParametricLunarLander(seed=X) reproduces dataset state at step 0."""
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest

import gymnasium as gym

import lewm.env  # registers LunarLanderSynthetic-v0


@pytest.mark.integration
def test_reset_seed_matches_dataset_step_0():
    """Reset with ep_seed should produce state matching dataset's first row."""
    h5_path = (
        "/media/hdd1/physics-priors-latent-space/lunar-lander-data/datasets/"
        "lunarlander_synthetic_heuristic.h5"
    )
    with h5py.File(h5_path, "r") as f:
        if "ep_seed" not in f:
            pytest.skip("ep_seed not augmented yet")
        offsets = f["ep_offset"][:10]
        seeds = f["ep_seed"][:10]
        dataset_states_step0 = f["state"][offsets][:, :6]

    env = gym.make("LunarLanderSynthetic-v0")
    for ep_idx in range(10):
        obs, info = env.reset(seed=int(seeds[ep_idx]))
        env_state = obs["state"][:6]
        gt_state = dataset_states_step0[ep_idx]
        np.testing.assert_allclose(
            env_state, gt_state, atol=1e-4,
            err_msg=f"ep {ep_idx}: env reset state does not match dataset step 0"
        )
    env.close()
