"""Tests for HDF5 seed-column augmentation."""
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — registers plugins for HDF5 I/O
import numpy as np


def _make_fake_source_episode(tmp_path: Path, idx: int, seed: int, n_steps: int = 5):
    """Write a minimal episode_*.npz with metadata_json containing a seed."""
    path = tmp_path / f"episode_{idx:05d}.npz"
    np.savez(
        path,
        states=np.zeros((n_steps, 6), dtype=np.float32),
        actions=np.zeros((n_steps, 2), dtype=np.float32),
        rewards=np.zeros(n_steps, dtype=np.float32),
        dones=np.zeros(n_steps, dtype=bool),
        metadata_json=json.dumps({"seed": seed}),
        rgb_frames=np.zeros((n_steps, 8, 8, 3), dtype=np.uint8),
    )
    return path


def _make_fake_hdf5(tmp_path: Path, n_episodes: int, frames_per_ep: int = 5):
    path = tmp_path / "fake.h5"
    with h5py.File(path, "w") as f:
        total = n_episodes * frames_per_ep
        f.create_dataset("pixels", data=np.zeros((total, 8, 8, 3), dtype=np.uint8))
        f.create_dataset("state", data=np.zeros((total, 6), dtype=np.float32))
        f.create_dataset("action", data=np.zeros((total, 2), dtype=np.float32))
        f.create_dataset("ep_len", data=np.full(n_episodes, frames_per_ep, dtype=np.int32))
        offsets = np.arange(n_episodes, dtype=np.int64) * frames_per_ep
        f.create_dataset("ep_offset", data=offsets)
    return path


def test_augment_adds_ep_seed_column(tmp_path):
    from lewm.scripts.augment_hdf5_seeds import augment_with_seeds

    src_dir = tmp_path / "source"
    src_dir.mkdir()
    for i in range(3):
        _make_fake_source_episode(src_dir, idx=i, seed=i * 10)

    h5 = _make_fake_hdf5(tmp_path, n_episodes=3)

    augment_with_seeds(h5_path=h5, source_dir=src_dir)

    with h5py.File(h5, "r") as f:
        assert "ep_seed" in f
        seeds = f["ep_seed"][:]
    np.testing.assert_array_equal(seeds, np.array([0, 10, 20], dtype=np.int32))


def test_augment_is_idempotent(tmp_path):
    from lewm.scripts.augment_hdf5_seeds import augment_with_seeds

    src_dir = tmp_path / "source"
    src_dir.mkdir()
    for i in range(2):
        _make_fake_source_episode(src_dir, idx=i, seed=i)

    h5 = _make_fake_hdf5(tmp_path, n_episodes=2)

    augment_with_seeds(h5_path=h5, source_dir=src_dir)
    augment_with_seeds(h5_path=h5, source_dir=src_dir)

    with h5py.File(h5, "r") as f:
        assert "ep_seed" in f
        assert f["ep_seed"].shape == (2,)


def test_augment_slices_excess_source_episodes(tmp_path):
    from lewm.scripts.augment_hdf5_seeds import augment_with_seeds
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    for i in range(5):
        _make_fake_source_episode(src_dir, idx=i, seed=i)
    h5 = _make_fake_hdf5(tmp_path, n_episodes=3)
    augment_with_seeds(h5_path=h5, source_dir=src_dir)
    with h5py.File(h5, "r") as f:
        seeds = f["ep_seed"][:]
    np.testing.assert_array_equal(seeds, np.array([0, 1, 2], dtype=np.int32))
