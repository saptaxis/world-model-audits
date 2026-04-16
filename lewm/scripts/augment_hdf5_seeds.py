#!/usr/bin/env python3
"""Add ep_seed column to a Lunar Lander HDF5 dataset.

Reads seeds from each source episode_*.npz's metadata_json["seed"] field and
writes them to the HDF5 as a new column named "ep_seed" (shape: (n_episodes,),
dtype int32).

Idempotent: running twice leaves the HDF5 unchanged after the first run.
If the source has more episodes than the HDF5, the prefix is taken.

Usage:
    python lewm/scripts/augment_hdf5_seeds.py \\
        --h5 /media/hdd1/.../datasets/lunarlander_synthetic_heuristic.h5 \\
        --source-dir /media/hdd1/.../world_model_data/gym-default/gym-default-heuristic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — loads blosc/zstd filters
import numpy as np


def augment_with_seeds(h5_path: str | Path, source_dir: str | Path) -> None:
    """Add `ep_seed` column to `h5_path` reading seeds from `source_dir`."""
    h5_path = Path(h5_path)
    source_dir = Path(source_dir)

    episode_files = sorted(source_dir.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {source_dir}")

    with h5py.File(h5_path, "a") as f:
        n_episodes = int(f["ep_len"].shape[0])

        if len(episode_files) < n_episodes:
            raise ValueError(
                f"Source has fewer episodes ({len(episode_files)}) "
                f"than HDF5 ({n_episodes})"
            )
        if len(episode_files) > n_episodes:
            episode_files = episode_files[:n_episodes]

        seeds: list[int] = []
        for path in episode_files:
            with np.load(path, allow_pickle=True) as d:
                metadata = json.loads(str(d["metadata_json"]))
            seed_val = metadata.get("seed")
            if seed_val is None:
                raise ValueError(f"{path} has no 'seed' in metadata_json")
            seeds.append(int(seed_val))
        seeds_arr = np.asarray(seeds, dtype=np.int32)

        if "ep_seed" in f:
            existing = f["ep_seed"][:]
            if not np.array_equal(existing, seeds_arr):
                raise ValueError(
                    "ep_seed already exists but does not match computed seeds. "
                    "Refusing to overwrite."
                )
            return  # idempotent no-op
        f.create_dataset("ep_seed", data=seeds_arr)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--h5", required=True, help="Path to target HDF5 file")
    p.add_argument("--source-dir", required=True,
                   help="Directory of source episode_*.npz files")
    args = p.parse_args()
    augment_with_seeds(args.h5, args.source_dir)
    print(f"Augmented {args.h5} with ep_seed column.")


if __name__ == "__main__":
    main()
