#!/usr/bin/env python3
"""Convert episodes to synthetic-frame HDF5 for LeWorldModel.

Reads state vectors from .npz episodes, renders synthetic frames
(large triangle on black background), writes HDF5 in the same format
as convert_npz_to_hdf5.py.

Usage:
    python lewm/scripts/convert_synthetic_to_hdf5.py \
        --input-dirs /path/to/heuristic /path/to/random \
        --output /path/to/datasets/lunarlander_synthetic_heuristic.h5 \
        --triangle-radius 35
"""

import argparse
import glob
import sys
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.synthetic_render import render_episode_synthetic


def convert(
    input_dirs: list[str],
    output_path: str,
    triangle_radius: int = 35,
    size: int = 224,
    limit: int = 0,
):
    """Convert episodes to synthetic-frame HDF5."""
    episode_paths = []
    for d in input_dirs:
        paths = sorted(glob.glob(str(Path(d) / "episode_*.npz")))
        print(f"  {Path(d).name}: {len(paths)} episodes")
        episode_paths.extend(paths)

    if limit > 0:
        episode_paths = episode_paths[:limit]
        print(f"Limited to first {limit} episodes")
    print(f"Total: {len(episode_paths)} episodes")

    if not episode_paths:
        raise ValueError("No episodes found")

    # First pass: compute sizes
    ep_lens = []
    for path in tqdm(episode_paths, desc="Scanning"):
        ep = np.load(path, mmap_mode="r")
        ep_lens.append(ep["states"].shape[0])  # T+1 frames

    ep_lens = np.array(ep_lens, dtype=np.int64)
    ep_offsets = np.concatenate([[0], np.cumsum(ep_lens[:-1])]).astype(np.int64)
    total_frames = int(ep_lens.sum())
    action_dim = 2

    print(f"Total frames: {total_frames}, triangle_radius: {triangle_radius}px")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        pixels_ds = f.create_dataset(
            "pixels",
            shape=(total_frames, size, size, 3),
            dtype=np.uint8,
            chunks=(64, size, size, 3),
            **hdf5plugin.Blosc(
                cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        action_ds = f.create_dataset(
            "action",
            shape=(total_frames, action_dim),
            dtype=np.float32,
        )
        state_ds = f.create_dataset(
            "state",
            shape=(total_frames, 15),
            dtype=np.float32,
        )

        idx = 0
        for path in tqdm(episode_paths, desc="Rendering"):
            ep = np.load(path)
            states = ep["states"]       # (T+1, 15)
            actions = ep["actions"]     # (T, 2)
            T_plus_1 = len(states)

            # Render synthetic frames from states
            frames = render_episode_synthetic(
                states, size=size, triangle_radius=triangle_radius
            )

            # Pad actions with NaN for final frame
            actions_padded = np.full(
                (T_plus_1, action_dim), np.nan, dtype=np.float32
            )
            actions_padded[: len(actions)] = actions

            pixels_ds[idx : idx + T_plus_1] = frames
            action_ds[idx : idx + T_plus_1] = actions_padded
            state_ds[idx : idx + T_plus_1] = states
            idx += T_plus_1

        f.create_dataset("ep_len", data=ep_lens)
        f.create_dataset("ep_offset", data=ep_offsets)

    print(f"Written to {output_path}")
    print(f"  {len(ep_lens)} episodes, {total_frames} total frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--triangle-radius", type=int, default=35)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0, help="Max episodes (0=all)")
    args = parser.parse_args()
    convert(args.input_dirs, args.output, args.triangle_radius, args.size, args.limit)


if __name__ == "__main__":
    main()
