#!/usr/bin/env python3
"""Convert Lunar Lander .npz episodes to HDF5 format for LeWorldModel.

Our frames are 400x600 (HxW). We pad to 600x600 (100px black bars top/bottom)
to preserve aspect ratio, then resize to target_size x target_size (default 224).

Usage:
    python lewm/scripts/convert_npz_to_hdf5.py \
        --input-dirs /path/to/heuristic /path/to/random \
        --output /path/to/datasets/lunarlander.h5 \
        --resize 224
"""

import argparse
import glob
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm


def load_episode(path: str) -> dict:
    """Load a single .npz episode file."""
    ep = np.load(path)
    return {
        "rgb_frames": ep["rgb_frames"],  # (T+1, 400, 600, 3)
        "actions": ep["actions"],  # (T, 2)
        "states": ep["states"],  # (T+1, 15)
    }


def pad_and_resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """Pad 400x600 frames to 600x600 (black bars top/bottom), then resize."""
    n, h, w, c = frames.shape
    assert h == 400 and w == 600, f"Expected 400x600, got {h}x{w}"
    # Pad: 100px black top, 100px black bottom → 600x600
    pad_top = (w - h) // 2  # 100
    pad_bottom = w - h - pad_top  # 100
    padded = np.zeros((n, w, w, c), dtype=np.uint8)
    padded[:, pad_top : pad_top + h, :, :] = frames
    # Resize to target
    out = np.empty((n, size, size, c), dtype=np.uint8)
    for i in range(n):
        out[i] = cv2.resize(padded[i], (size, size), interpolation=cv2.INTER_AREA)
    return out


def convert(
    input_dirs: list[str], output_path: str, resize: int = 224, limit: int = 0
):
    """Convert .npz episodes to a single HDF5 file."""
    # Collect all episode paths
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
    for path in tqdm(episode_paths, desc="Scanning episodes"):
        ep = np.load(path, mmap_mode="r")
        T_plus_1 = ep["rgb_frames"].shape[0]
        ep_lens.append(T_plus_1)

    ep_lens = np.array(ep_lens, dtype=np.int64)
    ep_offsets = np.concatenate([[0], np.cumsum(ep_lens[:-1])]).astype(np.int64)
    total_frames = int(ep_lens.sum())
    action_dim = 2

    print(f"Total frames: {total_frames}, resize: {resize}x{resize}")

    # Create HDF5 file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        pixels_ds = f.create_dataset(
            "pixels",
            shape=(total_frames, resize, resize, 3),
            dtype=np.uint8,
            chunks=(64, resize, resize, 3),
            **hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE),
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

        # Write episodes
        idx = 0
        for path in tqdm(episode_paths, desc="Converting"):
            ep = load_episode(path)
            frames = ep["rgb_frames"]  # (T+1, 400, 600, 3)
            actions = ep["actions"]  # (T, 2)
            states = ep["states"]  # (T+1, 15)
            T_plus_1 = len(frames)

            # Pad to square and resize
            frames_resized = pad_and_resize_frames(frames, resize)

            # Pad actions: append NaN so actions has T+1 rows like frames.
            # LeWorldModel's forward does torch.nan_to_num(batch["action"], 0.0)
            # to handle the final frame that has no next action.
            actions_padded = np.full(
                (T_plus_1, action_dim), np.nan, dtype=np.float32
            )
            actions_padded[: len(actions)] = actions

            pixels_ds[idx : idx + T_plus_1] = frames_resized
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
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0, help="Max episodes (0=all)")
    args = parser.parse_args()
    convert(args.input_dirs, args.output, args.resize, args.limit)


if __name__ == "__main__":
    main()
