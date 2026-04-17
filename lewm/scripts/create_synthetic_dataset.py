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
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.synthetic_render import render_episode_synthetic


def _truncate_idx(states: np.ndarray) -> int:
    """Return the smallest index k where y>1.5 or |x|>1.0. If no such index, return len(states).

    states: (T+1, >=6) with columns [x, y, vx, vy, angle, ang_vel, ...].
    The returned index is a slice end — episode is kept as states[:k].
    """
    x = states[:, 0]
    y = states[:, 1]
    bad = (y > 1.5) | (np.abs(x) > 1.0)
    if not bad.any():
        return len(states)
    return int(np.argmax(bad))


def convert(
    input_dirs: list[str],
    output_path: str,
    triangle_radius: int = 35,
    size: int = 224,
    limit: int = 0,
    truncate: bool = False,
    min_length: int = 0,
    stats_out: str | None = None,
    stats_label: str | None = None,
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

    # First pass: compute sizes (and truncation indices if requested).
    # kept_ep_paths parallel to ep_lens — episodes dropped by min_length
    # are removed here so second pass is simpler.
    ep_lens: list[int] = []
    kept_ep_paths: list[str] = []
    raw_lens: list[int] = []           # original length per input episode (for stats)
    dropped_too_short = 0
    for path in tqdm(episode_paths, desc="Scanning"):
        ep = np.load(path, mmap_mode="r")
        raw_len = int(ep["states"].shape[0])  # T+1 frames
        raw_lens.append(raw_len)
        if truncate:
            k = _truncate_idx(np.asarray(ep["states"][:, :6]))
            kept = k
        else:
            kept = raw_len
        if kept < min_length:
            dropped_too_short += 1
            continue
        ep_lens.append(kept)
        kept_ep_paths.append(path)

    if not kept_ep_paths:
        raise ValueError(
            f"All {len(episode_paths)} episodes dropped by min_length={min_length}"
        )

    print(
        f"After truncation/min-length filter: {len(kept_ep_paths)}/{len(episode_paths)} episodes kept"
        f" ({dropped_too_short} dropped for length < {min_length})"
    )

    ep_lens_arr = np.array(ep_lens, dtype=np.int64)
    ep_offsets = np.concatenate([[0], np.cumsum(ep_lens_arr[:-1])]).astype(np.int64)
    total_frames = int(ep_lens_arr.sum())
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
        for path, kept in tqdm(
            list(zip(kept_ep_paths, ep_lens)),
            desc="Rendering",
            total=len(kept_ep_paths),
        ):
            ep = np.load(path)
            states = ep["states"][:kept]      # (kept, 15) — truncated if kept<full
            actions_full = ep["actions"]      # (T, 2)
            # actions has length T (states has T+1). Trim to kept-1 action entries
            # so the final (kept-th) frame still gets a NaN-padded action like before.
            actions = actions_full[: max(kept - 1, 0)]

            # Render synthetic frames from (possibly truncated) states
            frames = render_episode_synthetic(
                states, size=size, triangle_radius=triangle_radius
            )

            # Pad actions with NaN for final frame
            actions_padded = np.full((kept, action_dim), np.nan, dtype=np.float32)
            actions_padded[: len(actions)] = actions

            pixels_ds[idx : idx + kept] = frames
            action_ds[idx : idx + kept] = actions_padded
            state_ds[idx : idx + kept] = states
            idx += kept

        f.create_dataset("ep_len", data=ep_lens_arr)
        f.create_dataset("ep_offset", data=ep_offsets)

    print(f"Written to {output_path}")
    print(f"  {len(ep_lens_arr)} episodes, {total_frames} total frames")

    if stats_out is not None:
        raw_lens_arr = np.array(raw_lens, dtype=np.int64)
        entry = {
            "label": stats_label or Path(output_path).stem,
            "output_path": str(output_path),
            "truncate_enabled": bool(truncate),
            "min_length": int(min_length),
            "episodes_in": int(len(episode_paths)),
            "episodes_kept": int(len(kept_ep_paths)),
            "episodes_dropped_too_short": int(dropped_too_short),
            "frames_in": int(raw_lens_arr.sum()),
            "frames_kept": int(total_frames),
            "kept_length_mean": float(ep_lens_arr.mean()) if ep_lens_arr.size else 0.0,
            "kept_length_median": float(np.median(ep_lens_arr)) if ep_lens_arr.size else 0.0,
            "kept_length_min": int(ep_lens_arr.min()) if ep_lens_arr.size else 0,
            "kept_length_max": int(ep_lens_arr.max()) if ep_lens_arr.size else 0,
            "raw_length_mean": float(raw_lens_arr.mean()),
            "raw_length_min": int(raw_lens_arr.min()),
            "raw_length_max": int(raw_lens_arr.max()),
        }
        stats_path = Path(stats_out)
        # Merge with existing file so parallel runs can share one stats doc.
        if stats_path.exists():
            try:
                with stats_path.open("r") as fh:
                    existing = json.load(fh)
            except json.JSONDecodeError:
                existing = {}
        else:
            existing = {}
        existing[entry["label"]] = entry
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically-ish: file-level lock would be better for true parallel
        # but at 12 small writes this is acceptable.
        tmp = stats_path.with_suffix(stats_path.suffix + ".tmp")
        with tmp.open("w") as fh:
            json.dump(existing, fh, indent=2, sort_keys=True)
        tmp.replace(stats_path)
        print(f"Stats appended to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--triangle-radius", type=int, default=35)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0, help="Max episodes (0=all)")
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate each episode at the first frame where y>1.5 or |x|>1.0 (off-screen).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=0,
        help="Drop episodes whose kept length is below this threshold (post-truncation).",
    )
    parser.add_argument(
        "--stats-out",
        type=str,
        default=None,
        help="If set, append a per-dataset stats entry to this JSON file.",
    )
    parser.add_argument(
        "--stats-label",
        type=str,
        default=None,
        help="Label to use as the stats JSON key. Defaults to the output file stem.",
    )
    args = parser.parse_args()
    convert(
        args.input_dirs,
        args.output,
        args.triangle_radius,
        args.size,
        args.limit,
        truncate=args.truncate,
        min_length=args.min_length,
        stats_out=args.stats_out,
        stats_label=args.stats_label,
    )


if __name__ == "__main__":
    main()
