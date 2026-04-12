#!/usr/bin/env python3
"""Render schematic rollout videos from LeWorldModel.

Rolls out in z-space, decodes via state head, renders predicted vs GT
trajectories using copied rollout_viz renderer.

Usage:
    python lewm/scripts/viz_rollout.py \
        --model /path/to/lewm_epoch_14_object.ckpt \
        --state-head /path/to/state_head/state_head.pt \
        --output-dir /path/to/run-dir/rollout_viz/ \
        --n-episodes 5
"""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.rollout import rollout_episodes
from eval.rollout_viz import render_trajectory_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="LeWorldModel _object.ckpt")
    parser.add_argument("--state-head", required=True, help="state_head.pt path")
    parser.add_argument("--dataset", default="lunarlander_heuristic")
    parser.add_argument("--cache-dir", default="/media/hdd1/physics-priors-latent-space/lunar-lander-data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--frameskip", type=int, default=10)
    parser.add_argument("--start-mode", default="random",
                        choices=["random", "episode_start", "episode_mid"],
                        help="Where to start clips within episodes")
    parser.add_argument("--rgb-dataset", default=None,
                        help="HDF5 dataset name for RGB frames (optional)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rolling out {args.n_episodes} episodes (start_mode={args.start_mode})...")
    results = rollout_episodes(
        model_path=args.model,
        state_head_path=args.state_head,
        dataset_name=args.dataset,
        cache_dir=args.cache_dir,
        n_episodes=args.n_episodes,
        seq_len=args.seq_len,
        frameskip=args.frameskip,
        start_mode=args.start_mode,
        rgb_dataset_name=args.rgb_dataset,
        device=args.device,
    )

    print(f"Rendering videos...")
    for i, rollout in enumerate(results):
        out_path = output_dir / f"rollout_{i:02d}.mp4"
        render_trajectory_video(
            rollout=rollout,
            output_path=str(out_path),
            fps=10,
            title=f"LeWM rollout {i}",
            actions=rollout["actions"],
        )
        print(f"  Saved {out_path}")

    print("\n=== Rollout Summary ===")
    for i, r in enumerate(results):
        z_mse = np.mean((r["z_pred"][-1] - r["z_gt"][-1]) ** 2)
        state_mse = np.mean((r["predicted_states"][-1] - r["actual_states"][-1]) ** 2)
        print(f"  ep {i}: z-MSE={z_mse:.4f}, state-MSE={state_mse:.4f}")


if __name__ == "__main__":
    main()
