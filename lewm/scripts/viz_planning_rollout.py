#!/usr/bin/env python3
"""Render annotated planner rollouts from an evaluate_replay logs npz.

Usage:
    python lewm/scripts/viz_planning_rollout.py \\
        --logs /path/to/replay_cem_logs.npz \\
        --output-dir /path/to/videos/ \\
        [--n-videos 10]
"""
import argparse
from pathlib import Path
import sys

# Repo root for `lewm.*` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from lewm.eval.rollout_viz import render_planner_trajectory_video


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True, help="npz from evaluate_replay")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-videos", type=int, default=20)
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    d = np.load(args.logs)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = min(args.n_videos, d["planner_states"].shape[0])
    # actual_steps: how many output steps ran before early termination
    actual_steps = int(d["actual_steps"]) if "actual_steps" in d else d["planner_states"].shape[1] - 1
    for i in range(n):
        render_planner_trajectory_video(
            planner_states=d["planner_states"][i, :actual_steps + 1],
            dataset_states=d["dataset_states"][i, :actual_steps + 1],
            goal_state=d["dataset_goal_states"][i],
            planner_actions=d["planner_actions"][i, :actual_steps],
            planner_costs=d["planner_costs"][i, :actual_steps],
            output_path=outdir / f"rollout_{i:02d}.mp4",
            title=f"Replay episode {i} (seed={int(d['ep_seeds'][i])})",
            fps=args.fps,
        )
        print(f"Wrote {outdir / f'rollout_{i:02d}.mp4'}")


if __name__ == "__main__":
    main()
