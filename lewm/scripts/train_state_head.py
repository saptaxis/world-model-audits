#!/usr/bin/env python3
"""Train state head probe on LeWorldModel latents.

Usage:
    python lewm/scripts/train_state_head.py \
        --model /path/to/lewm_epoch_14_object.ckpt \
        --dataset lunarlander_heuristic \
        --output-dir /path/to/run-dir/state_head/

Encodes frames (or loads cached z's), trains MLP probe, reports R² per dim.
"""

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.encode_dataset import encode_dataset
from eval.state_head import train_state_head, save_state_head, KIN_DIM_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="LeWorldModel _object.ckpt path")
    parser.add_argument("--dataset", default="lunarlander_heuristic", help="HDF5 dataset name")
    parser.add_argument("--cache-dir", default="/media/hdd1/physics-priors-latent-space/lunar-lander-data")
    parser.add_argument("--output-dir", required=True, help="Where to save state head + results")
    parser.add_argument("--max-frames", type=int, default=50000, help="Max frames to encode (0=all)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    z_cache = output_dir / "encoded_z.npz"

    # Step 1: Encode (or load cache)
    if z_cache.exists():
        print(f"Loading cached embeddings from {z_cache}")
        data = np.load(z_cache)
        z_all, state_all = data["z"], data["state"]
    else:
        encode_dataset(
            model_path=args.model,
            dataset_name=args.dataset,
            cache_dir=args.cache_dir,
            output_path=str(z_cache),
            device=args.device,
            max_frames=args.max_frames,
        )
        data = np.load(z_cache)
        z_all, state_all = data["z"], data["state"]

    # Use first 6 dims only (kinematics)
    state_kin = state_all[:, :6]

    # Train/val split (90/10)
    n = len(z_all)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    split = int(0.9 * n)
    z_train, z_val = z_all[perm[:split]], z_all[perm[split:]]
    s_train, s_val = state_kin[perm[:split]], state_kin[perm[split:]]
    print(f"Train: {len(z_train)}, Val: {len(z_val)}")

    # Step 2: Train
    print("Training state head...")
    head, metrics = train_state_head(
        z_train, s_train, z_val, s_val,
        epochs=args.epochs,
        device=args.device,
    )

    # Step 3: Report
    print("\n=== State Head R² ===")
    for name in KIN_DIM_NAMES:
        r2 = metrics["r2_per_dim"][name]
        print(f"  {name:8s}: R² = {r2:.4f}")
    print(f"  {'mean':8s}: R² = {metrics['r2_mean']:.4f}")
    print(f"  val MSE: {metrics['val_mse']:.6f}")

    # Step 4: Save
    save_state_head(head, metrics, str(output_dir / "state_head.pt"))

    # Save report as text
    with open(output_dir / "r2_report.txt", "w") as f:
        f.write("State Head R² Report\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Frames: {n}\n\n")
        for name in KIN_DIM_NAMES:
            f.write(f"  {name:8s}: R² = {metrics['r2_per_dim'][name]:.4f}\n")
        f.write(f"\n  mean R²: {metrics['r2_mean']:.4f}\n")
        f.write(f"  val MSE: {metrics['val_mse']:.6f}\n")
    print(f"Report saved to {output_dir / 'r2_report.txt'}")


if __name__ == "__main__":
    main()
