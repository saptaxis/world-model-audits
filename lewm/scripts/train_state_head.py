#!/usr/bin/env python3
"""Train state head probe on LeWorldModel latents.

Usage:
    # Single dataset (backward compatible):
    python lewm/scripts/train_state_head.py \
        --model /path/to/lewm_object.ckpt \
        --dataset lunarlander_heuristic \
        --output-dir /path/to/state_head/ \
        --max-frames 50000

    # Multiple datasets, N frames per dataset:
    python lewm/scripts/train_state_head.py \
        --model /path/to/lewm_object.ckpt \
        --dataset lunarlander_synthetic_heuristic lunarlander_synthetic_free-fall ... \
        --output-dir /path/to/state_head/ \
        --max-frames-per-dataset 5000

    # Multiple datasets, N frames total (split evenly):
    python lewm/scripts/train_state_head.py \
        --model /path/to/lewm_object.ckpt \
        --dataset lunarlander_synthetic_heuristic lunarlander_synthetic_free-fall ... \
        --output-dir /path/to/state_head/ \
        --max-frames 50000

    # No limit — all frames from all datasets:
    python lewm/scripts/train_state_head.py \
        --model /path/to/lewm_object.ckpt \
        --dataset lunarlander_synthetic_heuristic lunarlander_synthetic_free-fall ... \
        --output-dir /path/to/state_head/

Encodes frames (or loads cached z's), trains MLP probe, reports R² per dim.
"""

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.encode_dataset import encode_dataset
from eval.state_head import train_state_head, save_state_head, KIN_DIM_NAMES


def resolve_max_frames(datasets, max_frames, max_frames_per_dataset, cache_dir):
    """Return per-dataset max_frames list.

    Scenarios:
        - Both flags set: error
        - --max-frames-per-dataset N: N per dataset
        - --max-frames N (single dataset): N for that dataset (backward compat)
        - --max-frames N (multi-dataset): N total, distributed proportionally
          by dataset size (number of frames in each HDF5)
        - Neither flag: 0 (all frames) per dataset
    """
    n = len(datasets)
    if max_frames is not None and max_frames_per_dataset is not None:
        raise ValueError("Cannot specify both --max-frames and --max-frames-per-dataset")
    if max_frames_per_dataset is not None:
        return [max_frames_per_dataset] * n
    if max_frames is not None:
        if n == 1:
            return [max_frames]
        # Distribute proportionally by dataset size
        import h5py
        import hdf5plugin  # noqa: F401
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "le-wm"))
        import stable_worldmodel as swm
        datasets_dir = swm.data.utils.get_cache_dir(cache_dir, sub_folder="datasets")
        sizes = []
        for ds_name in datasets:
            h5_path = Path(datasets_dir) / f"{ds_name}.h5"
            with h5py.File(h5_path, "r") as f:
                sizes.append(int(f["ep_len"][:].sum()))
        total = sum(sizes)
        per_ds = [min(s, int(max_frames * s / total)) for s in sizes]
        print(f"Distributing {max_frames} frames proportionally across {n} datasets:")
        for ds_name, s, alloc in zip(datasets, sizes, per_ds):
            capped = " (all)" if alloc >= s else ""
            print(f"  {ds_name}: {s} total -> {alloc} sampled{capped}")
        return per_ds
    return [0] * n  # 0 = all frames


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="LeWorldModel _object.ckpt path")
    parser.add_argument("--dataset", nargs="+", default=["lunarlander_heuristic"],
                        help="One or more HDF5 dataset names")
    parser.add_argument("--cache-dir", default="/media/hdd1/physics-priors-latent-space/lunar-lander-data")
    parser.add_argument("--output-dir", required=True, help="Where to save state head + results")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Total max frames across all datasets (split evenly for multi-dataset). 0=all.")
    parser.add_argument("--max-frames-per-dataset", type=int, default=None,
                        help="Max frames per dataset (multi-dataset only). 0=all.")
    parser.add_argument("--encode-batch-size", type=int, default=1024, help="Batch size for ViT encoding")
    parser.add_argument("--read-chunk-size", type=int, default=10000,
                        help="Frames to read from HDF5 at a time (controls memory usage)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction held out for validation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--linear", action="store_true",
                        help="Use linear probe instead of MLP")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    datasets = args.dataset
    per_ds_limits = resolve_max_frames(datasets, args.max_frames, args.max_frames_per_dataset,
                                       args.cache_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    z_cache = output_dir / "encoded_z.npz"

    # Step 1: Encode (or load cache)
    if z_cache.exists():
        print(f"Loading cached embeddings from {z_cache}")
        data = np.load(z_cache)
        z_all, state_all = data["z"], data["state"]
    else:
        z_parts, state_parts = [], []
        for ds_name, ds_limit in zip(datasets, per_ds_limits):
            ds_cache = output_dir / f"encoded_z_{ds_name}.npz"
            print(f"\n--- Encoding {ds_name} (max_frames={ds_limit or 'all'}) ---")
            encode_dataset(
                model_path=args.model,
                dataset_name=ds_name,
                cache_dir=args.cache_dir,
                output_path=str(ds_cache),
                device=args.device,
                batch_size=args.encode_batch_size,
                max_frames=ds_limit,
                read_chunk_size=args.read_chunk_size,
            )
            data = np.load(ds_cache)
            z_parts.append(data["z"])
            state_parts.append(data["state"])
            print(f"  {ds_name}: {data['z'].shape[0]} frames encoded")

        z_all = np.concatenate(z_parts, axis=0)
        state_all = np.concatenate(state_parts, axis=0)
        np.savez(z_cache, z=z_all, state=state_all)
        print(f"\nCombined: {z_all.shape[0]} frames from {len(datasets)} datasets -> {z_cache}")

    # Use first 6 dims only (kinematics)
    state_kin = state_all[:, :6]

    n = len(z_all)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    split = int((1 - args.val_split) * n)
    z_train, z_val = z_all[perm[:split]], z_all[perm[split:]]
    s_train, s_val = state_kin[perm[:split]], state_kin[perm[split:]]
    print(f"Train: {len(z_train)}, Val: {len(z_val)}")

    # Step 2: Train
    probe_type = "linear" if args.linear else "MLP"
    print(f"Training state head ({probe_type})...")
    head, metrics = train_state_head(
        z_train, s_train, z_val, s_val,
        epochs=args.epochs,
        device=args.device,
        linear=args.linear,
    )

    # Step 3: Report
    print(f"\n=== State Head R² ({probe_type}) ===")
    for name in KIN_DIM_NAMES:
        r2 = metrics["r2_per_dim"][name]
        print(f"  {name:8s}: R² = {r2:.4f}")
    print(f"  {'mean':8s}: R² = {metrics['r2_mean']:.4f}")
    print(f"  val MSE: {metrics['val_mse']:.6f}")

    # Step 4: Save
    suffix = "_linear" if args.linear else ""
    save_state_head(head, metrics, str(output_dir / f"state_head{suffix}.pt"))

    # Save report as text
    ds_str = ", ".join(datasets)
    report_path = output_dir / f"r2_report{suffix}.txt"
    with open(report_path, "w") as f:
        f.write(f"State Head R² Report ({probe_type})\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Datasets ({len(datasets)}): {ds_str}\n")
        f.write(f"Frames: {n}\n\n")
        for name in KIN_DIM_NAMES:
            f.write(f"  {name:8s}: R² = {metrics['r2_per_dim'][name]:.4f}\n")
        f.write(f"\n  mean R²: {metrics['r2_mean']:.4f}\n")
        f.write(f"  val MSE: {metrics['val_mse']:.6f}\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
