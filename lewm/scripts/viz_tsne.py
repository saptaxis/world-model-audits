#!/usr/bin/env python3
"""Visualize LeWorldModel latent space with t-SNE, colored by kinematic dims.

Usage:
    # From cached encoded_z.npz (fast — no encoding needed):
    python lewm/scripts/viz_tsne.py \
        --encoded-z /path/to/state_head_epoch10_all/encoded_z.npz \
        --output-dir /path/to/state_head_epoch10_all/tsne/

    # From model + dataset (encodes first):
    python lewm/scripts/viz_tsne.py \
        --model /path/to/lewm_object.ckpt \
        --dataset lunarlander_synthetic_heuristic \
        --cache-dir ~/vsr-tmp/lewm-datasets \
        --output-dir /path/to/tsne/ \
        --max-frames 10000

Produces one scatter plot per kinematic dimension + one colored by dataset source.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


KIN_DIM_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoded-z", help="Path to encoded_z.npz (skips encoding)")
    parser.add_argument("--model", help="LeWorldModel _object.ckpt (if encoding needed)")
    parser.add_argument("--dataset", nargs="+", help="HDF5 dataset name(s)")
    parser.add_argument("--cache-dir", default="/home/vsr/vsr-tmp/lewm-datasets")
    parser.add_argument("--output-dir", required=True, help="Where to save plots")
    parser.add_argument("--max-frames", type=int, default=10000,
                        help="Max frames for t-SNE (subsample if more)")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Re-run t-SNE even if cached coords exist")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.encoded_z:
        print(f"Loading cached embeddings from {args.encoded_z}")
        data = np.load(args.encoded_z)
        z_all, state_all = data["z"], data["state"]
    else:
        raise ValueError("--encoded-z is required (run train_state_head.py first to generate)")

    # Subsample for t-SNE speed
    n = len(z_all)
    rng = np.random.default_rng(args.seed)
    if n > args.max_frames:
        idx = rng.choice(n, size=args.max_frames, replace=False)
        z_sub = z_all[idx]
        state_sub = state_all[idx]
        print(f"Subsampled {args.max_frames} from {n} frames")
    else:
        z_sub = z_all
        state_sub = state_all
        print(f"Using all {n} frames")

    # Run t-SNE (or load cached coords)
    coords_cache = output_dir / "tsne_coords.npz"
    if coords_cache.exists() and not args.force:
        print(f"Loading cached t-SNE coords from {coords_cache}")
        cached = np.load(coords_cache)
        z_2d, state_sub = cached["z_2d"], cached["state"]
    else:
        print(f"Running t-SNE (perplexity={args.perplexity})...")
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed,
                    init="pca", learning_rate="auto")
        z_2d = tsne.fit_transform(z_sub)
        print(f"  KL divergence: {tsne.kl_divergence_:.4f}")

    # Plot per kinematic dimension
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, (ax, name) in enumerate(zip(axes.flat, KIN_DIM_NAMES)):
        vals = state_sub[:, i]
        # Clip to 2nd-98th percentile to prevent outliers from washing out the colormap
        vmin, vmax = np.percentile(vals, [2, 98])
        sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=vals, cmap="coolwarm",
                        vmin=vmin, vmax=vmax,
                        s=3, alpha=0.7, rasterized=True)
        ax.set_title(name, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"t-SNE of LeWM latent space (n={len(z_sub)}, perp={args.perplexity})",
                 fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_kinematics.png", dpi=150, bbox_inches="tight")
    print(f"Saved {output_dir / 'tsne_kinematics.png'}")
    plt.close(fig)

    # Save t-SNE coordinates for reuse
    np.savez(output_dir / "tsne_coords.npz", z_2d=z_2d, state=state_sub)
    print(f"Saved t-SNE coords to {output_dir / 'tsne_coords.npz'}")


if __name__ == "__main__":
    main()
