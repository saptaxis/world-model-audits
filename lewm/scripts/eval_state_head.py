#!/usr/bin/env python3
"""Evaluate an existing state head on a new dataset (no retraining).

Encodes frames with the model's encoder, decodes with the trained state head,
reports R² against GT states.

Usage:
    python lewm/scripts/eval_state_head.py \
        --model /path/to/lewm_object.ckpt \
        --state-head /path/to/state_head.pt \
        --dataset lunarlander_synthetic_free-fall \
        --max-frames 50000
"""

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.encode_dataset import encode_dataset
from eval.state_head import load_state_head, KIN_DIM_NAMES

from sklearn.metrics import r2_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="LeWorldModel _object.ckpt")
    parser.add_argument("--state-head", required=True, help="Trained state_head.pt")
    parser.add_argument("--dataset", required=True, help="HDF5 dataset name to evaluate on")
    parser.add_argument("--cache-dir", default="/home/vsr/vsr-tmp/lewm-datasets")
    parser.add_argument("--max-frames", type=int, default=50000)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    # Load state head
    device = args.device
    state_head, train_metrics = load_state_head(args.state_head, device=device)
    print(f"Loaded state head (trained R²={train_metrics['r2_mean']:.4f})")

    # Encode frames
    output_path = f"/home/vsr/vsr-tmp/lewm-eval-cache/{args.dataset}_z.npz"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(output_path).exists():
        print(f"Loading cached z's from {output_path}")
        data = np.load(output_path)
    else:
        encode_dataset(
            model_path=args.model,
            dataset_name=args.dataset,
            cache_dir=args.cache_dir,
            output_path=output_path,
            device=device,
            batch_size=args.encode_batch_size,
            max_frames=args.max_frames,
        )
        data = np.load(output_path)

    z_all = data["z"]
    state_all = data["state"][:, :6]

    # Decode with state head
    z_tensor = torch.from_numpy(z_all).float().to(device)
    with torch.no_grad():
        pred = state_head(z_tensor).cpu().numpy()

    # R² per dim
    print(f"\n=== State Head Eval on {args.dataset} ({len(z_all)} frames) ===")
    for i, name in enumerate(KIN_DIM_NAMES):
        r2 = r2_score(state_all[:, i], pred[:, i])
        print(f"  {name:8s}: R² = {r2:.4f}")

    r2_mean = np.mean([r2_score(state_all[:, i], pred[:, i]) for i in range(6)])
    print(f"  {'mean':8s}: R² = {r2_mean:.4f}")


if __name__ == "__main__":
    main()
