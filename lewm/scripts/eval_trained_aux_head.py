#!/usr/bin/env python3
"""Apply the co-trained aux state_head (LinearStateHead, z_kin -> 6) from a
JEPA checkpoint directly to a probe's cached predicted-z npz — no retraining.

Purpose: localize the gap between training's validate/aux_kin_loss (logged
each epoch) and an offline --linear --z-dims 0:K probe trained from scratch.
Both share the same function class (Linear(K, 6) + per-dim normalize). If MSE
on normalized targets here is close to training's validate/aux_kin_loss, the
probe cache faithfully reproduces training's aux-loss data distribution. If
not, the cache pairing or input distribution mismatches training.

CACHE REGIME REQUIREMENT
    The cache must be encoded under the same regime the model was trained
    in, especially the action-input statistics. Use train_state_head.py
    --training-aligned with --normalize-actions matching the checkpoint's
    training. For broken-regime multi-dataset checkpoints (pre ConcatDataset
    transform-propagation fix; see e5-05) leave --normalize-actions OFF.

Usage:
    python lewm/scripts/eval_trained_aux_head.py \\
        --model /.../lewm_..._object.ckpt \\
        --cache /.../state_head_*/predicted_z_aligned_ctx{C}_np{P}_norm{Raw,Z}.npz \\
        [--val-split 0.25] [--seed 42] \\
        [--report-out /path/to/trained_aux_head_eval.txt]

Outputs: R² per dim + mean, MSE on normalized targets (directly comparable to
training's aux_kin_loss). Console-only by default; pass --report-out to also
write a text report.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

# le-wm vendor path (LinearStateHead, modules resolved on torch.load)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "le-wm"))

KIN = ["x", "y", "vx", "vy", "angle", "ang_vel"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="LeWorldModel _object.ckpt")
    p.add_argument("--cache", required=True,
                   help="predicted_z.npz from a train_state_head probe dir")
    p.add_argument("--val-split", type=float, default=0.25,
                   help="Must match the probe's --val-split to replicate its val set.")
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the probe's shuffle seed (fixed at 42 in train_state_head).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--report-out", default=None,
                   help="Optional: also write a text report to this path. "
                        "Default: console only.")
    args = p.parse_args()

    print(f"Loading model {args.model}")
    model = torch.load(args.model, map_location=args.device, weights_only=False)
    sh = getattr(model, "state_head", None)
    if sh is None:
        raise SystemExit("model has no .state_head — was aux loss enabled during training?")
    W = sh.linear.weight.detach().cpu().numpy()
    b = sh.linear.bias.detach().cpu().numpy()
    mean = sh.target_mean.detach().cpu().numpy()
    std = sh.target_std.detach().cpu().numpy()
    print(f"state_head.linear: W={W.shape}, b={b.shape}")
    print(f"target_mean={mean}")
    print(f"target_std ={std}")

    data = np.load(args.cache)
    z = data["z"]
    s = data["state"][:, :6]
    kin_in = W.shape[1]
    print(f"cache: z={z.shape}, state={s.shape}, using z[:, :{kin_in}]")

    # Replicate probe's train/val split (seed=42 hard-coded in train_state_head)
    n = len(z)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    split = int((1 - args.val_split) * n)
    val_idx = perm[split:]
    z_val = z[val_idx, :kin_in]
    s_val = s[val_idx]
    print(f"val: {len(val_idx)} frames")

    # Apply trained linear head: decoded is in normalized-state space
    decoded_norm = z_val @ W.T + b
    pred_state = decoded_norm * std + mean  # unnormalize for raw-state R²

    r2s = []
    lines = ["=== Trained aux head applied to probe val ==="]
    lines.append(f"model: {args.model}")
    lines.append(f"cache: {args.cache}")
    lines.append(f"val: {len(val_idx)} frames (val_split={args.val_split}, seed={args.seed})")
    lines.append("")
    for i, name in enumerate(KIN):
        r2 = r2_score(s_val[:, i], pred_state[:, i])
        r2s.append(r2)
        lines.append(f"  {name:8s}: R² = {r2:.4f}")
    lines.append(f"  mean    : R² = {np.mean(r2s):.4f}")

    gt_norm = (s_val - mean) / std
    mse_norm = float(np.mean((decoded_norm - gt_norm) ** 2))
    lines.append("")
    lines.append(f"  MSE on normalized targets = {mse_norm:.6f}")
    lines.append("  (compare to training log validate/aux_kin_loss — should match if")
    lines.append("   probe cache's (z, state) pairing matches training's.)")

    print("\n" + "\n".join(lines))

    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
