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
        [--report-out /path/to/trained_aux_head_eval.txt] \\
        [--state-head /other_ckpt_dir/state_head.pt]

Outputs: R² per dim + mean, MSE on normalized targets (directly comparable to
training's aux_kin_loss). Console-only by default; pass --report-out to also
write a text report. A JSON sibling is always written alongside the text report
(or beside the cache if --report-out is omitted).
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
    p.add_argument("--state-head", default=None,
                   help="Optional: load a state_head.pt from a DIFFERENT checkpoint "
                        "and apply it to --cache (cross-network portability — Test 5 "
                        "of the e5-jepa-eval-suite). Overrides the in-model aux head. "
                        "State-head's z_slice metadata is honored (auto-slicing).")
    args = p.parse_args()

    print(f"Loading model {args.model}")
    model = torch.load(args.model, map_location=args.device, weights_only=False)

    if args.state_head:
        from lewm.eval.state_head import load_state_head
        ext_head, ext_metrics = load_state_head(args.state_head, device=args.device)
        ext_head.eval()
        z_slice = ext_metrics.get("z_slice")
        def _decode(z_batch):
            z_t = torch.from_numpy(z_batch).float().to(args.device)
            if z_slice is not None:
                z_t = z_t[..., z_slice[0]:z_slice[1]]
            with torch.no_grad():
                return ext_head(z_t).cpu().numpy()
        print(f"Using external state_head: {args.state_head}")
        if z_slice is not None:
            print(f"  external head expects sliced z [{z_slice[0]}:{z_slice[1]}]")
        kin_in = None  # not used in external-head path
    else:
        sh = getattr(model, "state_head", None)
        if sh is None:
            raise SystemExit("model has no .state_head and no --state-head supplied.")
        W = sh.linear.weight.detach().cpu().numpy()
        b = sh.linear.bias.detach().cpu().numpy()
        mean = sh.target_mean.detach().cpu().numpy()
        std = sh.target_std.detach().cpu().numpy()
        print(f"state_head.linear: W={W.shape}, b={b.shape}")
        print(f"target_mean={mean}")
        print(f"target_std ={std}")
        kin_in = W.shape[1]
        def _decode(z_batch):
            z_slice_arr = z_batch[:, :kin_in]
            decoded_norm = z_slice_arr @ W.T + b
            return decoded_norm * std + mean

    data = np.load(args.cache)
    z = data["z"]
    s = data["state"][:, :6]
    print(f"cache: z={z.shape}, state={s.shape}"
          + (f", using z[:, :{kin_in}]" if kin_in is not None else ""))

    # Replicate probe's train/val split (seed=42 hard-coded in train_state_head)
    n = len(z)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    split = int((1 - args.val_split) * n)
    val_idx = perm[split:]
    z_val = z[val_idx]
    s_val = s[val_idx]
    print(f"val: {len(val_idx)} frames")

    pred_state = _decode(z_val)

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

    if not args.state_head:
        gt_norm = (s_val - mean) / std
        decoded_norm = z_val[:, :kin_in] @ W.T + b
        mse_norm = float(np.mean((decoded_norm - gt_norm) ** 2))
        lines.append("")
        lines.append(f"  MSE on normalized targets = {mse_norm:.6f}")
        lines.append("  (compare to training log validate/aux_kin_loss — should match if")
        lines.append("   probe cache's (z, state) pairing matches training's.)")
    else:
        lines.append("")
        lines.append("  (External state_head mode — MSE on normalized targets skipped.)")

    print("\n" + "\n".join(lines))

    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print(f"\nReport written to {out}")

    from lewm.eval.report_json import write_json_report, metadata_from_args
    per_dim_r2 = {name: float(r2) for name, r2 in zip(KIN, r2s)}
    mean_r2 = float(np.mean(r2s))
    test_key = "test_5_cross_network_state_head" if args.state_head else "aux_head_r2_in_model"

    if args.report_out:
        json_out_dir = Path(args.report_out).parent
        json_basename = Path(args.report_out).stem
    else:
        json_out_dir = Path(args.cache).parent
        json_basename = "trained_aux_head_eval"

    write_json_report(
        out_dir=json_out_dir,
        basename=json_basename,
        results_dict={
            test_key: {
                "source_state_head": args.state_head,
                "per_dim_r2": per_dim_r2,
                "mean_r2": mean_r2,
            }
        },
        metadata=metadata_from_args(args, extra={
            "cache": args.cache,
            "state_head_path": args.state_head,
            "val_split": args.val_split,
            "seed": args.seed,
        }),
    )


if __name__ == "__main__":
    main()
