#!/usr/bin/env python3
"""Multi-scenario rollout fidelity evaluator (Test 3 of the e5 JEPA eval suite).

Runs autoregressive rollouts on one-or-more probe datasets and reports
per-step state-MSE vs GT at canonical step cutoffs. Wraps rollout_episodes
from lewm/eval/rollout.py; no new rollout logic. Network-agnostic:
datasets and reference dataset supplied via CLI.

Usage:
    python lewm/scripts/eval_rollout_fidelity.py \\
        --model /.../lewm_..._object.ckpt \\
        --state-head /.../state_head_*.pt \\
        --datasets lunarlander_synthetic_heuristic_clean \\
                   lunarlander_synthetic_impulse-main_clean \\
                   lunarlander_synthetic_impulse-side_clean \\
        --action-norm-ref lunarlander_synthetic_heuristic_clean \\
        --cache-dir /home/scad/vsr-tmp/lewm-datasets \\
        --output-dir /.../rollout_fidelity_epoch{N} \\
        --ctx-len 3 --n-preds 1 \\
        [--n-episodes 20] [--seq-len 20]

Outputs:
    rollout_fidelity_table.txt inside output-dir.
"""
import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from lewm.eval.rollout import rollout_episodes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--state-head", required=True)
    p.add_argument("--datasets", nargs="+", required=True,
                   help="One or more probe dataset names. Each is rolled out separately "
                        "and gets its own row in the output table. Required — no default "
                        "because networks train on different dataset mixes.")
    p.add_argument("--cache-dir", required=True,
                   help="HDF5 cache dir (e.g. /home/scad/vsr-tmp/lewm-datasets).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ctx-len", type=int, required=True,
                   help="Training history_size. Required — must match the checkpoint.")
    p.add_argument("--n-preds", type=int, required=True,
                   help="Training num_preds. Required — must match the checkpoint.")
    p.add_argument("--frameskip", type=int, default=10)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=20,
                   help="Total rollout length (including seed). Cutoffs at 5, 10, 20.")
    p.add_argument("--action-norm-ref", required=True,
                   help="Reference dataset for training's action normalizer. Required — "
                        "networks train on different dataset mixes and a wrong reference "
                        "silently mis-normalizes actions. Supply the first dataset in the "
                        "training config's data.dataset.name list.")
    p.add_argument("--no-normalize-actions", action="store_false", dest="normalize_actions",
                   default=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "rollout_fidelity_table.txt"

    cutoffs = [5, 10, 20]
    cutoffs = [c for c in cutoffs if c <= args.seq_len]

    lines = ["=== ROLLOUT FIDELITY (Test 3) ==="]
    lines.append(f"model: {args.model}")
    lines.append(f"state_head: {args.state_head}")
    lines.append(f"ctx_len={args.ctx_len} n_preds={args.n_preds} "
                 f"frameskip={args.frameskip} n_episodes={args.n_episodes}")
    lines.append(f"datasets: {args.datasets}")
    lines.append("")
    header = f"  {'dataset':<48s}  " + "  ".join(f"MSE@{c:>2d}" for c in cutoffs)
    lines.append(header)

    per_dataset_json = {}

    for scenario in args.datasets:
        results = rollout_episodes(
            model_path=args.model,
            state_head_path=args.state_head,
            dataset_name=scenario,
            cache_dir=args.cache_dir,
            n_episodes=args.n_episodes,
            seq_len=args.seq_len,
            frameskip=args.frameskip,
            start_mode="episode_start",
            device=args.device,
            normalize_actions=args.normalize_actions,
            action_norm_ref=args.action_norm_ref,
            ctx_len=args.ctx_len,
            n_preds=args.n_preds,
        )

        per_step_mse = []
        for t in range(args.seq_len):
            mses = []
            for r in results:
                pred = r["predicted_states"]
                actual = r["actual_states"]
                if t < len(pred) and t < len(actual):
                    mses.append(float(np.mean((pred[t] - actual[t, :6]) ** 2)))
            per_step_mse.append(float(np.mean(mses)) if mses else float("nan"))

        cutoff_vals = [per_step_mse[c - 1] for c in cutoffs]
        row = f"  {scenario:<48s}  " + "  ".join(f"{v:>6.4f}" for v in cutoff_vals)
        lines.append(row)

        per_dataset_json[scenario] = {
            f"mse_at_{c}": float(per_step_mse[c - 1]) for c in cutoffs
        }

    report = "\n".join(lines) + "\n"
    print(report)
    report_path.write_text(report)
    print(f"Wrote {report_path}")

    from lewm.eval.report_json import write_json_report, metadata_from_args
    write_json_report(
        out_dir=out_dir,
        basename="rollout_fidelity",
        results_dict={"test_3_rollout_fidelity": {"per_dataset": per_dataset_json}},
        metadata=metadata_from_args(args, extra={
            "n_episodes": args.n_episodes,
            "seq_len": args.seq_len,
        }),
    )


if __name__ == "__main__":
    main()
