#!/usr/bin/env python3
"""E5 JEPA Eval Suite orchestrator.

Given a run_dir + epoch, runs (or reads) the selected first-wave tests
and emits a unified report.

Usage:
    python lewm/scripts/eval_suite.py \\
        --run-dir /media/hdd1/.../synthetic-all-clean-fs10-aux \\
        --epoch 1 \\
        --cache-dir /home/scad/vsr-tmp/lewm-datasets \\
        [--include-clusters A B C D E]
        [--tests 1 3 5 9 10 11]
        [--skip-tests 3]
        [--probe-dataset <name>]
        [--reference-state-head /.../state_head.pt]
        [--reference-tag <short_name>]

Behavior:
    - Reads <run-dir>/config.yaml for ctx_len, n_preds, action_norm_ref, datasets.
    - For each selected test, checks <run-dir>/epoch_<N>/<subdir>/...json.
        If present, loads it. If absent, runs the sub-script.
    - Writes <run-dir>/epoch_<N>/eval_suite_report.{txt,json} with
        cluster-organized results.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from lewm.eval.suite_runner import (
    infer_config_from_run_dir, resolve_requested_tests,
    EvalTarget, CLUSTERS, TEST_ARTIFACTS,
    _run_encoder_z_probe, _run_action_response_group,
    _run_rollout_fidelity, _run_cross_head, _run_predicted_z_probe,
)


CLUSTER_DESCS = {
    "A": "Encoder state-structure (Q1)",
    "B": "Predictor basic activity (Q2)",
    "C": "Predictor action-pathway (Q3)",
    "D": "Predictor physics fidelity (Q4)",
    "E": "Decoder portability (Case 2 attack)",
}


def format_report(target, results: dict) -> str:
    """Human-readable report, cluster-organized."""
    lines = [
        "=" * 70,
        "E5 JEPA Eval Suite — First Wave Report",
        "=" * 70,
        f"Checkpoint: {target.ckpt_path()}",
        f"Epoch: {target.epoch}",
        f"Config: ctx_len={target.cfg['ctx_len']}, n_preds={target.cfg['n_preds']}, "
        f"kin_block={target.cfg['kin_block']}, action_norm_ref={target.cfg['action_norm_ref']}",
        "",
    ]
    for cluster_id, test_names in CLUSTERS.items():
        lines.append(f"--- Cluster {cluster_id}: {CLUSTER_DESCS[cluster_id]} ---")
        for test_name in test_names:
            if test_name not in results:
                lines.append(f"  {test_name}: not run")
                continue
            lines.append(f"  {test_name}:")
            for k, v in results[test_name].items():
                lines.append(f"    {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def _load_result(target, test_name, reference_tag):
    """Load a single test's results from its artifact JSON."""
    rel, _ = TEST_ARTIFACTS[test_name]
    rel = rel.format(source_tag=reference_tag)
    path = target.epoch_dir() / rel
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    data = json.loads(path.read_text())
    # The JSON has {schema_version, metadata, results: {test_key: {...}}}
    # Return the specific test_name's results if present in results dict
    results = data.get("results", {})
    if test_name in results:
        return results[test_name]
    # Fall through: return the full results dict (some tests share a JSON)
    return results.get(test_name, data)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="Training run dir (contains config.yaml + checkpoints).")
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--include-clusters", nargs="+", default=None,
                   choices=["A", "B", "C", "D", "E"])
    p.add_argument("--tests", nargs="+", type=int, default=None,
                   help="Explicit test numbers (e.g., 1 3 5 9). Overrides --include-clusters.")
    p.add_argument("--skip-tests", nargs="+", type=int, default=[])
    p.add_argument("--probe-dataset", default=None,
                   help="Action-response probe dataset. Default: first cfg dataset.")
    p.add_argument("--reference-state-head", default=None,
                   help="Path to state_head.pt from another checkpoint for Test 5.")
    p.add_argument("--reference-tag", default="ref",
                   help="Short name for the reference-head source (used as subdir suffix).")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg = infer_config_from_run_dir(run_dir)
    target = EvalTarget(run_dir=run_dir, epoch=args.epoch, cfg=cfg)
    target.epoch_dir().mkdir(parents=True, exist_ok=True)

    selected = resolve_requested_tests(args.include_clusters, args.tests, args.skip_tests)
    print(f"Selected tests: {selected}")

    probe_dataset = args.probe_dataset or cfg["datasets"][0]
    cache_dir = Path(args.cache_dir)

    # State_head path for action-response + rollout
    probe_subdir = target.epoch_dir() / "predicted_z"
    state_head_path = probe_subdir / "state_head_normZ.pt"

    # Determine what needs to run
    needs_action_response = any(t in selected for t in [
        "test_1_raw_zdiff", "test_2_action_magnitude_sweep", "test_4_ood_actions",
        "test_7_fresh_state_head_action_response",
        "test_8_in_model_aux_head_action_response",
        "test_9_predictor_modification", "test_10_offline_predloss",
        "test_11_encoder_forward_coherence",
    ])
    needs_rollout = "test_3_rollout_fidelity" in selected
    needs_encoder_z = "encoder_z_probe" in selected
    needs_cross_head = "test_5_cross_network_state_head" in selected

    if needs_encoder_z:
        if not (target.epoch_dir() / "encoder_z" / "r2_report.json").exists():
            _run_encoder_z_probe(target, cache_dir)

    ar_json_exists = (target.epoch_dir() / "action_response" / "action_response_report_normZ.json").exists()
    rf_json_exists = (target.epoch_dir() / "rollout_fidelity" / "rollout_fidelity.json").exists()
    ar_needs_run = needs_action_response and not ar_json_exists
    rf_needs_run = needs_rollout and not rf_json_exists

    if ar_needs_run or rf_needs_run:
        if not state_head_path.exists():
            _run_predicted_z_probe(target, cache_dir)

        if ar_needs_run:
            _run_action_response_group(target, cache_dir, state_head_path, probe_dataset,
                                       tests_requested=set(selected))

        if rf_needs_run:
            _run_rollout_fidelity(target, cache_dir, state_head_path)

    if needs_cross_head:
        cross_head_json = target.epoch_dir() / f"cross_head_{args.reference_tag}" / "cross_head.json"
        if not cross_head_json.exists():
            assert args.reference_state_head, \
                "Test 5 (cluster E) requires --reference-state-head"
            predicted_z_cache = next(probe_subdir.glob("predicted_z_aligned_*.npz"))
            _run_cross_head(target, cache_dir, Path(args.reference_state_head),
                            args.reference_tag, predicted_z_cache)

    # Aggregate results
    results = {}
    for test_name in selected:
        try:
            results[test_name] = _load_result(target, test_name, args.reference_tag)
        except FileNotFoundError:
            pass

    # Emit report
    txt = format_report(target, results)
    out_txt = target.epoch_dir() / "eval_suite_report.txt"
    out_json = target.epoch_dir() / "eval_suite_report.json"
    out_txt.write_text(txt)
    combined = {
        "schema_version": "e5-eval-suite-first-wave-v1",
        "model": str(target.ckpt_path()),
        "epoch": target.epoch,
        "cfg": cfg,
        "results": results,
    }
    out_json.write_text(json.dumps(combined, indent=2))
    print(txt)
    print(f"\nWrote {out_txt}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
