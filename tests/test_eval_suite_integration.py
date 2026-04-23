"""End-to-end integration test for eval_suite orchestrator.

Populates a fake run_dir with pre-computed JSONs for every first-wave test,
invokes the orchestrator via subprocess, and checks that the aggregated
report pulls them in correctly. No real model / dataset / GPU needed.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _seed_fake_run_dir(run_dir: Path, epoch: int):
    """Write a config.yaml + a fake checkpoint file + JSONs for every test."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({
        "wm": {"history_size": 3, "num_preds": 1,
               "aux_loss": {"enabled": True, "kin_block": 16}},
        "data": {"dataset": {"name": ["dsA", "dsB"], "frameskip": 10}},
    }))
    (run_dir / f"model_epoch_{epoch}_object.ckpt").write_bytes(b"")

    ep = run_dir / f"epoch_{epoch}"

    # Cluster A — both probes
    _write(ep / "encoder_z" / "r2_report.json", {
        "schema_version": "e5-eval-suite-v1",
        "metadata": {}, "results": {
            "encoder_z_probe": {"mean_r2": 0.85,
                                 "per_dim_r2": {"x": 0.9, "y": 0.8, "vx": 0.7,
                                                "vy": 0.95, "angle": 0.9, "ang_vel": 0.4}}}})
    _write(ep / "predicted_z" / "r2_report_normZ.json", {
        "schema_version": "e5-eval-suite-v1",
        "metadata": {}, "results": {
            "predicted_z_probe": {"mean_r2": 0.88,
                                   "per_dim_r2": {"x": 0.9, "y": 0.8, "vx": 0.75,
                                                  "vy": 0.97, "angle": 0.92, "ang_vel": 0.55}}}})

    # Clusters B + C + D (shared action-response JSON)
    _write(ep / "action_response" / "action_response_report_normZ.json", {
        "schema_version": "e5-eval-suite-v1",
        "metadata": {}, "results": {
            "test_1_raw_zdiff": {"main_thrust": {"rel_norm": 0.42, "top_dims": [0]}},
            "test_7_fresh_state_head_action_response": {"main_thrust_dvy": 0.1,
                                                        "side_right_dang_vel": -0.05,
                                                        "side_left_dang_vel": 0.05},
            "test_9_predictor_modification": {"rel_norm": 0.3, "top_dims": [0]},
            "test_10_offline_predloss": {"mse": 0.01, "baseline_mse": 0.05,
                                         "beats_baseline": True},
            "test_11_encoder_forward_coherence": {"rel_norm": 0.2, "top_dims": [0]},
            "test_2_action_magnitude_sweep": {"main": [], "side": [],
                                              "linearity": {"main_r2": 0.95,
                                                            "side_r2": 0.9}},
            "test_4_ood_actions": {"ood_actions": []},
            "test_8_in_model_aux_head_action_response": {"main_thrust_dvy": 0.09,
                                                         "side_right_dang_vel": -0.04},
        }})

    # Cluster D — rollout
    _write(ep / "rollout_fidelity" / "rollout_fidelity.json", {
        "schema_version": "e5-eval-suite-v1",
        "metadata": {}, "results": {
            "test_3_rollout_fidelity": {"per_dataset": {
                "dsA": {"mse_at_5": 0.1, "mse_at_10": 0.2, "mse_at_20": 0.4}}}}})

    # Cluster E
    _write(ep / "cross_head_ref" / "cross_head.json", {
        "schema_version": "e5-eval-suite-v1",
        "metadata": {}, "results": {
            "test_5_cross_network_state_head": {"source_state_head": "/x.pt",
                                                 "per_dim_r2": {"x": 0.7},
                                                 "mean_r2": 0.7}}})


def test_orchestrator_aggregates_all_clusters_from_cached_jsons(tmp_path):
    run_dir = tmp_path / "fake_run"
    _seed_fake_run_dir(run_dir, epoch=1)

    cmd = [sys.executable, "lewm/scripts/eval_suite.py",
           "--run-dir", str(run_dir), "--epoch", "1",
           "--cache-dir", str(tmp_path / "unused-cache"),
           "--rollout-write-videos", "0"]  # test isn't about video dispatch
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    report_txt = (run_dir / "epoch_1" / "eval_suite_report.txt").read_text()
    report_json = json.loads(
        (run_dir / "epoch_1" / "eval_suite_report.json").read_text())

    for cluster_id in ["A", "B", "C", "D", "E"]:
        assert f"Cluster {cluster_id}:" in report_txt

    expected_tests = {
        "encoder_z_probe", "predicted_z_probe",
        "test_1_raw_zdiff", "test_7_fresh_state_head_action_response",
        "test_8_in_model_aux_head_action_response",
        "test_9_predictor_modification", "test_10_offline_predloss",
        "test_11_encoder_forward_coherence",
        "test_2_action_magnitude_sweep", "test_3_rollout_fidelity",
        "test_4_ood_actions",
        "test_5_cross_network_state_head",
    }
    assert expected_tests.issubset(set(report_json["results"].keys()))


def test_report_only_mode_skips_dispatch(tmp_path):
    """--report-only re-renders from disk without running any sub-script.

    The fake checkpoint is 0 bytes, so any attempt to actually load it would
    crash. Successful completion proves nothing was dispatched.
    """
    run_dir = tmp_path / "fake_run"
    _seed_fake_run_dir(run_dir, epoch=1)

    cmd = [sys.executable, "lewm/scripts/eval_suite.py",
           "--run-dir", str(run_dir), "--epoch", "1",
           "--cache-dir", str(tmp_path / "unused-cache"),
           "--report-only"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    report_json = json.loads(
        (run_dir / "epoch_1" / "eval_suite_report.json").read_text())
    assert "encoder_z_probe" in report_json["results"]
    assert "predicted_z_probe" in report_json["results"]
    assert "test_9_predictor_modification" in report_json["results"]
