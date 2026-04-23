"""Helpers for eval_suite.py orchestrator — config inference, artifact
discovery, subprocess orchestration."""
from __future__ import annotations
from pathlib import Path
import json
import subprocess
import yaml
from dataclasses import dataclass


def infer_config_from_run_dir(run_dir) -> dict:
    """Read <run_dir>/config.yaml and extract the fields the eval scripts need.

    Hydra emits config.yaml when training starts, containing the resolved
    full training config. We extract: ctx_len (wm.history_size), n_preds
    (wm.num_preds), kin_block (wm.aux_loss.kin_block if aux enabled else
    None), action_norm_ref (first dataset name), datasets (full list),
    frameskip.
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found. Run this orchestrator against a run_dir "
            "that contains hydra's training-time config.yaml (auto-written "
            "by le-wm training on start)."
        )
    raw = yaml.safe_load(cfg_path.read_text())
    wm = raw.get("wm", {})
    ds_name = raw.get("data", {}).get("dataset", {}).get("name")
    datasets = [ds_name] if isinstance(ds_name, str) else list(ds_name)

    aux = wm.get("aux_loss") or {}
    kin_block = aux.get("kin_block") if aux.get("enabled") else None

    return {
        "ctx_len": int(wm["history_size"]),
        "n_preds": int(wm["num_preds"]),
        "kin_block": kin_block,
        "action_norm_ref": datasets[0],
        "datasets": datasets,
        "frameskip": int(raw["data"]["dataset"].get("frameskip", 10)),
    }


CLUSTERS = {
    "A": ["encoder_z_probe"],
    "B": ["test_9_predictor_modification", "test_10_offline_predloss", "test_11_encoder_forward_coherence"],
    "C": ["test_1_raw_zdiff", "test_7_fresh_state_head_action_response",
          "test_8_in_model_aux_head_action_response"],
    "D": ["test_2_action_magnitude_sweep", "test_3_rollout_fidelity", "test_4_ood_actions"],
    "E": ["test_5_cross_network_state_head"],
}

TEST_ARTIFACTS = {
    "encoder_z_probe":               ("encoder_z/r2_report.json", None),
    "test_1_raw_zdiff":              ("action_response/action_response_report_normZ.json", None),
    "test_7_fresh_state_head_action_response":
                                     ("action_response/action_response_report_normZ.json", None),
    "test_9_predictor_modification": ("action_response/action_response_report_normZ.json", None),
    "test_10_offline_predloss":      ("action_response/action_response_report_normZ.json", None),
    "test_11_encoder_forward_coherence":
                                     ("action_response/action_response_report_normZ.json", None),
    "test_2_action_magnitude_sweep": ("action_response/action_response_report_normZ.json", None),
    "test_4_ood_actions":            ("action_response/action_response_report_normZ.json", None),
    "test_8_in_model_aux_head_action_response":
                                     ("action_response/action_response_report_normZ.json", None),
    "test_3_rollout_fidelity":       ("rollout_fidelity/rollout_fidelity.json", None),
    "test_5_cross_network_state_head":
                                     ("cross_head_{source_tag}/cross_head.json", None),
}


@dataclass
class EvalTarget:
    run_dir: Path
    epoch: int
    cfg: dict

    def epoch_dir(self) -> Path:
        return self.run_dir / f"epoch_{self.epoch}"

    def ckpt_path(self) -> Path:
        matches = list(self.run_dir.glob(f"*_epoch_{self.epoch}_object.ckpt"))
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint matching *_epoch_{self.epoch}_object.ckpt in {self.run_dir}"
            )
        return matches[0]


def resolve_requested_tests(include_clusters, tests, skip_tests,
                            include_encoder_z: bool = False) -> list[str]:
    """Given user's --include-clusters, --tests, --skip-tests, return the
    final list of test_name strings to run/read."""
    all_names = [n for cluster_tests in CLUSTERS.values() for n in cluster_tests]
    if tests:
        resolved = [n for n in all_names
                    if any(n.startswith(f"test_{t}_") for t in tests)]
        if include_encoder_z:
            resolved.append("encoder_z_probe")
    else:
        clusters = include_clusters or list(CLUSTERS.keys())
        resolved = [n for c in clusters for n in CLUSTERS.get(c, [])]
    if skip_tests:
        resolved = [n for n in resolved
                    if not any(n.startswith(f"test_{t}_") for t in skip_tests)]
    return resolved


def _run_encoder_z_probe(target: EvalTarget, cache_dir: Path):
    out_dir = target.epoch_dir() / "encoder_z"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "lewm/scripts/train_state_head.py",
        "--model", str(target.ckpt_path()),
        "--dataset", *target.cfg["datasets"],
        "--cache-dir", str(cache_dir),
        "--output-dir", str(out_dir),
        "--action-norm-ref", target.cfg["action_norm_ref"],
        "--max-frames-per-dataset", "75000",
        "--val-split", "0.25",
    ]
    subprocess.run(cmd, check=True)


def _run_predicted_z_probe(target: EvalTarget, cache_dir: Path):
    out_dir = target.epoch_dir() / "predicted_z"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "lewm/scripts/train_state_head.py",
        "--model", str(target.ckpt_path()),
        "--dataset", *target.cfg["datasets"],
        "--cache-dir", str(cache_dir),
        "--output-dir", str(out_dir),
        "--action-norm-ref", target.cfg["action_norm_ref"],
        "--ctx-len", str(target.cfg["ctx_len"]),
        "--n-preds", str(target.cfg["n_preds"]),
        "--predicted-z",
        "--training-aligned",
        "--normalize-actions",
        "--max-frames-per-dataset", "75000",
        "--val-split", "0.25",
    ]
    subprocess.run(cmd, check=True)


def _run_action_response_group(target: EvalTarget, cache_dir: Path,
                               state_head_path: Path, probe_dataset: str,
                               tests_requested: set[str]):
    """Runs test_action_response.py in ONE invocation covering all action-response-
    family tests that share z_history: Tests 1, 2, 4, 7, 9, 10, 11, and 8."""
    out_dir = target.epoch_dir() / "action_response"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "lewm/scripts/test_action_response.py",
        "--model", str(target.ckpt_path()),
        "--state-head", str(state_head_path),
        "--dataset", probe_dataset,
        "--cache-dir", str(cache_dir),
        "--output-dir", str(out_dir),
        "--ctx-len", str(target.cfg["ctx_len"]),
        "--n-preds", str(target.cfg["n_preds"]),
        "--action-norm-ref", target.cfg["action_norm_ref"],
        "--n-frames", "500",
    ]
    if "test_1_raw_zdiff" in tests_requested:
        cmd.append("--raw-zdiff")
    if {"test_9_predictor_modification", "test_10_offline_predloss",
        "test_11_encoder_forward_coherence"} & tests_requested:
        cmd.append("--predictor-activity")
    if "test_2_action_magnitude_sweep" in tests_requested:
        cmd.append("--magnitude-sweep")
    if "test_4_ood_actions" in tests_requested:
        cmd.append("--ood-actions")
    if "test_8_in_model_aux_head_action_response" in tests_requested:
        cmd.append("--use-internal-state-head")
    if "test_7_fresh_state_head_action_response" not in tests_requested:
        cmd.append("--skip-test-7")
    subprocess.run(cmd, check=True)


def _run_rollout_fidelity(target: EvalTarget, cache_dir: Path,
                          state_head_path: Path):
    out_dir = target.epoch_dir() / "rollout_fidelity"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "lewm/scripts/eval_rollout_fidelity.py",
        "--model", str(target.ckpt_path()),
        "--state-head", str(state_head_path),
        "--datasets", *target.cfg["datasets"][:4],
        "--cache-dir", str(cache_dir),
        "--output-dir", str(out_dir),
        "--ctx-len", str(target.cfg["ctx_len"]),
        "--n-preds", str(target.cfg["n_preds"]),
        "--action-norm-ref", target.cfg["action_norm_ref"],
        "--n-episodes", "20", "--seq-len", "20",
    ]
    subprocess.run(cmd, check=True)


def _run_cross_head(target: EvalTarget, cache_dir: Path,
                    reference_state_head: Path, source_tag: str,
                    predicted_z_cache: Path):
    out_dir = target.epoch_dir() / f"cross_head_{source_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "lewm/scripts/eval_trained_aux_head.py",
        "--model", str(target.ckpt_path()),
        "--cache", str(predicted_z_cache),
        "--state-head", str(reference_state_head),
        "--report-out", str(out_dir / "cross_head.txt"),
    ]
    subprocess.run(cmd, check=True)
