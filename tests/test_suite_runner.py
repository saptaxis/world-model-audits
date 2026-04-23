"""Tests for lewm.eval.suite_runner — config auto-inference + discovery."""
from pathlib import Path

import pytest
import yaml


def test_infer_config_reads_hydra_config_yaml(tmp_path):
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump({
        "wm": {"history_size": 3, "num_preds": 2,
               "aux_loss": {"enabled": True, "kin_block": 16}},
        "data": {"dataset": {"name": ["dsA_clean", "dsB_clean"], "frameskip": 10}},
    }))

    from lewm.eval.suite_runner import infer_config_from_run_dir
    cfg = infer_config_from_run_dir(run_dir)
    assert cfg["ctx_len"] == 3
    assert cfg["n_preds"] == 2
    assert cfg["kin_block"] == 16
    assert cfg["action_norm_ref"] == "dsA_clean"
    assert cfg["datasets"] == ["dsA_clean", "dsB_clean"]
    assert cfg["frameskip"] == 10


def test_infer_config_no_aux_block(tmp_path):
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump({
        "wm": {"history_size": 3, "num_preds": 1},
        "data": {"dataset": {"name": ["heur"], "frameskip": 10}},
    }))
    from lewm.eval.suite_runner import infer_config_from_run_dir
    cfg = infer_config_from_run_dir(run_dir)
    assert cfg["kin_block"] is None


def test_resolve_tests_include_clusters():
    from lewm.eval.suite_runner import resolve_requested_tests
    out = resolve_requested_tests(include_clusters=["B"], tests=None, skip_tests=None)
    assert set(out) == {"test_9_predictor_modification",
                         "test_10_offline_predloss",
                         "test_11_encoder_forward_coherence"}


def test_resolve_tests_explicit_numbers():
    from lewm.eval.suite_runner import resolve_requested_tests
    out = resolve_requested_tests(include_clusters=None, tests=[1, 3], skip_tests=None)
    assert set(out) == {"test_1_raw_zdiff", "test_3_rollout_fidelity"}


def test_resolve_tests_with_skip():
    from lewm.eval.suite_runner import resolve_requested_tests
    out = resolve_requested_tests(include_clusters=["B"], tests=None, skip_tests=[10])
    assert set(out) == {"test_9_predictor_modification",
                         "test_11_encoder_forward_coherence"}
