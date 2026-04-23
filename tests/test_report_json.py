"""Tests for lewm.eval.report_json — shared JSON report emission."""
import argparse
import json

import pytest

from lewm.eval.report_json import write_json_report, metadata_from_args, SCHEMA_VERSION


def test_write_json_report_writes_file_with_schema(tmp_path):
    path = write_json_report(tmp_path, "myreport",
                             results_dict={"test_1_foo": {"x": 1.0}},
                             metadata={"model": "m", "dataset": "d"})
    assert path == tmp_path / "myreport.json"
    loaded = json.loads(path.read_text())
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["metadata"] == {"model": "m", "dataset": "d"}
    assert loaded["results"] == {"test_1_foo": {"x": 1.0}}


def test_metadata_from_args_pulls_common_fields():
    ns = argparse.Namespace(
        model="/m.ckpt", dataset="ds1", action_norm_ref="ds1",
        ctx_len=3, n_preds=1, normalize_actions=True, n_frames=100,
        unrelated_flag=True,   # should NOT appear
    )
    meta = metadata_from_args(ns)
    assert meta["model"] == "/m.ckpt"
    assert meta["dataset"] == "ds1"
    assert meta["ctx_len"] == 3
    assert "unrelated_flag" not in meta


def test_metadata_from_args_dataset_list_is_preserved():
    ns = argparse.Namespace(model="m", dataset=["a", "b"])
    meta = metadata_from_args(ns)
    assert meta["dataset"] == ["a", "b"]


def test_metadata_from_args_extra_overrides_and_extends():
    ns = argparse.Namespace(model="m", dataset="d")
    meta = metadata_from_args(ns, extra={"seed": 42, "model": "override"})
    assert meta["seed"] == 42
    assert meta["model"] == "override"
