"""Shared JSON report emission for the E5 JEPA evaluation suite.

All eval scripts (test_action_response.py, eval_rollout_fidelity.py,
eval_trained_aux_head.py, train_state_head.py) write a sibling JSON
alongside their text report using the helpers in this module. Using
one module keeps the schema single-sourced.

Schema: {schema_version, metadata, results}. See per-task JSON-contract
table in the plan for the expected `results` keys per script.
"""
from __future__ import annotations
from pathlib import Path
import json


SCHEMA_VERSION = "e5-eval-suite-v1"


_STANDARD_FIELDS = (
    "model", "dataset", "datasets", "action_norm_ref",
    "ctx_len", "n_preds", "normalize_actions", "n_frames",
)


def write_json_report(out_dir, basename: str, results_dict: dict,
                      metadata: dict) -> Path:
    """Write a structured JSON sibling of a text report.

    Args:
        out_dir: directory to write into (created if missing).
        basename: filename stem, e.g. 'action_response_report_normZ'.
        results_dict: mapping {test_name: test_result_dict}.
        metadata: checkpoint/config metadata to embed at top-level.

    Returns the path written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{basename}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "results": results_dict,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")
    return path


def metadata_from_args(args, extra: dict | None = None) -> dict:
    """Build the standard metadata dict from an argparse Namespace.

    Pulls common fields if present on args (model, dataset(s), action_norm_ref,
    ctx_len, n_preds, normalize_actions, n_frames). Caller may pass `extra`
    to add per-script fields (e.g. state_head_path, seq_len) or override.

    `dataset` values that are lists are preserved as lists; strings preserved
    as strings.
    """
    meta: dict = {}
    for field in _STANDARD_FIELDS:
        val = getattr(args, field, None)
        if val is None:
            continue
        if field in ("dataset", "datasets") and not isinstance(val, str):
            val = list(val)
        meta[field] = val
    if extra:
        meta.update(extra)
    return meta
