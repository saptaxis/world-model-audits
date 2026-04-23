"""Pure functions for computing predicted/encoded-z differential metrics
used by the E5 JEPA evaluation suite (Tests 1, 9, 10, 11).

All functions take numpy arrays and return dicts of scalar floats / lists.
No torch, no I/O, no model loading — keeps things testable and fast.
See specs/e5-jepa-eval-suite-Apr222026.md for test definitions.
"""

from __future__ import annotations

import numpy as np


def compute_relative_shift(z_a: np.ndarray, z_b: np.ndarray, top_k: int = 5) -> dict:
    """‖z_b − z_a‖ / ‖z_a‖ averaged over the batch, plus top-k L∞ dims.

    Used by Test 1 (z_a = z_no_action, z_b = z_action) and by Test 9
    (z_a = last_context_z, z_b = z_predicted_at_recorded_action) and by
    Test 11 (z_a = z_t, z_b = z_{t+n_preds}).

    Args:
        z_a, z_b: arrays of shape (N, D). z_a is the reference for
            normalization in the denominator.
        top_k: how many top-moving z-dims to report (by mean |Δ| across samples).

    Returns:
        dict with:
          - rel_norm: mean of per-sample ‖Δ‖ / ‖z_a‖, over the N samples.
          - top_dims: list of dim indices, sorted by descending mean |Δ|.
    """
    z_a = np.asarray(z_a, dtype=np.float32)
    z_b = np.asarray(z_b, dtype=np.float32)
    assert z_a.shape == z_b.shape and z_a.ndim == 2, \
        f"expected (N, D) arrays, got {z_a.shape} vs {z_b.shape}"
    delta = z_b - z_a                               # (N, D)
    delta_norm = np.linalg.norm(delta, axis=1)      # (N,)
    ref_norm = np.linalg.norm(z_a, axis=1)          # (N,)
    # Avoid 0/0: if ref_norm is exactly zero for a row, treat that row as
    # skipped (very unlikely with real encoder output, but be explicit).
    valid = ref_norm > 0
    rel = np.zeros_like(delta_norm)
    rel[valid] = delta_norm[valid] / ref_norm[valid]
    rel_mean = float(rel.mean()) if rel.size > 0 else 0.0

    mean_abs = np.abs(delta).mean(axis=0)           # (D,)
    top_dims = np.argsort(-mean_abs)[:top_k].tolist()
    return {"rel_norm": rel_mean, "top_dims": top_dims}


def compute_offline_predloss(
    z_pred: np.ndarray,
    z_true_next: np.ndarray,
    z_last_ctx: np.ndarray,
) -> dict:
    """Predictor accuracy on the probe set (Test 10).

    Compares the predictor's output to the encoder's actual next-frame z,
    with a "predict-identity" baseline for context.

    Args:
        z_pred: (N, D) model.predict(history, action)[:, -1].
        z_true_next: (N, D) projector(encoder(frame_{t+n_preds})).
        z_last_ctx: (N, D) z_history[:, -1] — the last real history z.

    Returns:
        dict with:
          - mse: mean((z_pred − z_true_next)^2).
          - baseline_mse: mean((z_last_ctx − z_true_next)^2).
          - beats_baseline: True iff mse < baseline_mse.
    """
    z_pred = np.asarray(z_pred, dtype=np.float32)
    z_true = np.asarray(z_true_next, dtype=np.float32)
    z_ctx = np.asarray(z_last_ctx, dtype=np.float32)
    assert z_pred.shape == z_true.shape == z_ctx.shape, \
        f"shape mismatch: {z_pred.shape} {z_true.shape} {z_ctx.shape}"
    mse = float(((z_pred - z_true) ** 2).mean())
    baseline_mse = float(((z_ctx - z_true) ** 2).mean())
    return {"mse": mse, "baseline_mse": baseline_mse,
            "beats_baseline": bool(mse < baseline_mse)}
