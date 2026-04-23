"""Tests for lewm.eval.action_response_metrics pure functions."""
import numpy as np
import pytest

from lewm.eval.action_response_metrics import compute_relative_shift, compute_offline_predloss


def test_relative_shift_identical_arrays_is_zero():
    z = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    out = compute_relative_shift(z, z)
    assert out["rel_norm"] == pytest.approx(0.0, abs=1e-6)


def test_relative_shift_rel_norm_handcompute():
    # z_a: norm sqrt(1+4+9+16) = sqrt(30)
    # z_b = z_a + [1,0,0,0]: delta norm = 1
    # rel = 1 / sqrt(30)
    z_a = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    z_b = z_a + np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    out = compute_relative_shift(z_a, z_b)
    expected = 1.0 / np.sqrt(30)
    assert out["rel_norm"] == pytest.approx(expected, rel=1e-5)


def test_relative_shift_top_dims_identifies_biggest_mover():
    # Delta moves dim 2 by 5, dim 0 by 3, dim 1 by 0, dim 3 by 1.
    # top-2 by L∞ should be [2, 0].
    z_a = np.zeros((1, 4), dtype=np.float32)
    z_b = np.array([[3.0, 0.0, 5.0, 1.0]], dtype=np.float32)
    out = compute_relative_shift(z_a, z_b, top_k=2)
    assert list(out["top_dims"]) == [2, 0]


def test_relative_shift_averages_across_batch():
    # Two samples: delta norms 1 and 3, z_a norms 1 and 1 → rel_norms [1, 3], mean 2.
    z_a = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    z_b = np.array([[1.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    out = compute_relative_shift(z_a, z_b)
    assert out["rel_norm"] == pytest.approx(2.0, rel=1e-5)


def test_offline_predloss_zero_when_pred_equals_true():
    z_pred = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    z_true = z_pred.copy()
    z_baseline = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    out = compute_offline_predloss(z_pred, z_true, z_baseline)
    assert out["mse"] == pytest.approx(0.0, abs=1e-6)
    # Baseline MSE = mean((0-1)² + (0-2)² + (0-3)²) = (1+4+9)/3 = 14/3
    assert out["baseline_mse"] == pytest.approx(14.0 / 3, rel=1e-5)


def test_offline_predloss_mse_handcompute():
    # z_pred - z_true = [[1, 0], [0, 2]]; squared = [[1, 0], [0, 4]];
    # mean over all elements = (1+0+0+4) / 4 = 1.25
    z_pred = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    z_true = np.zeros_like(z_pred)
    z_baseline = np.zeros_like(z_pred)  # baseline matches true → MSE 0
    out = compute_offline_predloss(z_pred, z_true, z_baseline)
    assert out["mse"] == pytest.approx(1.25, rel=1e-5)
    assert out["baseline_mse"] == pytest.approx(0.0, abs=1e-6)


def test_offline_predloss_beats_baseline_flag():
    # Predictor achieves MSE < baseline MSE → beats_baseline = True
    z_true = np.array([[10.0, 20.0]], dtype=np.float32)
    z_pred = np.array([[10.1, 20.1]], dtype=np.float32)  # small error
    z_baseline = np.array([[0.0, 0.0]], dtype=np.float32)  # big error
    out = compute_offline_predloss(z_pred, z_true, z_baseline)
    assert out["beats_baseline"] is True

    # Flip: predictor worse than baseline
    out2 = compute_offline_predloss(z_baseline, z_true, z_pred)
    assert out2["beats_baseline"] is False
