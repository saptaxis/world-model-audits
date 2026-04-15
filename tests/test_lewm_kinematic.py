"""Tests for LeWMKinematic: state-head-based cost for kinematic goals."""
import numpy as np
import pytest
import torch


def _make_fake_model(device="cpu"):
    """Build a minimal fake model exposing the .encoder/.predictor/.projector/.action_encoder interface."""

    class FakeEncoder(torch.nn.Module):
        def forward(self, x, interpolate_pos_encoding=True):
            class Out:
                pass
            out = Out()
            out.last_hidden_state = torch.zeros(x.size(0), 1, 8, device=x.device)
            return out

    class FakePredictor(torch.nn.Module):
        def forward(self, z, a):
            return z

    class FakeActionEncoder(torch.nn.Module):
        def forward(self, a):
            return a

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FakeEncoder()
            self.predictor = FakePredictor()
            self.projector = torch.nn.Identity()
            self.action_encoder = FakeActionEncoder()
            self.pred_proj = torch.nn.Identity()

    return FakeModel()


def test_kinematic_cost_zero_at_target():
    """If predicted state == target, cost should be ~0."""
    from lewm.eval.lewm_kinematic import LeWMKinematic

    model = _make_fake_model()
    target = torch.tensor([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_head = torch.nn.Linear(8, 6)
    with torch.no_grad():
        state_head.weight.zero_()
        state_head.bias.copy_(target)

    weights = torch.tensor([1, 1, 0.5, 0.5, 1, 0])
    kin = LeWMKinematic.from_base(model, state_head, target, weights)

    B, S, T, D = 2, 4, 5, 8
    info = {
        "emb": torch.zeros(B, S, T, D),
    }
    cost = kin.compute_kinematic_cost(info["emb"])
    assert cost.shape == (B, S)
    assert torch.all(cost.abs() < 1e-5), f"Expected near-zero cost, got {cost}"


def test_kinematic_cost_nonzero_away_from_target():
    """If predicted state differs, cost should be positive."""
    from lewm.eval.lewm_kinematic import LeWMKinematic

    model = _make_fake_model()
    state_head = torch.nn.Linear(8, 6)
    with torch.no_grad():
        state_head.weight.zero_()
        state_head.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    target = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    weights = torch.tensor([1, 1, 0.5, 0.5, 1, 0])
    kin = LeWMKinematic.from_base(model, state_head, target, weights)

    emb = torch.zeros(2, 4, 5, 8)
    cost = kin.compute_kinematic_cost(emb)
    assert torch.all(cost > 0.5)


def test_build_kinematic_from_paths(tmp_path):
    """Factory loads LeWM ckpt + state head pt, returns LeWMKinematic."""
    import os

    model_ckpt = "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/lewm-runs/synthetic-heuristic-fs10/lewm_lunarlander_synthetic_heuristic_fs10_epoch_30_object.ckpt"
    sh_ckpt = "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/lewm-runs/synthetic-heuristic-fs10/state_head_epoch30/state_head.pt"
    if not (os.path.exists(model_ckpt) and os.path.exists(sh_ckpt)):
        pytest.skip("Heuristic ckpt not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from lewm.eval.lewm_kinematic import build_kinematic_from_paths

    kin = build_kinematic_from_paths(
        model_path=model_ckpt,
        state_head_path=sh_ckpt,
        target_state=[0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        kinematic_weights=[1, 1, 0.5, 0.5, 1, 0],
        device="cuda",
    )
    assert hasattr(kin, "get_cost")
    assert hasattr(kin, "state_head")
    assert kin.target_state.shape == (6,)
