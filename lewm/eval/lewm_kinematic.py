"""LeWM subclass with kinematic cost (state-head-based) instead of image cost."""

from pathlib import Path
import sys

import torch
from einops import rearrange

LEWM_DIR = Path(__file__).resolve().parent.parent / "vendor" / "le-wm"
sys.path.insert(0, str(LEWM_DIR))

from stable_worldmodel.wm.lewm.lewm import LeWM  # noqa: E402


class LeWMKinematic(LeWM):
    """LeWM variant with kinematic-goal cost.

    get_cost() is overridden: instead of image-z distance, we roll out in z
    space, decode each predicted z via a state head, and compute weighted
    MSE between the decoded kinematic state and a target state.
    """

    def __init__(self, *args, state_head=None, target_state=None,
                 kinematic_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert state_head is not None
        assert target_state is not None
        assert kinematic_weights is not None
        self.state_head = state_head
        self.register_buffer("target_state", torch.as_tensor(target_state, dtype=torch.float32))
        self.register_buffer("kinematic_weights", torch.as_tensor(kinematic_weights, dtype=torch.float32))

    @classmethod
    def from_base(cls, base, state_head, target_state, kinematic_weights):
        """Build a LeWMKinematic that shares weights with an existing LeWM instance."""
        inst = cls.__new__(cls)
        torch.nn.Module.__init__(inst)
        inst.encoder = base.encoder
        inst.predictor = base.predictor
        inst.action_encoder = base.action_encoder
        inst.projector = base.projector
        inst.pred_proj = base.pred_proj
        inst.state_head = state_head
        inst.register_buffer(
            "target_state", torch.as_tensor(target_state, dtype=torch.float32)
        )
        inst.register_buffer(
            "kinematic_weights", torch.as_tensor(kinematic_weights, dtype=torch.float32)
        )
        return inst

    def compute_kinematic_cost(self, emb_rollout: torch.Tensor) -> torch.Tensor:
        """Compute weighted kinematic MSE at the terminal step of each rollout.

        Args:
            emb_rollout: (B, S, T, D) predicted embedding trajectory from rollout().

        Returns:
            cost: (B, S) weighted MSE between decoded terminal state and target.
        """
        B, S, T, D = emb_rollout.shape
        z_terminal = emb_rollout[:, :, -1, :]  # (B, S, D)
        z_flat = rearrange(z_terminal, "b s d -> (b s) d")
        state_pred = self.state_head(z_flat)  # ((B*S), 6)
        state_pred = rearrange(state_pred, "(b s) k -> b s k", b=B, s=S)

        # target_state can be (K,) for single target or (B, K) for per-env targets
        if self.target_state.ndim == 1:
            target = self.target_state.view(1, 1, -1)       # (1, 1, K)
        else:
            target = self.target_state.unsqueeze(1)          # (B, 1, K)
        weights = self.kinematic_weights.view(1, 1, -1)      # (1, 1, K)
        diff2 = (state_pred - target) ** 2
        cost = (diff2 * weights).sum(dim=-1)  # (B, S)
        return cost

    def set_target(self, target_state: torch.Tensor):
        """Update target state (supports batched (B, K) for per-env goals)."""
        self.target_state = target_state.to(self.kinematic_weights.device)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Override: kinematic cost instead of image-goal cost."""
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        info_dict = self.rollout(info_dict, action_candidates)
        pred_rollout = info_dict["predicted_emb"]  # (B, S, T, D)
        return self.compute_kinematic_cost(pred_rollout)


def build_kinematic_from_paths(
    model_path: str,
    state_head_path: str,
    target_state: "list[float] | torch.Tensor",
    kinematic_weights: "list[float] | torch.Tensor",
    device: str = "cuda",
) -> "LeWMKinematic":
    """Load a LeWM ckpt + state head and return a LeWMKinematic ready for planning."""
    from lewm.eval.state_head import load_state_head

    base = torch.load(model_path, map_location=device, weights_only=False)
    base.eval()
    state_head, _ = load_state_head(state_head_path, device=device)
    state_head.eval()
    kin = LeWMKinematic.from_base(base, state_head, target_state, kinematic_weights)
    kin = kin.to(device)
    kin.eval()
    return kin
