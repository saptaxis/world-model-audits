"""State head probe: MLP mapping LeWorldModel z → 6D kinematic state.

Train on cached (z, state) pairs from encode_dataset.py.
Reports R² per kinematic dimension.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


KIN_DIM_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel"]


class StateHead(nn.Module):
    """Small MLP: z (192) → kinematic state (6)."""

    def __init__(self, z_dim: int = 192, hidden: int = 128, state_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def train_state_head(
    z_train: np.ndarray,
    state_train: np.ndarray,
    z_val: np.ndarray,
    state_val: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = "cuda",
) -> tuple[StateHead, dict]:
    """Train state head and return model + R² scores."""
    device = torch.device(device)
    head = StateHead(z_dim=z_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    z_t = torch.from_numpy(z_train).float()
    s_t = torch.from_numpy(state_train).float()
    z_v = torch.from_numpy(z_val).float().to(device)
    s_v = torch.from_numpy(state_val).float().to(device)

    n = len(z_t)
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            zb = z_t[idx].to(device)
            sb = s_t[idx].to(device)
            pred = head(zb)
            loss = nn.functional.mse_loss(pred, sb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            head.eval()
            with torch.no_grad():
                val_pred = head(z_v)
                val_mse = nn.functional.mse_loss(val_pred, s_v).item()
            print(f"  epoch {epoch+1:3d}: train_mse={epoch_loss/n_batches:.6f}  val_mse={val_mse:.6f}")

    # Final evaluation
    head.eval()
    with torch.no_grad():
        val_pred = head(z_v).cpu().numpy()
    val_gt = state_val

    r2_per_dim = {}
    for i, name in enumerate(KIN_DIM_NAMES):
        r2 = r2_score(val_gt[:, i], val_pred[:, i])
        r2_per_dim[name] = r2

    r2_mean = np.mean(list(r2_per_dim.values()))
    val_mse_final = float(np.mean((val_pred - val_gt) ** 2))

    return head, {
        "r2_per_dim": r2_per_dim,
        "r2_mean": r2_mean,
        "val_mse": val_mse_final,
    }


def save_state_head(head: StateHead, metrics: dict, path: str):
    """Save state head checkpoint + metrics."""
    torch.save({"model": head.state_dict(), "metrics": metrics}, path)
    print(f"Saved state head to {path}")


def load_state_head(path: str, device: str = "cpu") -> tuple[StateHead, dict]:
    """Load state head from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    head = StateHead()
    head.load_state_dict(ckpt["model"])
    head.to(device)
    head.eval()
    return head, ckpt["metrics"]
