"""Autoregressive rollout in LeWorldModel z-space with state head decode.

Produces predicted vs actual state arrays for visualization.
"""

import sys
from pathlib import Path

import numpy as np
import torch

LEWM_DIR = Path(__file__).resolve().parent.parent / "vendor" / "le-wm"
sys.path.insert(0, str(LEWM_DIR))

import stable_worldmodel as swm
from utils import get_img_preprocessor

from .state_head import StateHead, load_state_head


def rollout_episode(
    model,
    state_head: StateHead,
    pixels: torch.Tensor,
    actions: torch.Tensor,
    history_size: int = 3,
    device: str = "cuda",
) -> dict:
    """Rollout one episode: encode seed frames, predict autoregressively, decode states."""
    device = torch.device(device)
    model.eval()
    state_head.eval()
    T = len(pixels)
    HS = history_size

    with torch.no_grad():
        px = pixels.to(device)
        output = model.encoder(px, interpolate_pos_encoding=True)
        z_gt = model.projector(output.last_hidden_state[:, 0])

        actual_states = state_head(z_gt).cpu().numpy()

        z_pred = z_gt[:HS].clone()
        actions_t = actions.to(device)
        actions_t = torch.nan_to_num(actions_t, 0.0)
        act_so_far = actions_t[:HS].unsqueeze(0)

        for t in range(HS, T):
            act_emb = model.action_encoder(act_so_far)
            z_trunc = z_pred[-HS:].unsqueeze(0)
            act_trunc = act_emb[:, -HS:]
            pred = model.predict(z_trunc, act_trunc)[:, -1]  # (1, z_dim)
            z_pred = torch.cat([z_pred, pred], dim=0)
            act_so_far = torch.cat(
                [act_so_far, actions_t[t : t + 1].unsqueeze(0)], dim=1
            )

        predicted_states = state_head(z_pred).cpu().numpy()

    return {
        "predicted_states": predicted_states,
        "actual_states": actual_states,
        "z_pred": z_pred.cpu().numpy(),
        "z_gt": z_gt.cpu().numpy(),
    }


def rollout_episodes(
    model_path: str,
    state_head_path: str,
    dataset_name: str,
    cache_dir: str,
    n_episodes: int = 5,
    seq_len: int = 50,
    frameskip: int = 10,
    device: str = "cuda",
) -> list[dict]:
    """Rollout multiple episodes and return results."""
    device_t = torch.device(device)

    model = torch.load(model_path, map_location=device_t, weights_only=False)
    model.eval()

    state_head, _ = load_state_head(state_head_path, device=device)

    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    ds = swm.data.HDF5Dataset(
        name=dataset_name,
        num_steps=seq_len,
        frameskip=frameskip,
        keys_to_load=["pixels", "action", "state"],
        cache_dir=cache_dir,
        transform=transform,
    )

    # Sample random episodes (not evenly spaced clips)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(ds), size=n_episodes, replace=False)
    results = []

    for idx in indices:
        sample = ds[int(idx)]
        pixels = sample["pixels"]
        actions = sample["action"]
        gt_states = sample["state"].numpy()[:, :6]  # real GT kinematics

        rollout = rollout_episode(
            model, state_head, pixels, actions,
            device=device,
        )
        rollout["actual_states"] = gt_states  # override with real GT
        rollout["actions"] = actions.numpy()
        results.append(rollout)
        print(f"  clip {idx}: {len(pixels)} steps, "
              f"final z-MSE={np.mean((rollout['z_pred'][-1] - rollout['z_gt'][-1])**2):.4f}")

    return results
