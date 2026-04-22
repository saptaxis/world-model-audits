"""Autoregressive rollout in LeWorldModel z-space with state head decode.

Produces predicted vs actual state arrays for visualization.
"""

import sys
from pathlib import Path

import numpy as np
import torch

LEWM_DIR = Path(__file__).resolve().parent.parent / "vendor" / "le-wm"
sys.path.insert(0, str(LEWM_DIR))

import h5py
import hdf5plugin  # noqa: F401
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
    act_mean: np.ndarray | None = None,
    act_std: np.ndarray | None = None,
    z_slice: tuple[int, int] | None = None,
) -> dict:
    """Rollout one episode: encode seed frames, predict autoregressively, decode states.

    If act_mean/act_std are provided, actions are z-score normalized per raw
    action dim before being fed to the action_encoder (mirrors training for
    norm-trained checkpoints). Actions come out of HDF5Dataset shape (T, fs*2)
    where each row is row-major-flattened (raw0_a0, raw0_a1, raw1_a0, ...),
    so normalization tiles the (2,) mean/std across the fs raw frames.

    If z_slice is provided, z is sliced before feeding to state_head (for heads
    trained on a z-subspace, e.g. (0, 16) for v2's dedicated z_kin block).

    NOTE on num_preds>1: this rollout advances history by one logical position
    per iteration regardless of the predictor's training n_preds. For n_preds>1
    checkpoints the rollout is off-regime (predictor's pred[:, -1] is n_preds
    steps ahead, but the loop integrates it as if 1 step ahead). This keeps
    the signal directionally interpretable but the time axis is compressed by
    n_preds. A proper n_preds-aware rollout would advance by n_preds each iter.
    """
    device = torch.device(device)
    model.eval()
    state_head.eval()
    T = len(pixels)
    HS = history_size

    with torch.no_grad():
        px = pixels.to(device)
        output = model.encoder(px, interpolate_pos_encoding=True)
        z_gt = model.projector(output.last_hidden_state[:, 0])

        z_pred = z_gt[:HS].clone()
        actions_t = actions.to(device)
        actions_t = torch.nan_to_num(actions_t, 0.0)
        if act_mean is not None:
            # actions_t: (T, fs*action_dim=20). Reshape -> (T, fs, action_dim)
            # so (2,) mean/std broadcasts over the last dim, then flatten back.
            act_dim = len(act_mean)
            fs = actions_t.shape[1] // act_dim
            m = torch.from_numpy(act_mean).to(device)
            s = torch.from_numpy(act_std).to(device)
            actions_t = ((actions_t.view(T, fs, act_dim) - m) / s).reshape(T, fs * act_dim)
        act_so_far = actions_t[:HS].unsqueeze(0)

        for t in range(HS, T):
            act_emb = model.action_encoder(act_so_far)
            z_trunc = z_pred[-HS:].unsqueeze(0)
            act_trunc = act_emb[:, -HS:]
            pred = model.predict(z_trunc, act_trunc)[:, -1]
            z_pred = torch.cat([z_pred, pred], dim=0)
            act_so_far = torch.cat(
                [act_so_far, actions_t[t : t + 1].unsqueeze(0)], dim=1
            )

        z_for_head = z_pred if z_slice is None else z_pred[..., z_slice[0]:z_slice[1]]
        predicted_states = state_head(z_for_head).cpu().numpy()

    return {
        "predicted_states": predicted_states,
        "z_pred": z_pred.cpu().numpy(),
        "z_gt": z_gt.cpu().numpy(),
    }


def _get_episode_clip_indices(
    h5_path: str, frameskip: int, seq_len: int
) -> list[tuple[int, int, int]]:
    """Get valid (ep_idx, start_in_episode, global_offset) for each episode.

    Returns list of (episode_index, local_start, global_start) tuples
    where a clip of seq_len frames (with frameskip) fits within the episode.
    """
    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]

    span = seq_len * frameskip
    clips = []
    for ep_idx in range(len(ep_len)):
        length = int(ep_len[ep_idx])
        if length >= span:
            clips.append((ep_idx, int(ep_offset[ep_idx]), length))
    return clips, ep_len, ep_offset


def rollout_episodes(
    model_path: str,
    state_head_path: str,
    dataset_name: str,
    cache_dir: str,
    n_episodes: int = 5,
    seq_len: int = 50,
    frameskip: int = 10,
    start_mode: str = "random",
    rgb_dataset_name: str | None = None,
    device: str = "cuda",
    seed: int = 42,
    normalize_actions: bool = True,
    action_norm_ref: str = "lunarlander_synthetic_heuristic_clean",
    ctx_len: int = 3,
    n_preds: int = 1,
) -> list[dict]:
    """Rollout multiple episodes and return results.

    Args:
        start_mode: How to pick clips within episodes.
            'random' — random clips from anywhere in the dataset
            'episode_start' — start from the beginning of random episodes
            'episode_mid' — start from 25-75% through random episodes
        rgb_dataset_name: If set, also loads RGB frames from this dataset
            and includes them in the rollout dict as 'rgb_frames'.
    """
    device_t = torch.device(device)

    model = torch.load(model_path, map_location=device_t, weights_only=False)
    model.eval()

    state_head, sh_metrics = load_state_head(state_head_path, device=device)
    z_slice = sh_metrics.get("z_slice")  # (start, end) or None for full z
    if z_slice is not None:
        print(f"  state head expects sliced z [{z_slice[0]}:{z_slice[1]}]")

    # Optional action normalizer (mirrors training for norm-trained checkpoints).
    act_mean = act_std = None
    if normalize_actions:
        from .action_norm import compute_action_normalizer
        act_mean, act_std = compute_action_normalizer(action_norm_ref, cache_dir)
        print(f"  action normalizer ON (ref={action_norm_ref}): "
              f"mean={act_mean.tolist()} std={act_std.tolist()}")
    else:
        print(f"  action normalizer OFF (raw actions fed to action_encoder)")

    print(f"  ctx_len={ctx_len}, n_preds={n_preds}"
          + (" (rollout will advance 1 logical step per iter; off-regime for n_preds>1)"
             if n_preds > 1 else ""))

    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=224)

    # Main dataset (what the encoder sees)
    ds = swm.data.HDF5Dataset(
        name=dataset_name,
        num_steps=seq_len,
        frameskip=frameskip,
        keys_to_load=["pixels", "action", "state"],
        cache_dir=cache_dir,
        transform=transform,
    )

    # Optional RGB dataset for visualization
    ds_rgb = None
    if rgb_dataset_name:
        ds_rgb = swm.data.HDF5Dataset(
            name=rgb_dataset_name,
            num_steps=seq_len,
            frameskip=frameskip,
            keys_to_load=["pixels"],
            cache_dir=cache_dir,
            transform=transform,
        )

    rng = np.random.default_rng(seed)

    if start_mode == "random":
        indices = rng.choice(len(ds), size=n_episodes, replace=False)
    elif start_mode in ("episode_start", "episode_mid"):
        # Get episode structure
        datasets_dir = swm.data.utils.get_cache_dir(cache_dir, sub_folder="datasets")
        h5_path = str(Path(datasets_dir) / f"{dataset_name}.h5")
        clips, ep_len, ep_offset = _get_episode_clip_indices(
            h5_path, frameskip, seq_len
        )
        # Pick random episodes that are long enough
        ep_indices = rng.choice(len(clips), size=n_episodes, replace=False)

        # Map episode starts to dataset clip indices
        indices = []
        span = seq_len * frameskip
        for ei in ep_indices:
            ep_idx, global_off, length = clips[ei]
            if start_mode == "episode_start":
                local_start = 0
            else:  # episode_mid
                max_start = length - span
                mid_lo = max(0, int(0.25 * max_start))
                mid_hi = min(max_start, int(0.75 * max_start))
                if mid_hi <= mid_lo:
                    local_start = mid_lo
                else:
                    local_start = rng.integers(mid_lo, mid_hi + 1)

            # Find the clip index in ds.clip_indices that matches
            target = (ep_idx, local_start)
            try:
                clip_idx = ds.clip_indices.index(target)
            except ValueError:
                # Find closest valid clip start for this episode
                ep_clips = [
                    (i, s) for i, (e, s) in enumerate(ds.clip_indices) if e == ep_idx
                ]
                if ep_clips:
                    closest = min(ep_clips, key=lambda x: abs(x[1] - local_start))
                    clip_idx = closest[0]
                else:
                    clip_idx = 0
            indices.append(clip_idx)
        indices = np.array(indices)
    else:
        raise ValueError(f"Unknown start_mode: {start_mode}")

    # ImageNet denormalization for RGB display
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    results = []
    for idx in indices:
        sample = ds[int(idx)]
        pixels = sample["pixels"]
        actions = sample["action"]
        gt_states = sample["state"].numpy()[:, :6]

        rollout = rollout_episode(
            model, state_head, pixels, actions, history_size=ctx_len, device=device,
            act_mean=act_mean, act_std=act_std, z_slice=z_slice,
        )
        rollout["actual_states"] = gt_states
        rollout["actions"] = actions.numpy()

        # Load RGB frames if requested
        if ds_rgb is not None:
            rgb_sample = ds_rgb[int(idx)]
            # Denormalize ImageNet preprocessing → uint8
            rgb_tensor = rgb_sample["pixels"].numpy()  # (T, C, H, W)
            rgb_denorm = np.clip((rgb_tensor * std + mean) * 255, 0, 255).astype(
                np.uint8
            )
            # (T, C, H, W) → (T, H, W, C)
            rollout["rgb_frames"] = rgb_denorm.transpose(0, 2, 3, 1)

        ep_idx, local_start = ds.clip_indices[int(idx)]
        results.append(rollout)
        print(
            f"  ep {ep_idx} start {local_start}: {len(pixels)} steps, "
            f"final z-MSE={np.mean((rollout['z_pred'][-1] - rollout['z_gt'][-1])**2):.4f}"
        )

    return results
