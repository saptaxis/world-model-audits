#!/usr/bin/env python3
"""Train state head probe on LeWorldModel latents.

Encodes frames (or loads cached z's), trains MLP/linear probe, reports R²
per dim. Supports three z sources:
  - encoder z (default): pass `--dataset` only.
  - predictor z, legacy single-position single-action mode: `--predicted-z`.
  - predictor z, training-aligned mode (recommended): `--training-aligned`.
    Mirrors train.py::lejepa_forward windowing exactly: T=ctx_len+n_preds
    frames per window, ctx_len predicted positions emitted, real per-position
    actions. Requires `--predicted-z`.

Action normalization (`--normalize-actions`):
    Training intent is to z-score actions via train.py:120's
    get_column_normalizer. For SINGLE-dataset training configs this fires
    correctly. For MULTI-dataset configs it silently does NOT fire (the
    transform set on swm.data.ConcatDataset is never read — see
    e5-05-action-norm-bug-and-invalidations-Apr222026.md). So whether to
    normalize here depends on what the trained checkpoint actually saw:
      - heur-only / any single-dataset training: --normalize-actions ON
      - v1/v2 / any multi-dataset training before the bug fix: OFF
      - any multi-dataset training AFTER the bug fix: ON

Sliced probes (`--z-dims START:END`):
    Slice z to a specific block before probing. Use 0:16 for v2's dedicated
    z_kin block. Cache stores the full z; slicing happens post-load so
    multiple slice runs reuse one cache.

Cache filenames distinguish modes so different settings cannot silently
reuse stale caches:
  - encoder mode: encoded_z[_<ds>].npz
  - predicted-z legacy: predicted_z[_<ds>].npz
  - predicted-z aligned: predicted_z_aligned_ctx{C}_np{P}_norm{Raw,Z}[_<ds>].npz

Usage patterns
--------------
Encoder z, multiple datasets:
    python lewm/scripts/train_state_head.py \\
        --model /path/to/lewm_object.ckpt \\
        --dataset lunarlander_synthetic_heuristic_clean ... \\
        --output-dir /path/to/state_head/ \\
        --max-frames-per-dataset 5000

E5-03 heur-only ep30 (single-dataset, normalization fired in training):
    python lewm/scripts/train_state_head.py \\
        --model /.../synthetic-heuristic-fs10/lewm_..._object.ckpt \\
        --dataset lunarlander_synthetic_heuristic \\
        --output-dir /.../state_head_aligned \\
        --predicted-z --training-aligned --ctx-len 3 --n-preds 1 \\
        --normalize-actions --action-norm-ref lunarlander_synthetic_heuristic \\
        --max-frames-per-dataset 75000 --val-split 0.25

v1 broken-regime multi-dataset (synthetic-all-clean-fs10-aux, num_preds=1):
    python lewm/scripts/train_state_head.py \\
        --model /.../synthetic-all-clean-fs10-aux/lewm_..._object.ckpt \\
        --dataset <12 _clean datasets> \\
        --output-dir /.../state_head_aligned \\
        --predicted-z --training-aligned --ctx-len 3 --n-preds 1 \\
        --max-frames-per-dataset 75000 --val-split 0.25

v2 broken-regime z_kin probe (synthetic-all-clean-fs10-zkin16-np2):
    python lewm/scripts/train_state_head.py \\
        --model /.../synthetic-all-clean-fs10-zkin16-np2/lewm_..._object.ckpt \\
        --dataset <12 _clean datasets> \\
        --output-dir /.../state_head_aligned \\
        --predicted-z --training-aligned --ctx-len 3 --n-preds 2 \\
        --z-dims 0:16 \\
        --max-frames-per-dataset 75000 --val-split 0.25

Future post-fix multi-dataset network (any num_preds, any kin_block):
    python lewm/scripts/train_state_head.py \\
        --model /.../<post-fix-ckpt>_object.ckpt \\
        --dataset <list> \\
        --output-dir /.../state_head_aligned \\
        --predicted-z --training-aligned --ctx-len 3 --n-preds <P> \\
        --normalize-actions \\
        [--z-dims 0:<kin_block>] \\
        --max-frames-per-dataset 75000 --val-split 0.25
"""

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.encode_dataset import encode_dataset
from eval.state_head import train_state_head, save_state_head, KIN_DIM_NAMES
from eval.action_norm import compute_action_normalizer


def encode_predicted(
    model_path: str,
    dataset_name: str,
    cache_dir: str,
    output_path: str,
    device: str = "cuda",
    batch_size: int = 256,
    max_frames: int = 0,
    seed: int = 42,
    read_chunk_size: int = 10000,
):
    """LEGACY: predicted-z cache, single-position-output + broadcasted-action.

    Kept for backward compat with prior probe results (and as a "this is what
    the broken-pipeline numbers looked like" reference). For all new probes
    use encode_predicted_aligned() via the --training-aligned flag.

    For each transition: encode 3-frame history → run predictor with the action
    AT the last history step, broadcast across all 3 history positions →
    take only the last predicted z. Pair with GT state at t+HS. This pairing
    DOES NOT match training's aux-loss windowing (see encode_predicted_aligned).

    Two-pass structure mirroring eval/encode_dataset.py::encode_dataset so
    encoder work amortizes across many episodes per HDF5 read + GPU launch
    (critical for short-episode datasets where per-episode overhead dominates):

      Pass 1 (encode): accumulate adjacent selected-episode slices into
        contiguous HDF5 reads up to ~read_chunk_size raw frames. Each chunk
        is one big blosc decompress + one strided numpy view to get the
        output-step (fs-stride) frames, then batched encoder forwards of
        batch_size frames at a time. Per-episode z's are cached in a dict
        keyed by ep_idx.

      Pass 2 (predict): for each episode, index the cached z's to form
        (B, HS, D) history windows, run predictor in batches of batch_size
        transitions, record the (z_pred, state_{t+HS}) pairs.

    This makes throughput independent of episode length: a 4-frame episode
    no longer triggers its own GPU launch.

    Output npz has same (z, state) keys as encode_dataset for compatibility.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    import torch
    from tqdm import tqdm

    LEWM_DIR = Path(__file__).resolve().parent.parent / "vendor" / "le-wm"
    sys.path.insert(0, str(LEWM_DIR))
    import stable_worldmodel as swm

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    device_t = torch.device(device)
    model = torch.load(model_path, map_location=device_t, weights_only=False)
    model.eval()

    HS = 3  # history size
    fs = 10  # frameskip (hardcoded to match training)

    datasets_dir = swm.data.utils.get_cache_dir(cache_dir, sub_folder="datasets")
    h5_path = Path(datasets_dir) / f"{dataset_name}.h5"

    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        n_episodes = len(ep_len)

        # Count valid transitions and select episodes
        valid_eps = []
        total_trans = 0
        for ep in range(n_episodes):
            raw_len = int(ep_len[ep])
            n_output = raw_len // fs
            n_trans_ep = max(0, n_output - HS)
            if n_trans_ep > 0:
                valid_eps.append((ep, n_trans_ep))
                total_trans += n_trans_ep

        rng = np.random.default_rng(seed)
        rng.shuffle(valid_eps)

        if max_frames > 0:
            selected = []
            count = 0
            for ep, n_t in valid_eps:
                selected.append((ep, n_t))
                count += n_t
                if count >= max_frames:
                    break
            n_trans = min(count, max_frames)
        else:
            selected = valid_eps
            n_trans = total_trans
        # Sort by offset so adjacent episodes merge into contiguous HDF5 reads.
        selected.sort(key=lambda x: int(ep_offset[x[0]]))
        print(f"  {n_trans} transitions from {len(selected)} episodes")

        # ---------- Pass 1: batched encode across episodes ----------

        # Per-episode cached arrays we'll need in pass 2.
        ep_z: dict[int, np.ndarray] = {}          # ep_idx -> (n_output, D) float32 on CPU
        ep_state: dict[int, np.ndarray] = {}      # ep_idx -> (n_output, state_dim) float32
        ep_action: dict[int, np.ndarray] = {}     # ep_idx -> (raw_len, 2) float32

        def encode_chunk(chunk_start: int, chunk_end: int, members: list[tuple[int, int, int]]):
            """Read one contiguous HDF5 range, encode all output-step frames in it.

            members: list of (ep_idx, ep_base_within_chunk, raw_len) describing
            which episode each region of the chunk belongs to.
            """
            pixels_raw = f["pixels"][chunk_start:chunk_end]   # uint8, one big blosc decompress
            states_raw = f["state"][chunk_start:chunk_end]
            actions_raw = f["action"][chunk_start:chunk_end]
            # Gather only the output-step pixels (stride fs within each episode).
            pix_list = []
            per_ep_counts = []
            for ep_idx, ep_base, raw_len in members:
                n_output = raw_len // fs
                ep_pix = pixels_raw[ep_base:ep_base + raw_len:fs][:n_output]
                ep_sta = states_raw[ep_base:ep_base + raw_len:fs][:n_output]
                ep_act = actions_raw[ep_base:ep_base + raw_len]
                ep_state[ep_idx] = np.asarray(ep_sta, dtype=np.float32)
                ep_action[ep_idx] = np.asarray(ep_act, dtype=np.float32)
                pix_list.append(ep_pix)
                per_ep_counts.append(n_output)
            if not pix_list:
                return
            all_pix = np.concatenate(pix_list, axis=0)        # (sum_n_output, H, W, 3) uint8

            # Batched encoder forwards.
            z_parts = []
            for s in range(0, all_pix.shape[0], batch_size):
                e = min(s + batch_size, all_pix.shape[0])
                p = torch.from_numpy(all_pix[s:e]).float().permute(0, 3, 1, 2) / 255.0
                p = (p - MEAN) / STD
                p = p.to(device_t, non_blocking=True)
                with torch.no_grad():
                    out = model.encoder(p, interpolate_pos_encoding=True)
                    z = model.projector(out.last_hidden_state[:, 0])
                z_parts.append(z.cpu().numpy())
            all_z = np.concatenate(z_parts, axis=0)           # (sum_n_output, D) float32

            # Split back to per-episode z arrays.
            offset = 0
            for (ep_idx, _, _), n_out in zip(members, per_ep_counts):
                ep_z[ep_idx] = all_z[offset:offset + n_out]
                offset += n_out

        # Merge adjacent selected episodes into contiguous read chunks.
        chunk_start = int(ep_offset[selected[0][0]])
        chunk_end = chunk_start + int(ep_len[selected[0][0]])
        members = [(selected[0][0], 0, int(ep_len[selected[0][0]]))]

        encode_pbar = tqdm(total=len(selected), desc="encoding", unit="ep")
        for ep_idx, _ in selected[1:]:
            ep_s = int(ep_offset[ep_idx])
            ep_l = int(ep_len[ep_idx])
            if ep_s == chunk_end and (chunk_end - chunk_start + ep_l) <= read_chunk_size:
                members.append((ep_idx, chunk_end - chunk_start, ep_l))
                chunk_end = ep_s + ep_l
            else:
                encode_chunk(chunk_start, chunk_end, members)
                encode_pbar.update(len(members))
                chunk_start = ep_s
                chunk_end = ep_s + ep_l
                members = [(ep_idx, 0, ep_l)]
        encode_chunk(chunk_start, chunk_end, members)
        encode_pbar.update(len(members))
        encode_pbar.close()

        # ---------- Pass 2: transition assembly + predictor forward ----------

        all_z_parts = []
        all_state_parts = []

        pbar = tqdm(total=n_trans, desc="predicting z", unit="trans")
        for ep_idx, n_trans_ep in selected:
            z_cache = ep_z[ep_idx]               # (n_output, D)
            states = ep_state[ep_idx]            # (n_output, state_dim)
            actions = ep_action[ep_idx]          # (raw_len, 2)

            z_cache_t = torch.from_numpy(z_cache).to(device_t)

            for t_start in range(0, n_trans_ep, batch_size):
                t_end = min(t_start + batch_size, n_trans_ep)
                B = t_end - t_start
                idx = np.arange(t_start, t_end)

                hist_idx = torch.from_numpy(
                    np.stack([idx + h for h in range(HS)], axis=1)
                ).to(device_t)
                z_history = z_cache_t[hist_idx]    # (B, HS, D)

                act_row = idx + (HS - 1)
                actions_batch = np.zeros((B, fs * 2), dtype=np.float32)
                for i, t in enumerate(act_row):
                    a_s = int(t) * fs
                    a_e = min(a_s + fs, len(actions))
                    raw = actions[a_s:a_e]
                    if len(raw) < fs:
                        raw = np.concatenate([raw, np.zeros((fs - len(raw), 2), dtype=np.float32)])
                    actions_batch[i] = raw.flatten()
                states_next = states[idx + HS]

                act_t = torch.from_numpy(actions_batch).to(device_t, non_blocking=True)
                act_expanded = act_t.unsqueeze(1).expand(B, HS, -1)
                with torch.no_grad():
                    act_emb = model.action_encoder(act_expanded)
                    pred = model.predict(z_history, act_emb)
                    z_pred = pred[:, -1]

                all_z_parts.append(z_pred.cpu().numpy())
                all_state_parts.append(states_next)
                pbar.update(B)
        pbar.close()

        all_z = np.concatenate(all_z_parts, axis=0)[:n_trans]
        all_state = np.concatenate(all_state_parts, axis=0)[:n_trans]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, z=all_z, state=all_state)
    print(f"Saved {n_trans} predicted z's to {output_path}")
    print(f"  z: {all_z.shape}, state: {all_state.shape}")


def encode_predicted_aligned(
    model_path: str,
    dataset_name: str,
    cache_dir: str,
    output_path: str,
    ctx_len: int = 3,
    n_preds: int = 2,
    normalize_actions: bool = False,
    action_norm_ref: str = "lunarlander_synthetic_heuristic_clean",
    device: str = "cuda",
    batch_size: int = 256,
    max_frames: int = 0,
    seed: int = 42,
    read_chunk_size: int = 10000,
):
    """Training-aligned predicted-z cache.

    Mirrors train.py::lejepa_forward exactly for the aux-kinematic-loss
    pipeline, so offline probes computed from this cache are directly
    comparable to the co-trained state_head's validate/aux_kin_loss.

    Training does:
        ctx_emb = emb[:, :ctx_len]               # (B, ctx_len, D)
        ctx_act = act_emb[:, :ctx_len]           # (B, ctx_len, A_emb)
        pred_emb = model.predict(ctx_emb, ctx_act)   # (B, ctx_len, D)
        aux_target = state[:, n_preds:, :6]      # (B, ctx_len, 6)

    So each window (num_steps = ctx_len + n_preds output-step frames) emits
    ctx_len (pred_emb[i], state[i + n_preds]) pairs. With ctx_len=3, n_preds=2:
        pred[0] <-> state[t+2]
        pred[1] <-> state[t+3]
        pred[2] <-> state[t+4]
    ARPredictor is causal, so pred[i] is conditioned on ctx_emb[:i+1] and
    ctx_act[:i+1] (i.e. frames t..t+i).

    CRITICAL (vs legacy encode_predicted):
      - Actions are REAL per-position (not broadcast): at context position i,
        the action input is the raw fs=10 action frames starting at raw
        index (t + i) * fs, flattened to fs * action_dim.
      - ALL ctx_len predicted positions are emitted (not just pred[-1]).

    All indices `t`, `t + i`, `t + n_preds + i` are output-step indices
    (post-frameskip). Raw-frame indices only appear inside the action-slice
    expression `actions_raw[(t+i)*fs : (t+i+1)*fs]`.

    max_frames caps the total number of emitted (pred, state) pairs.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    import torch
    from tqdm import tqdm

    LEWM_DIR = Path(__file__).resolve().parent.parent / "vendor" / "le-wm"
    sys.path.insert(0, str(LEWM_DIR))
    import stable_worldmodel as swm

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    device_t = torch.device(device)
    model = torch.load(model_path, map_location=device_t, weights_only=False)
    model.eval()

    fs = 10  # frameskip (hardcoded to match training)
    num_steps = ctx_len + n_preds   # output-step window length
    emit_per_window = ctx_len        # one pair per context position

    print(f"  predicted_z mode: training_aligned ctx={ctx_len} np={n_preds} "
          f"(num_steps={num_steps}, emit_per_window={emit_per_window})")

    # Action normalization is controlled by `normalize_actions`.
    # Background: train.py:120 sets up an action z-score normalizer via
    # get_column_normalizer(ref_dataset, 'action', 'action'). For SINGLE-dataset
    # training, transform is set on HDF5Dataset directly and the normalizer
    # fires correctly. For MULTI-dataset training, transform is set on
    # swm.data.ConcatDataset whose __getitem__ does NOT invoke transform — so
    # the normalizer silently never runs (see e5-05). Whether to normalize here
    # therefore depends on what the trained action_encoder actually saw at
    # training time, which the caller must specify.
    if normalize_actions:
        act_mean, act_std = compute_action_normalizer(action_norm_ref, cache_dir,
                                                      ctx_len, n_preds)
        print(f"  action normalizer ON (ref={action_norm_ref}): "
              f"mean={act_mean.tolist()} std={act_std.tolist()}")
    else:
        act_mean = act_std = None
        print(f"  action normalizer OFF (raw actions fed to action_encoder)")

    datasets_dir = swm.data.utils.get_cache_dir(cache_dir, sub_folder="datasets")
    h5_path = Path(datasets_dir) / f"{dataset_name}.h5"

    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        n_episodes = len(ep_len)

        # Count valid windows per episode in output-step units.
        # A window at output-step t uses frames t..t+ctx_len-1 for context and
        # state targets at t+n_preds..t+n_preds+ctx_len-1, so the last valid
        # t satisfies t + num_steps - 1 < n_output, i.e. t <= n_output - num_steps.
        valid_eps = []
        total_pairs = 0
        for ep in range(n_episodes):
            raw_len = int(ep_len[ep])
            n_output = raw_len // fs
            n_windows_ep = max(0, n_output - num_steps + 1)
            n_pairs_ep = n_windows_ep * emit_per_window
            if n_windows_ep > 0:
                valid_eps.append((ep, n_windows_ep))
                total_pairs += n_pairs_ep

        rng = np.random.default_rng(seed)
        rng.shuffle(valid_eps)

        if max_frames > 0:
            selected = []
            count = 0
            for ep, n_w in valid_eps:
                selected.append((ep, n_w))
                count += n_w * emit_per_window
                if count >= max_frames:
                    break
            n_pairs = min(count, max_frames)
        else:
            selected = valid_eps
            n_pairs = total_pairs
        # Sort by offset so adjacent episodes merge into contiguous HDF5 reads.
        selected.sort(key=lambda x: int(ep_offset[x[0]]))
        print(f"  {n_pairs} aligned pairs from {len(selected)} episodes "
              f"({sum(n_w for _, n_w in selected)} windows)")

        # ---------- Pass 1: batched encode across episodes ----------

        ep_z: dict[int, np.ndarray] = {}          # ep_idx -> (n_output, D)
        ep_state: dict[int, np.ndarray] = {}      # ep_idx -> (n_output, state_dim)
        ep_action: dict[int, np.ndarray] = {}     # ep_idx -> (raw_len, 2)

        def encode_chunk(chunk_start, chunk_end, members):
            pixels_raw = f["pixels"][chunk_start:chunk_end]
            states_raw = f["state"][chunk_start:chunk_end]
            actions_raw = f["action"][chunk_start:chunk_end]
            pix_list = []
            per_ep_counts = []
            for ep_idx, ep_base, raw_len in members:
                n_output = raw_len // fs
                ep_pix = pixels_raw[ep_base:ep_base + raw_len:fs][:n_output]
                ep_sta = states_raw[ep_base:ep_base + raw_len:fs][:n_output]
                ep_act = actions_raw[ep_base:ep_base + raw_len]
                ep_state[ep_idx] = np.asarray(ep_sta, dtype=np.float32)
                ep_action[ep_idx] = np.asarray(ep_act, dtype=np.float32)
                pix_list.append(ep_pix)
                per_ep_counts.append(n_output)
            if not pix_list:
                return
            all_pix = np.concatenate(pix_list, axis=0)

            z_parts = []
            for s in range(0, all_pix.shape[0], batch_size):
                e = min(s + batch_size, all_pix.shape[0])
                p = torch.from_numpy(all_pix[s:e]).float().permute(0, 3, 1, 2) / 255.0
                p = (p - MEAN) / STD
                p = p.to(device_t, non_blocking=True)
                with torch.no_grad():
                    out = model.encoder(p, interpolate_pos_encoding=True)
                    z = model.projector(out.last_hidden_state[:, 0])
                z_parts.append(z.cpu().numpy())
            all_z = np.concatenate(z_parts, axis=0)

            offset = 0
            for (ep_idx, _, _), n_out in zip(members, per_ep_counts):
                ep_z[ep_idx] = all_z[offset:offset + n_out]
                offset += n_out

        chunk_start = int(ep_offset[selected[0][0]])
        chunk_end = chunk_start + int(ep_len[selected[0][0]])
        members = [(selected[0][0], 0, int(ep_len[selected[0][0]]))]

        encode_pbar = tqdm(total=len(selected), desc="encoding", unit="ep")
        for ep_idx, _ in selected[1:]:
            ep_s = int(ep_offset[ep_idx])
            ep_l = int(ep_len[ep_idx])
            if ep_s == chunk_end and (chunk_end - chunk_start + ep_l) <= read_chunk_size:
                members.append((ep_idx, chunk_end - chunk_start, ep_l))
                chunk_end = ep_s + ep_l
            else:
                encode_chunk(chunk_start, chunk_end, members)
                encode_pbar.update(len(members))
                chunk_start = ep_s
                chunk_end = ep_s + ep_l
                members = [(ep_idx, 0, ep_l)]
        encode_chunk(chunk_start, chunk_end, members)
        encode_pbar.update(len(members))
        encode_pbar.close()

        # ---------- Pass 2: aligned window assembly + predictor forward ----------

        all_z_parts = []
        all_state_parts = []

        pbar = tqdm(total=n_pairs, desc="predicting z (aligned)", unit="pair")
        for ep_idx, n_windows_ep in selected:
            z_cache = ep_z[ep_idx]         # (n_output, D)  output-step
            states = ep_state[ep_idx]      # (n_output, state_dim)  output-step
            actions = ep_action[ep_idx]    # (raw_len, 2)  raw-frame

            z_cache_t = torch.from_numpy(z_cache).to(device_t)

            # Batch over window starts t in output-step units.
            for w_start in range(0, n_windows_ep, batch_size):
                w_end = min(w_start + batch_size, n_windows_ep)
                B = w_end - w_start
                t_idx = np.arange(w_start, w_end)          # (B,) output-step start indices

                # Context z: (B, ctx_len, D)
                ctx_idx = torch.from_numpy(
                    np.stack([t_idx + i for i in range(ctx_len)], axis=1)
                ).to(device_t)
                ctx_z = z_cache_t[ctx_idx]

                # Per-position actions: for each window, for each ctx position i,
                # take raw actions[(t+i)*fs : (t+i+1)*fs] (z-score per raw-action
                # dim if normalize_actions, else raw), then flatten to (fs*act_dim,).
                act_dim = actions.shape[1]
                actions_batch = np.zeros((B, ctx_len, fs * act_dim), dtype=np.float32)
                for bi, t in enumerate(t_idx):
                    for i in range(ctx_len):
                        a_s = int((t + i) * fs)
                        a_e = min(a_s + fs, len(actions))
                        raw = actions[a_s:a_e]
                        if len(raw) < fs:
                            raw = np.concatenate(
                                [raw, np.zeros((fs - len(raw), act_dim), dtype=np.float32)]
                            )
                        if act_mean is not None:
                            raw = (raw - act_mean) / act_std
                        actions_batch[bi, i] = raw.flatten()

                # State targets: (B, ctx_len, state_dim), at output-step t+n_preds+i.
                state_targets = np.stack(
                    [states[t_idx + n_preds + i] for i in range(ctx_len)], axis=1
                )

                act_t = torch.from_numpy(actions_batch).to(device_t, non_blocking=True)
                with torch.no_grad():
                    act_emb = model.action_encoder(act_t)      # (B, ctx_len, A_emb)
                    pred = model.predict(ctx_z, act_emb)        # (B, ctx_len, D)

                # Flatten (B, ctx_len, ...) -> (B*ctx_len, ...) to emit all pairs.
                all_z_parts.append(pred.reshape(-1, pred.shape[-1]).cpu().numpy())
                all_state_parts.append(state_targets.reshape(-1, state_targets.shape[-1]))
                pbar.update(B * emit_per_window)
        pbar.close()

        all_z = np.concatenate(all_z_parts, axis=0)[:n_pairs]
        all_state = np.concatenate(all_state_parts, axis=0)[:n_pairs]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, z=all_z, state=all_state)
    print(f"Saved {n_pairs} aligned predicted z's to {output_path}")
    print(f"  z: {all_z.shape}, state: {all_state.shape}")


def resolve_max_frames(datasets, max_frames, max_frames_per_dataset, cache_dir):
    """Return per-dataset max_frames list.

    Scenarios:
        - Both flags set: error
        - --max-frames-per-dataset N: N per dataset
        - --max-frames N (single dataset): N for that dataset (backward compat)
        - --max-frames N (multi-dataset): N total, distributed proportionally
          by dataset size (number of frames in each HDF5)
        - Neither flag: 0 (all frames) per dataset
    """
    n = len(datasets)
    if max_frames is not None and max_frames_per_dataset is not None:
        raise ValueError("Cannot specify both --max-frames and --max-frames-per-dataset")
    if max_frames_per_dataset is not None:
        return [max_frames_per_dataset] * n
    if max_frames is not None:
        if n == 1:
            return [max_frames]
        # Distribute proportionally by dataset size
        import h5py
        import hdf5plugin  # noqa: F401
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "le-wm"))
        import stable_worldmodel as swm
        datasets_dir = swm.data.utils.get_cache_dir(cache_dir, sub_folder="datasets")
        sizes = []
        for ds_name in datasets:
            h5_path = Path(datasets_dir) / f"{ds_name}.h5"
            with h5py.File(h5_path, "r") as f:
                sizes.append(int(f["ep_len"][:].sum()))
        total = sum(sizes)
        per_ds = [min(s, int(max_frames * s / total)) for s in sizes]
        print(f"Distributing {max_frames} frames proportionally across {n} datasets:")
        for ds_name, s, alloc in zip(datasets, sizes, per_ds):
            capped = " (all)" if alloc >= s else ""
            print(f"  {ds_name}: {s} total -> {alloc} sampled{capped}")
        return per_ds
    return [0] * n  # 0 = all frames


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="LeWorldModel _object.ckpt path")
    parser.add_argument("--dataset", nargs="+", default=["lunarlander_heuristic"],
                        help="One or more HDF5 dataset names")
    parser.add_argument("--cache-dir", default="/media/hdd1/physics-priors-latent-space/lunar-lander-data")
    parser.add_argument("--output-dir", required=True, help="Where to save state head + results")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Total max frames across all datasets (split evenly for multi-dataset). 0=all.")
    parser.add_argument("--max-frames-per-dataset", type=int, default=None,
                        help="Max frames per dataset (multi-dataset only). 0=all.")
    parser.add_argument("--encode-batch-size", type=int, default=1024, help="Batch size for ViT encoding")
    parser.add_argument("--read-chunk-size", type=int, default=10000,
                        help="Frames to read from HDF5 at a time (controls memory usage)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction held out for validation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--linear", action="store_true",
                        help="Use linear probe instead of MLP")
    parser.add_argument("--predicted-z", action="store_true",
                        help="Train on predictor z's (action-conditioned) instead of encoder z's")
    parser.add_argument("--training-aligned", action="store_true",
                        help="Use training-aligned predicted-z pipeline (mirrors train.py aux-loss "
                             "windowing: real per-position actions, all ctx_len output positions). "
                             "Implies --predicted-z. Writes to a separate cache file so it does not "
                             "overwrite legacy predicted_z.npz.")
    parser.add_argument("--ctx-len", type=int, default=3,
                        help="history_size / context length (must match training config).")
    parser.add_argument("--n-preds", type=int, default=1,
                        help="num_preds / prediction horizon (must match training config). "
                             "Default 1. Pass 2 for v2-style multi-step-target checkpoints.")
    parser.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Apply z-score normalizer to actions before action_encoder. "
                             "Default ON. Pass --no-normalize-actions for broken-regime "
                             "multi-dataset checkpoints (pre ConcatDataset transform-"
                             "propagation fix; see e5-05).")
    parser.add_argument("--action-norm-ref", default="lunarlander_synthetic_heuristic_clean",
                        help="Reference dataset for reproducing training's action normalizer "
                             "(must be the FIRST dataset in the training config's data.dataset.name "
                             "list). Only consulted when --normalize-actions is set.")
    parser.add_argument("--z-dims", default=None,
                        help="Slice of z to feed probe, as START:END (e.g. '0:16' for dedicated z_kin). "
                             "Default: use full z.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    datasets = args.dataset
    per_ds_limits = resolve_max_frames(datasets, args.max_frames, args.max_frames_per_dataset,
                                       args.cache_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --training-aligned implies --predicted-z. Cache filename encodes the mode,
    # (ctx_len, n_preds), AND the normalize-actions setting so a config change
    # never silently reuses a stale cache from a different pipeline or different
    # action-input statistics.
    if args.training_aligned and not args.predicted_z:
        args.predicted_z = True
    aligned = args.training_aligned
    if aligned:
        norm_tag = "Z" if args.normalize_actions else "Raw"
        cache_stem = f"predicted_z_aligned_ctx{args.ctx_len}_np{args.n_preds}_norm{norm_tag}"
        print(f"predicted_z mode: training_aligned ctx={args.ctx_len} np={args.n_preds} "
              f"normalize_actions={args.normalize_actions}")
    else:
        cache_stem = "predicted_z"
        if args.predicted_z:
            print("predicted_z mode: legacy_broadcast (single-position output, "
                  "action broadcast across history)")
            if args.normalize_actions:
                print("  WARNING: --normalize-actions has no effect in legacy mode.")

    if args.predicted_z:
        # Predicted-z mode: use predictor(z_history, action) → z_{t+k}
        z_cache = output_dir / f"{cache_stem}.npz"
        if z_cache.exists():
            print(f"Loading cached predicted embeddings from {z_cache}")
            data = np.load(z_cache)
            z_all, state_all = data["z"], data["state"]
        else:
            # For predicted-z, max_frames applies to total emitted pairs.
            total_limit = args.max_frames if args.max_frames is not None else 0
            if args.max_frames_per_dataset is not None:
                total_limit = args.max_frames_per_dataset * len(datasets)
            z_parts, state_parts = [], []
            for ds_name in datasets:
                ds_cache = output_dir / f"{cache_stem}_{ds_name}.npz"
                ds_limit = total_limit // len(datasets) if total_limit > 0 else 0
                if ds_cache.exists():
                    print(f"\n--- Reusing cached predicted z for {ds_name}: {ds_cache.name} ---")
                else:
                    print(f"\n--- Predicting z for {ds_name} (max_frames={ds_limit or 'all'}) ---")
                    if aligned:
                        encode_predicted_aligned(
                            model_path=args.model,
                            dataset_name=ds_name,
                            cache_dir=args.cache_dir,
                            output_path=str(ds_cache),
                            ctx_len=args.ctx_len,
                            n_preds=args.n_preds,
                            normalize_actions=args.normalize_actions,
                            action_norm_ref=args.action_norm_ref,
                            device=args.device,
                            batch_size=args.encode_batch_size,
                            max_frames=ds_limit,
                            read_chunk_size=args.read_chunk_size,
                        )
                    else:
                        encode_predicted(
                            model_path=args.model,
                            dataset_name=ds_name,
                            cache_dir=args.cache_dir,
                            output_path=str(ds_cache),
                            device=args.device,
                            batch_size=args.encode_batch_size,
                            max_frames=ds_limit,
                            read_chunk_size=args.read_chunk_size,
                        )
                data = np.load(ds_cache)
                z_parts.append(data["z"])
                state_parts.append(data["state"])
                print(f"  {ds_name}: {data['z'].shape[0]} pairs")
            z_all = np.concatenate(z_parts, axis=0)
            state_all = np.concatenate(state_parts, axis=0)
            np.savez(z_cache, z=z_all, state=state_all)
            print(f"\nCombined: {z_all.shape[0]} predicted z's -> {z_cache}")
    else:
        # Standard encoder-z mode
        z_cache = output_dir / "encoded_z.npz"
        if z_cache.exists():
            print(f"Loading cached embeddings from {z_cache}")
            data = np.load(z_cache)
            z_all, state_all = data["z"], data["state"]
        else:
            z_parts, state_parts = [], []
            for ds_name, ds_limit in zip(datasets, per_ds_limits):
                ds_cache = output_dir / f"encoded_z_{ds_name}.npz"
                if ds_cache.exists():
                    print(f"\n--- Reusing cached encoded z for {ds_name}: {ds_cache.name} ---")
                else:
                    print(f"\n--- Encoding {ds_name} (max_frames={ds_limit or 'all'}) ---")
                    encode_dataset(
                        model_path=args.model,
                        dataset_name=ds_name,
                        cache_dir=args.cache_dir,
                        output_path=str(ds_cache),
                        device=args.device,
                        batch_size=args.encode_batch_size,
                        max_frames=ds_limit,
                        read_chunk_size=args.read_chunk_size,
                    )
                data = np.load(ds_cache)
                z_parts.append(data["z"])
                state_parts.append(data["state"])
                print(f"  {ds_name}: {data['z'].shape[0]} frames")

            z_all = np.concatenate(z_parts, axis=0)
            state_all = np.concatenate(state_parts, axis=0)
            np.savez(z_cache, z=z_all, state=state_all)
            print(f"\nCombined: {z_all.shape[0]} frames from {len(datasets)} datasets -> {z_cache}")

    # Optional z-slice (e.g. --z-dims 0:16 to probe the dedicated z_kin block only).
    # Caches above store the full z; slicing happens here so switching slices does
    # not trigger a re-encode.
    z_slice = None
    if args.z_dims is not None:
        parts = args.z_dims.split(":")
        if len(parts) != 2:
            raise ValueError(f"--z-dims must be START:END, got {args.z_dims!r}")
        z_slice = (int(parts[0]), int(parts[1]))
        print(f"Slicing z to dims [{z_slice[0]}:{z_slice[1]}] (was {z_all.shape[1]}-dim)")
        z_all = z_all[:, z_slice[0]:z_slice[1]]

    # Use first 6 dims only (kinematics)
    state_kin = state_all[:, :6]

    n = len(z_all)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    split = int((1 - args.val_split) * n)
    z_train, z_val = z_all[perm[:split]], z_all[perm[split:]]
    s_train, s_val = state_kin[perm[:split]], state_kin[perm[split:]]
    print(f"Train: {len(z_train)}, Val: {len(z_val)}")

    # Step 2: Train
    probe_type = "linear" if args.linear else "MLP"
    print(f"Training state head ({probe_type})...")
    head, metrics = train_state_head(
        z_train, s_train, z_val, s_val,
        epochs=args.epochs,
        device=args.device,
        linear=args.linear,
    )

    # Step 3: Report
    print(f"\n=== State Head R² ({probe_type}) ===")
    for name in KIN_DIM_NAMES:
        r2 = metrics["r2_per_dim"][name]
        print(f"  {name:8s}: R² = {r2:.4f}")
    print(f"  {'mean':8s}: R² = {metrics['r2_mean']:.4f}")
    print(f"  val MSE: {metrics['val_mse']:.6f}")

    # Step 4: Save. Suffix encodes norm mode in aligned runs so probes from
    # different norm settings in the same output_dir don't overwrite each other.
    suffix = ""
    if aligned:
        suffix += "_norm" + ("Z" if args.normalize_actions else "Raw")
    if args.linear:
        suffix += "_linear"
    if z_slice is not None:
        suffix += f"_z{z_slice[0]}-{z_slice[1]}"
    save_state_head(head, metrics, str(output_dir / f"state_head{suffix}.pt"),
                    z_slice=z_slice)

    # Save report as text
    ds_str = ", ".join(datasets)
    report_path = output_dir / f"r2_report{suffix}.txt"
    with open(report_path, "w") as f:
        f.write(f"State Head R² Report ({probe_type})\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Datasets ({len(datasets)}): {ds_str}\n")
        f.write(f"Frames: {n}\n")
        if z_slice is not None:
            f.write(f"z-slice: [{z_slice[0]}:{z_slice[1]}]\n")
        f.write("\n")
        for name in KIN_DIM_NAMES:
            f.write(f"  {name:8s}: R² = {metrics['r2_per_dim'][name]:.4f}\n")
        f.write(f"\n  mean R²: {metrics['r2_mean']:.4f}\n")
        f.write(f"  val MSE: {metrics['val_mse']:.6f}\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
