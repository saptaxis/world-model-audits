"""Evaluate a trained DreamerV3 agent on LunarLander.

Uses r2dreamer's own ParallelEnv infrastructure and follows the same
eval loop pattern as OnlineTrainer.eval() — no manual tensor construction.

Usage:
    cd dreamerv3/vendor/r2dreamer
    python ../../../dreamerv3/scripts/eval_agent.py \
        --logdir /media/hdd1/.../dreamerv3/s42 \
        --episodes 100
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

# Add r2dreamer to path so its internal imports (envs, tools, dreamer) resolve
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "vendor" / "r2dreamer"))

from dreamer import Dreamer
from envs import make_envs
import tools


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DreamerV3 on LunarLander")
    parser.add_argument("--logdir", required=True, help="Training run logdir (contains latest.pt)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of parallel eval episodes")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)
    device = args.device
    print(f"Loading agent from {logdir}")

    # Load saved hydra config and override eval episode count.
    # eval_episode_num controls how many parallel eval envs make_envs creates.
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(config, False)
    config.env.eval_episode_num = args.episodes

    # Create envs through r2dreamer's factory — same wrapper chain as training.
    # ParallelEnv spawns subprocess workers per env, returns TensorDicts from step().
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    # Reconstruct agent and load checkpoint
    agent = Dreamer(config.model, obs_space, act_space).to(device)
    checkpoint = torch.load(logdir / "latest.pt", map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    # --- Eval loop (follows OnlineTrainer.eval() lines 27-98 exactly) ---
    print(f"Running {args.episodes} eval episodes...")
    envs = eval_envs
    # (B,) tensors — one per parallel env
    done = torch.ones(envs.env_num, dtype=torch.bool, device=device)
    once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=device)
    steps = torch.zeros(envs.env_num, dtype=torch.int32, device=device)
    returns = torch.zeros(envs.env_num, dtype=torch.float32, device=device)

    agent_state = agent.get_initial_state(envs.env_num)
    # (B, A) — initial action vector
    act = agent_state["prev_action"].clone()

    while not once_done.all():
        steps += ~done * ~once_done

        # Step envs on CPU (ParallelEnv builds pinned TensorDicts)
        act_cpu = act.detach().to("cpu")
        done_cpu = done.detach().to("cpu")
        trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)

        # Async transfer to GPU
        trans = trans_cpu.to(device, non_blocking=True)
        done = done_cpu.to(device)

        # Agent policy inference
        trans["action"] = act
        act, agent_state = agent.act(trans, agent_state, eval=True)

        # Accumulate returns (only for episodes still running)
        returns += trans["reward"][:, 0] * ~once_done
        once_done |= done

    # --- Report results ---
    rewards = returns.cpu().numpy()
    lengths = steps.cpu().numpy().astype(float)

    print(f"\nResults ({args.episodes} episodes):")
    print(f"  Mean reward:  {rewards.mean():.1f} +/- {rewards.std():.1f}")
    print(f"  Mean length:  {lengths.mean():.1f}")
    print(f"  Landing rate: {(rewards > 200).sum() / len(rewards) * 100:.0f}%")
    print(f"  Min/Max:      {rewards.min():.1f} / {rewards.max():.1f}")


if __name__ == "__main__":
    main()
