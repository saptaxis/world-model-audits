"""Lunar Lander env producing synthetic-triangle frames matching training data."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

import parametric_lunar_lander  # noqa: F401 — registers ParametricLunarLander-v0
from lewm.utils.synthetic_render import render_synthetic_frame


class LunarLanderSynthetic(gym.Env):
    """Wraps ParametricLunarLander-v0 with synthetic-frame rendering.

    The physics is unchanged — we just replace the default RGB render with
    a triangle-on-black frame matching what the JEPA was trained on.

    Observation is a dict: {"pixels": (224, 224, 3) uint8}.
    Info dict contains "state": the raw 6-dim kinematic state (x, y, vx, vy,
    angle, ang_vel) so downstream evaluation can report kinematic success.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, img_size: int = 224, triangle_radius: int = 35,
                 render_mode: str | None = "rgb_array"):
        self._inner = gym.make("ParametricLunarLander-v0")
        self.action_space = self._inner.action_space  # Box(-1, 1, (2,))
        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(
                low=0, high=255, shape=(img_size, img_size, 3), dtype=np.uint8
            ),
        })
        self.img_size = img_size
        self.triangle_radius = triangle_radius
        self.render_mode = render_mode
        self._last_raw_state: np.ndarray | None = None
        self._goal_state = None

    def _render_obs(self, raw_obs: np.ndarray) -> dict:
        """Convert raw 8-dim LL obs to pixels dict."""
        x, y, _vx, _vy, angle = raw_obs[0], raw_obs[1], raw_obs[2], raw_obs[3], raw_obs[4]
        frame = render_synthetic_frame(
            x=float(x), y=float(y), angle=float(angle),
            size=self.img_size, triangle_radius=self.triangle_radius,
        )
        return {"pixels": frame}

    def reset(self, *, seed=None, options=None):
        raw_obs, info = self._inner.reset(seed=seed, options=options)
        self._last_raw_state = raw_obs
        info = dict(info)
        info["state"] = raw_obs[:6].astype(np.float32).copy()
        return self._render_obs(raw_obs), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self._inner.step(action)
        self._last_raw_state = raw_obs
        info = dict(info)
        info["state"] = raw_obs[:6].astype(np.float32).copy()
        return self._render_obs(raw_obs), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self._last_raw_state is None:
            raise RuntimeError("render() called before reset()")
        return self._render_obs(self._last_raw_state)["pixels"]

    def close(self):
        self._inner.close()

    def _set_state(self, state_payload):
        """LeWM callable hook. Reaches a target state via replay-forward (not teleport).

        state_payload keys:
            actions_prefix: (K, 2) array of actions to step after fresh reset.
            reset_seed: int seed for the fresh reset (determinism).
        """
        prefix = np.asarray(state_payload["actions_prefix"], dtype=np.float32)
        seed = int(state_payload.get("reset_seed", 42))
        raw_obs, info = self._inner.reset(seed=seed)
        for a in prefix:
            raw_obs, _, terminated, truncated, info = self._inner.step(a)
            if terminated or truncated:
                raise RuntimeError(
                    "Episode terminated during replay-forward prefix. "
                    "Check the prefix length and reset seed."
                )
        self._last_raw_state = raw_obs

    def _set_goal_state(self, goal_state):
        """LeWM callable hook. Stores goal state for evaluation metrics."""
        self._goal_state = np.asarray(goal_state, dtype=np.float32) if goal_state is not None else None


register(
    id="LunarLanderSynthetic-v0",
    entry_point="lewm.env.lunarlander_env:LunarLanderSynthetic",
    max_episode_steps=1000,
)
