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

    Observation is a dict: {"state": (6,) float32} (matches swm envs like pusht
    which return `{"proprio"/"state": ...}` without pixels). The wrapper chain
    (MegaWrapper → AddPixelsWrapper) calls our `render()` to produce the
    pixels that get added to info separately.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    STATE_DIM = 15  # 6 kinematic + 2 leg contacts + 7 physics params, matches dataset

    def __init__(self, img_size: int = 224, triangle_radius: int = 35,
                 render_mode: str | None = "rgb_array"):
        self._inner = gym.make("ParametricLunarLander-v0")
        self.action_space = self._inner.action_space  # Box(-1, 1, (2,))
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.STATE_DIM,), dtype=np.float32
            ),
        })
        self.img_size = img_size
        self.triangle_radius = triangle_radius
        self.render_mode = render_mode
        self._last_raw_state: np.ndarray | None = None
        self._goal_state = None

    def _state_obs(self, raw_obs: np.ndarray) -> dict:
        """Return 15-dim state: 6 kinematic + 2 leg contacts + 7 physics params.

        ParametricLunarLander's 8-dim obs (kinematic+legs) is extended with the
        7 physics params to match the dataset's 15-dim state column, which is
        required by the scaler fitted from dataset stats.
        """
        raw8 = raw_obs[:8].astype(np.float32)  # kinematic + legs
        physics7 = np.array(self._get_physics_params(), dtype=np.float32)
        return {"state": np.concatenate([raw8, physics7])}

    def _get_physics_params(self) -> list[float]:
        """Read physics params from the inner env (matches dataset convention)."""
        inner = self._inner.unwrapped
        return [
            float(inner.gravity) if hasattr(inner, "gravity") else -10.0,
            float(inner.main_engine_power) if hasattr(inner, "main_engine_power") else 13.0,
            float(inner.side_engine_power) if hasattr(inner, "side_engine_power") else 0.6,
            float(getattr(inner, "lander_density", 5.0)),
            float(getattr(inner, "enable_wind", 0)),
            float(getattr(inner, "wind_power", 15.0)),
            float(getattr(inner, "turbulence_power", 1.5)),
        ]

    def reset(self, *, seed=None, options=None):
        raw_obs, info = self._inner.reset(seed=seed, options=options)
        self._last_raw_state = raw_obs
        return self._state_obs(raw_obs), dict(info)

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self._inner.step(action)
        self._last_raw_state = raw_obs
        return self._state_obs(raw_obs), float(reward), bool(terminated), bool(truncated), dict(info)

    def render(self):
        if self._last_raw_state is None:
            raise RuntimeError("render() called before reset()")
        raw = self._last_raw_state
        return render_synthetic_frame(
            x=float(raw[0]), y=float(raw[1]), angle=float(raw[4]),
            size=self.img_size, triangle_radius=self.triangle_radius,
        )

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
