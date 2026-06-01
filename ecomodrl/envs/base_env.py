# -----------------------------------------------------------------------------
# Gymnasium envs compatible with SB3. 
# Each env wraps a model instance from models.py and implements the Gym API. 
# The model handles all the economic logic; the env just translates to/from the Gym interface.  
#
# Design (uniform across all three models):
#   * Action space:  Box(-1, +1, (1,)).  share = MIN_SHARE + (MAX_SHARE-MIN_SHARE)*(a+1)/2
#                    Linear map -> uniform sensitivity in share space.
#                    TODO: Boundaries [MIN_SHARE, MAX_SHARE] = [1e-3, 1-1e-3] keep
#                    u(c) finite (no -inf reward) AND let the optimal BS
#                    policy approach c/m -> 1 at low m as required.
#   * Observation:   model-relative, log-scaled, centered at the reference
#                    state.  
#                    TODO: No VecNormalize (running-stats drift would poison 
#                    off-policy replay buffers; PPO works fine on the fixed transform too).  
#                    obs is fixed, deterministic, dimensionless, and O(1) at the steady state.
#                       m-channel: log(max(m, EPS_M) / m_ref)   (centered at 0 at the SS)
#                       z-channel: log(z)                       (BM only; centered at 0 at z=1) 
#   * Reward:        u(c).  TransformReward in train.py clips at -100 to
#                    bound occasional outliers from saturated transitions.
#   * Episodes:      Truncated via TimeLimit in setup_envs.py, with max_episode_steps=cfg.max_episode_steps.
# -----------------------------------------------------------------------------

import numpy as np
import gymnasium as gym
from abc import abstractmethod
from gymnasium import spaces
from ecomodrl.benchmarks.base_model import CassKoopmans, BrockMirman, BufferStock

# TODO: This is not very elegant.
# Floor on m before taking log (BufferStock m_grid starts at 0).  Far enough
# below any realised m that no clipping bites in practice; keeps log() finite
# at the artificial m=0 grid edge.
EPS = 1e-6

# TODO: Understand m_target + Make gym environment independent from EGM solution
def _m_ref(model):
    """Reference cash-on-hand: deterministic SS for CK/BM, target m* for BS."""
    if hasattr(model, "m_target"):
        return float(model.m_target)
    return float(model.m_ref)


def obs_from_state(state, model):
    """Map raw economic state (m, [z, ...]) to the Gym observation.

    m-channel: log(max(m, EPS_M) / m_ref)   (centered at 0 at SS)
    z-channel: log(z)                       (BM only, centered at 0 at z=1)

    Used by both `_BaseEconEnv` (training) and `policies.RLPolicy.share` (eval).
    Both must apply the same transform; this single helper enforces that.
    Supports both single states (shape (D,)) and batched states (shape (..., D)).
    """
    state = np.asarray(state, dtype=np.float64)
    m_ref = _m_ref(model)
    m = np.maximum(state[..., 0], EPS)
    m_obs = np.log(m / m_ref)
    if model.STATE_DIM == 1:
        # Add a singleton last axis so the obs has the expected shape (..., 1)
        return np.asarray(m_obs, dtype=np.float32)[..., None]
    z_obs = np.log(np.maximum(state[..., 1], EPS))
    return np.stack([m_obs, z_obs], axis=-1).astype(np.float32)


class _BaseEconEnv(gym.Env):
    """Base class for the economic envs."""

    MODEL_CLS = None

    def __init__(self, params):
        super().__init__()
        self.model = self.MODEL_CLS(params)

        n_obs = self.model.STATE_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_state(self.np_random).astype(np.float32)
        return obs_from_state(self.state, self.model), {}

    def step(self, action):
        m = float(self.state[0])
        c = m * self.share_from_action(action).item()
        reward = float(self.model.utility(c))

        shocks = self.model.sample_shocks(self.np_random, ())
        self.state = self.model.transition(self.state, c, shocks).astype(np.float32)

        terminated, truncated = False, False # Truncation handled by TimeLimit wrapper in setup_envs.py
        return obs_from_state(self.state, self.model), reward, terminated, truncated, {}

    def share_from_action(self, action):
        """Linear map [-1, +1] -> [MIN_ACTION_SHARE, MAX_ACTION_SHARE]; clips outside.
        Works for scalars, 1-D arrays, and arbitrary batch shapes.
        """
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return self.model.MIN_ACTION_SHARE + 0.5 * (self.model.MAX_ACTION_SHARE - self.model.MIN_ACTION_SHARE) * (a + 1.0)

    @abstractmethod
    def initial_state(self, rng):
        """Sample a single starting state for an RL episode."""
        ...


class CassKoopmansEnv(_BaseEconEnv):
    MODEL_CLS = CassKoopmans

    def initial_state(self, rng):
        return np.array([rng.choice(self.model.m_grid)])


class BrockMirmanEnv(_BaseEconEnv):
    MODEL_CLS = BrockMirman

    def initial_state(self, rng):
        m0 = rng.choice(self.model.xi_grid)
        z0 = rng.choice(self.model.xi_grid)
        return np.array([m0, z0])


class BufferStockEnv(_BaseEconEnv):
    MODEL_CLS = BufferStock

    def initial_state(self, rng):
        return np.array([rng.choice(self.model.m_grid[1:])])