# -----------------------------------------------------------------------------
# Gymnasium environments for economic models.
#
# The environment delegates economic logic to the model. The model defines:
#     state_dim
#     action_dim
#     controls_from_action()
#     transition()
#     sample_shocks()
#     sample_initial_state()
#
# State convention:
#     state = (m, x1, ..., xd)
#
# Observation convention:
#     obs = (log(m / m_ref), log(x1), ..., log(xd))
# -----------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ecomodrl.models.base_model import CassKoopmans, BrockMirman, BufferStock


EPS = 1e-8


def obs_from_state(state, model):
    """Map raw economic state to Gym observation.

    Arguments
    ----------
    state: Raw state with final dimension `model.state_dim`.
    model: EconomicModel instance.

    Returns
    -------
    obs: Gym observation with final dimension `model.state_dim`.
    """
    state = np.asarray(state, dtype=np.float64)

    m = np.maximum(state[..., 0], EPS)
    m_ref = max(float(model.m_ref), EPS)

    m_obs = np.log(m / m_ref)

    if model.exogenous_dim == 0:
        return m_obs[..., None].astype(np.float64)

    x = np.maximum(state[..., 1:], EPS)
    x_obs = np.log(x)

    return np.concatenate([m_obs[..., None], x_obs], axis=-1).astype(np.float64)


class BaseEconEnv(gym.Env, ABC):
    """Base Gymnasium environment for economic models."""

    MODEL_CLS = None

    def __init__(self, params=None, **model_kwargs):
        super().__init__()

        if self.MODEL_CLS is None:
            raise ValueError("MODEL_CLS must be set by subclasses.")

        self.model = self.MODEL_CLS(params=params, **model_kwargs)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.model.state_dim,), dtype=np.float64,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.action_dim,), dtype=np.float64,
        )

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.asarray(
            self.model.sample_initial_state(self.np_random),
            dtype=np.float64,
        )

        return obs_from_state(self.state, self.model), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)

        controls = self.model.controls_from_action(action, state=self.state)
        controls["consumption"] = np.maximum(controls["consumption"], EPS)

        reward = float(self.model.utility(controls["consumption"]))

        shocks = self.model.sample_shocks(self.np_random, ())
        self.state = np.asarray(
            self.model.transition(self.state, controls, shocks),
            dtype=np.float64,
        )

        terminated = False
        truncated = False

        info = {
            "consumption": float(np.asarray(controls["consumption"])),
        }

        if "share" in controls:
            info["share"] = float(np.asarray(controls["share"]))

        return (
            obs_from_state(self.state, self.model),
            reward,
            terminated,
            truncated,
            info,
        )


class CassKoopmansEnv(BaseEconEnv):
    MODEL_CLS = CassKoopmans


class BrockMirmanEnv(BaseEconEnv):
    MODEL_CLS = BrockMirman


class BufferStockEnv(BaseEconEnv):
    MODEL_CLS = BufferStock