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
# -----------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ecomodrl.models import CassKoopmans, BrockMirman, BufferStock, PortfolioChoice


EPS = 1e-8


class BaseEconEnv(gym.Env, ABC):
    """Base Gymnasium environment for economic models."""

    MODEL_CLS = None

    def __init__(self, params=None, **model_kwargs):
        super().__init__()

        if self.MODEL_CLS is None:
            raise ValueError("MODEL_CLS must be set by subclasses.")

        self.model = self.MODEL_CLS(params=params, **model_kwargs)
        self.beta_eff = float(self.model.beta_eff)

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

        return self.model.obs_from_state(self.state), {}

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
            
        if "portfolio_weights" in controls:
            info["portfolio_weights"] = np.asarray(
                controls["portfolio_weights"],
                dtype=np.float64,
            ).copy()

        return (
            self.model.obs_from_state(self.state),
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
    
class PortfolioChoiceEnv(BaseEconEnv):
    MODEL_CLS = PortfolioChoice
