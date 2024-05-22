import logging
from typing import List, Tuple, Optional

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

__all__ = ["CassKoopmansEnv"]

A = 1 # technology factor
GAMMA = 2 # coefficient of relative risk aversion
DELTA = 0.02 # depreciation rate of capital
ALPHA = 0.33  # return to capital per capita
RESETSTATE = 5.0  # Capital in first period
TOL = 1e-6  # Minimum distance to no/complete consumption 


class CassKoopmansEnv(gym.Env):
    def __init__(self, 
                 max_episode_steps=100, 
                 finite=False,
                 A=A,
                 GAMMA=GAMMA,
                 DELTA=DELTA,
                 ALPHA=ALPHA,
                 RESETSTATE=RESETSTATE):
        
        # parameters
        self.A = A
        self.GAMMA = GAMMA
        self.DELTA = DELTA
        self.ALPHA = ALPHA
        self.RESETSTATE = RESETSTATE
        
        # action (consumption)
        self.action_space = Box(TOL, 1.0-TOL, shape=(1,), dtype=np.float32)

        # states (capital)
        self.observation_space = Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)

        self.max_episode_steps = max_episode_steps
        self._finite = finite

    def _get_obs(self):
        return np.array([np.float32(self._capital)])

    def _get_info(self):
        return {
            "capital": self._capital,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.period_iter = 0
        self._capital = np.float32(self.RESETSTATE)  # capital

        observations = self._get_obs()
        info = self._get_info()

        return observations, info

    def step(self, action: float) -> Tuple[dict, float, bool, bool, Optional[dict]]:
        """
        Update capital.
        Calculate the reward (crra utility function) and determine whether episode is to be truncated.
        Args:
            action: float consumption (share of available capital)
        Returns: Tuple[obs, reward, terminated, truncated, info]
        """
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self._capital is not None, "Call reset before using step method."

        self.period_iter += 1
        
        # get consumption and reward
        consumption = self._capital * action[0]
        reward = crra_utility(consumption, gamma=self.GAMMA)
        
        # get production
        y = self.A * self._capital ** self.ALPHA

        # update capital
        self._capital *= 1 - self.DELTA
        self._capital += y
        self._capital -= consumption
        
        assert self._capital >= 0.0, f"capital is not allowed to be smaller than {0.0} but is {self._capital} " f"A {A} " f"DELTA {DELTA} " f"consumption {consumption} " f"reward {reward} "

        observations = self._get_obs()
        terminated = self._finite and self.period_iter >= self.max_episode_steps
        truncated = not self._finite and self.period_iter >= self.max_episode_steps
        info = self._get_info()
        info["consumption_frac"] = action[0]
        info["consumption"] = consumption
        info["reward"] = reward
        info["A"] = self.A
        info["DELTA"] = self.DELTA
        info["GAMMA"] = self.GAMMA

        return observations, reward, terminated, truncated, info


def crra_utility(consumption: float, gamma: float = GAMMA) -> float:
    """Get the utility based on consumption.
    For consumption ~ 0.0 return a very small utility (not -inf however).
    For consumption above 0.0 return the log utility
    Args:
        consumption: the consumption of risky asset of the period
    Returns:
        float: the utility
    """
    assert consumption > 0.0, "Non-positive consumption not allowed"
    
    if gamma == 1:
        u = np.log(consumption, dtype=np.float32)
    else:
        u = np.float32((consumption ** (1 - gamma)) / (1 - gamma))
    return u
