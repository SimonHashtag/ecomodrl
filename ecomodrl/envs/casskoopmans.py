from typing import Union, Tuple, Optional

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from ecomodrl.configs.parameters import CassKoopmansParams

__all__ = ["CassKoopmansEnv"]

# ---- Default parameters (can be overridden via __init__) ----
RESET_K_DEFAULT = 5.0  # Capital in first period
TOL = 1e-3  # Minimum distance to no/complete consumption 


class CassKoopmansEnv(gym.Env):
    """
    Deterministic Cass-Koopmans growth environment with a single control:
    action a_t in (0,1) is the fraction of available resources consumed.
    State s_t is capital k_t (scalar).
    Dynamics:
        y_t   = A * k_t^alpha
        k_{t+1} = (1 - delta) * k_t + y_t - c_t
        c_t = a_t * [(1 - delta) * k_t + y_t]
    Reward r_t = u(c_t) (CRRA utility).
    """
    def __init__(
        self, 
        *,
        model_params: CassKoopmansParams,
        fixed_start: bool = False,      # if True, always start from RESETSTATE
        reset_state: float = RESET_K_DEFAULT,
        state_min: float = 1.0,            # min random start capital (if not fixed)
        state_max: float = 15.0,            # max random start capital (if not fixed)
        floattype = np.float32
        ):
        super().__init__()
        
        # parameters
        self.A = model_params.A
        self.GAMMA = model_params.gamma
        self.DELTA = model_params.delta
        self.ALPHA = model_params.alpha
        self._fixed_start = fixed_start
        self.reset_state = reset_state
        self.state_min, self.state_max = state_min, state_max
        self.floattype = floattype
        
        # action (consumption)
        self.action_space = Box(TOL, 1.0-TOL, shape=(1,), dtype=self.floattype)

        # states (log(k))
        self.observation_space = Box(
            low=np.array([-np.inf], dtype=self.floattype), 
            high=np.array([np.inf], dtype=self.floattype), 
            shape=(1,), 
            dtype=self.floattype,)

        # RNG
        self._seed: Optional[int] = None
        self._rng = np.random.default_rng()
        
        # internal counters / state holders
        self._t: int = 0
        self._k: Optional[float] = None

    def _get_obs(self) -> np.ndarray:
        # Log-scale k
        return np.array([np.log(self._k)], dtype=self.floattype)

    def _get_info(self) -> dict:
        return {
            "capital": self._k,
            "seed": self._seed,
            "A": self.A,
            "DELTA": self.DELTA,
            "GAMMA": self.GAMMA,
            "ALPHA": self.ALPHA,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
            try:
                self.action_space.seed(self._seed)
                self.observation_space.seed(self._seed)
            except Exception:
                pass

        self._t = 0
        
        if self._fixed_start or (options and options.get("fixed_start", False)):
            self._k = self.floattype(self.reset_state)
        else:
            self._k = self.floattype(self._rng.uniform(self.state_min, self.state_max))

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Update capital.
        Calculate the reward (crra utility function) and determine whether episode is to be truncated.
        Args:
            action: float consumption (share of available capital)
        Returns: Tuple[obs, reward, terminated, truncated, info]
        """
        # ensure shape and bounds (avoid failing hard on tiny numeric drift)
        a = float(np.clip(np.asarray(action, dtype=self.floattype).reshape(-1)[0], TOL, 1.0 - TOL))
        assert self._k is not None, "Call reset before using step method."

        self._t += 1
        
        # production and resources available before consumption
        y = self.A * (self._k ** self.ALPHA)
        resources = (1.0 - self.DELTA) * self._k + y
        
        # get consumption and reward
        c = a * resources
        r = crra_utility(c, gamma=self.GAMMA, floattype=self.floattype)
        
        # capital transition
        self._k = resources - c
        # numerical safety: capital should stay non-negative
        self._k = float(max(self._k, 0.0))

        obs = self._get_obs()
        terminated = False
        truncated = False
        info = self._get_info()
        info.update({
            "consumption_frac": a,
            "consumption": c,
            "reward": r,
        })

        return obs, float(r), terminated, truncated, info


def crra_utility(consumption: float, gamma: float, floattype=np.float32) -> float:
    """Get the CRRA utility from consumption.
    Args:
        consumption: the consumption of the period
    Returns:
        float: the utility
    """
    
    # Guard against non-positive consumption
    c = float(np.maximum(consumption, TOL))
    
    if gamma == 1:
        u = np.log(c)
    else:
        u = (c ** (1 - gamma)) / (1 - gamma)
    return floattype(u)


def c_from_next_k(k:float, a:float, alpha:float, delta:Union[float, int]):
    f = k**alpha
    y_res = f + (1.0 - delta) * k
    return np.clip(a, 0.0, 1.0) * y_res
