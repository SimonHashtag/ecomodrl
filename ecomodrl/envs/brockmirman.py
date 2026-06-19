from typing import Union, Tuple, Optional

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from ecomodrl.configs.parameters import BrockMirmanParams

__all__ = ["BrockMirmanEnv"]

# ---- Default parameters (can be overridden via __init__) ----
RESET_K_DEFAULT = 5.0  # Capital in first period
TOL = 1e-3             # Minimum distance to no/complete consumption 


class BrockMirmanEnv(gym.Env):
    """
    Stochastic Brock–Mirman growth environment with a single control:
    action a_t in (0,1) is the fraction of available resources consumed.
    State s_t is (z_t, k_t) with multiplicative technology shock z_t.

    Dynamics:
        y_t = z_t * k_t^alpha
        resources_t = (1 - delta) * k_t + y_t
        c_t = a_t * resources_t
        k_{t+1} = resources_t - c_t
        log z_{t+1} = rho * log z_t + sigma * eps_{t+1},   eps ~ N(0,1)

    Reward r_t = u(c_t) (CRRA utility).
    """
    def __init__(
        self,
        *,
        model_params: BrockMirmanParams,
        fixed_start: bool = False,      # if True, always start from RESETSTATE (for k)
        reset_state: float = RESET_K_DEFAULT,
        state_min: float = 1.0,         # min random start capital (if not fixed)
        state_max: float = 15.0,        # max random start capital (if not fixed)
        floattype = np.float32
    ):
        super().__init__()

        # parameters
        self.GAMMA = model_params.gamma
        self.DELTA = model_params.delta
        self.ALPHA = model_params.alpha
        self.RHO   = model_params.rho
        self.SIGMA = model_params.sigma

        self._fixed_start = fixed_start
        self.reset_state = reset_state
        self.state_min, self.state_max = state_min, state_max
        self.floattype = floattype

        # action (consumption share)
        self.action_space = Box(TOL, 1.0 - TOL, shape=(1,), dtype=self.floattype)

        # states (log(z), log(k))
        self.observation_space = Box(
            low=np.array([-np.inf, -np.inf], dtype=self.floattype),
            high=np.array([np.inf, np.inf], dtype=self.floattype),
            shape=(2,),
            dtype=self.floattype,
        )

        # RNG
        self._seed: Optional[int] = None
        self._rng = np.random.default_rng()

        # internal counters / state holders
        self._t: int = 0
        self._k: Optional[float] = None
        self._z: Optional[float] = None

    def _get_obs(self) -> np.ndarray:
        # Log-scale z and k
        assert self._z is not None and self._k is not None
        return np.array([np.log(self._z), np.log(self._k)], dtype=self.floattype)

    def _get_info(self) -> dict:
        return {
            "z": self._z,
            "capital": self._k,
            "seed": self._seed,
            "DELTA": self.DELTA,
            "GAMMA": self.GAMMA,
            "ALPHA": self.ALPHA,
            "RHO": self.RHO,
            "SIGMA": self.SIGMA,
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

        # capital & technology
        if self._fixed_start or (options and options.get("fixed_start", False)):
            self._k = self.floattype(self.reset_state)
            self._z = self.floattype(1.0)
        else:
            self._k = self.floattype(self._rng.uniform(self.state_min, self.state_max))
            self._z = self.floattype(self._rng.lognormal(mean=0.0, sigma=np.sqrt(self.SIGMA**2 / (1 - self.RHO**2))))

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Update (z, k), compute reward, and return transition tuple.
        Args:
            action: float consumption share a_t in (TOL, 1-TOL)
        Returns: Tuple[obs, reward, terminated, truncated, info]
        """
        # ensure shape and bounds
        a = float(np.clip(np.asarray(action, dtype=self.floattype).reshape(-1)[0], TOL, 1.0 - TOL))
        assert self._k is not None and self._z is not None, "Call reset before using step method."

        self._t += 1

        # production and resources before consumption
        y = float(self._z) * (float(self._k) ** self.ALPHA)
        resources = (1.0 - self.DELTA) * float(self._k) + y

        # consumption and reward
        c = a * resources
        r = crra_utility(c, gamma=self.GAMMA, floattype=self.floattype)

        # capital transition
        self._k = resources - c
        self._k = float(max(self._k, 0.0))

        # technology shock: log z_{t+1} = rho log z_t + sigma * eps, eps ~ N(0,1)
        eps = self._rng.normal()
        logz_next = self.RHO * np.log(float(self._z)) + self.SIGMA * eps
        self._z = float(np.exp(logz_next))

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
    """CRRA utility"""
    c = float(np.maximum(consumption, TOL))
    if gamma == 1:
        u = np.log(c)
    else:
        u = (c ** (1 - gamma)) / (1 - gamma)
    return floattype(u)