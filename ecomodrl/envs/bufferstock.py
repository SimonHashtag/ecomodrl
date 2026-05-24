# bufferstock.py

from typing import Tuple, Optional
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from ecomodrl.envs.parameters import BufferStockParams  # drop-in dataclass below

__all__ = ["BufferStockSavingsEnv"]

# Defaults (can be overridden via __init__)
RESET_M_DEFAULT = 5.0      # normalized market resources at reset
TOL = 1e-3                 # clamp to avoid zero/one consumption corners


class BufferStockSavingsEnv(gym.Env):
    """
    Buffer-Stock Savings model (Carroll, 1997) with *normalized* variables.

    State s_t = m_t  (scalar): normalized market resources (by current permanent income).
    Action a_t in (0,1): fraction of m_t consumed, i.e., c_t = a_t * m_t.

    Shocks each period:
      - Transitory income θ_{t+1}: mixture with unemployment mass at θ_low (prob=p),
        otherwise lognormal with mean adjusted so E[θ]=1.
      - Permanent shock ψ_{t+1} ~ lognormal with E[ψ]=1 (mean of log = -½ σ_perm^2).

    Transition in normalized units:
        a^{post}_t = m_t - c_t = (1 - a_t) m_t
        m_{t+1}    = (R / (G * ψ_{t+1})) * a^{post}_t + θ_{t+1}

    Reward:
        r_t = u(c_t) = CRRA(c_t), applied to normalized consumption.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        *,
        model_params: BufferStockParams,
        fixed_start: bool = False,            # if True, always start from RESET_M_DEFAULT (or provided reset_state)
        reset_state: float = RESET_M_DEFAULT,
        state_min: float = 1.0,               # min random start m (if not fixed)
        state_max: float = 15.0,              # max random start m (if not fixed)
        floattype: type = np.float32
    ):
        super().__init__()

        # --- parameters ---
        self.GAMMA = model_params.gamma
        self.BETA  = model_params.beta
        self.R     = 1.0 + model_params.r
        self.G     = 1.0 + model_params.g
        self.SIGMA_TRAN = model_params.sigma_tran
        self.SIGMA_PERM = model_params.sigma_perm
        self.PROB_UNEMP = model_params.unemp_prob
        self.THETA_LOW  = model_params.theta_low

        # mean of log θ for E[θ]=1 (given point mass at θ_low with prob p)
        # μ_z = log((1 - p*θ_low)/(1 - p)) - 0.5 σ_tran^2
        self.MU_Z = float(
            np.log((1.0 - self.PROB_UNEMP * self.THETA_LOW) / max(1e-12, (1.0 - self.PROB_UNEMP)))
            - 0.5 * (self.SIGMA_TRAN ** 2)
        )
        self.psi_log_mean = -0.5 * self.SIGMA_PERM ** 2

        # --- start / rng / dtype ---
        self._fixed_start = fixed_start
        self.reset_state = reset_state
        self.state_min, self.state_max = state_min, state_max
        self.floattype = floattype

        # action: consumption share in (TOL, 1]
        self.action_space = Box(TOL, 1.0, shape=(1,), dtype=self.floattype)

        # observation: log-scaled market resources log(m_t)
        self.observation_space = Box(
            low=np.array([-np.inf], dtype=self.floattype),
            high=np.array([np.inf], dtype=self.floattype),
            shape=(1,),
            dtype=self.floattype,
        )

        # RNG
        self._seed: Optional[int] = None
        self._rng = np.random.default_rng()

        # internal counters / state holders
        self._t: int = 0
        self._m: Optional[float] = None

        # last shocks (exposed in info for debugging/analysis)
        self._last_theta: Optional[float] = None
        self._last_psi: Optional[float] = None

    # ---------- helpers ----------
    def _get_obs(self) -> np.ndarray:
        # Log-scale m
        return np.array([np.log(self._m)], dtype=self.floattype)

    def _get_info(self) -> dict:
        return {
            "market_resources": self._m,
            "seed": self._seed,
            "BETA": self.BETA,
            "GAMMA": self.GAMMA,
            "R": self.R,
            "G": self.G,
            "SIGMA_TRAN": self.SIGMA_TRAN,
            "SIGMA_PERM": self.SIGMA_PERM,
            "PROB_UNEMP": self.PROB_UNEMP,
            "THETA_LOW": self.THETA_LOW,
            "MU_Z": self.MU_Z,
            "theta": self._last_theta,
            "psi": self._last_psi,
        }

    def _draw_shocks(self) -> Tuple[float, float]:
        """
        Draw (θ, ψ). Uses the env RNG for determinism under seeding.
        """
        # θ: unemployment mass at θ_low with prob p; else lognormal with mean-adjusted μ_z
        if self._rng.uniform(0.0, 1.0) < self.PROB_UNEMP:
            theta = self.THETA_LOW
        else:
            theta = self.floattype(np.exp(self._rng.normal(self.MU_Z, self.SIGMA_TRAN)))

        # ψ: lognormal with mean 1  (log-mean = -½ σ^2, log-std = σ)
        psi = self.floattype(np.exp(self._rng.normal(self.psi_log_mean, self.SIGMA_PERM)))
        return theta, psi

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
            try:
                self.action_space.seed(self._seed)
                self.observation_space.seed(self._seed)
            except Exception:
                pass

        self._t = 0
        self._last_theta, self._last_psi = None, None

        if self._fixed_start or (options and options.get("fixed_start", False)):
            self._m = self.floattype(self.reset_state)
        else:
            self._m = self.floattype(self._rng.uniform(self.state_min, self.state_max))

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        """
        Execute one period:
          - consume c = a * m,
          - reward r = u(c),
          - transition m' = (R/(G*ψ)) * (m - c) + θ .
        """
        a_share = self.floattype(np.clip(np.asarray(action, dtype=self.floattype).reshape(-1)[0], TOL, 1.0))
        assert self._m is not None, "Call reset before using step method."

        self._t += 1

        # Consumption and reward (normalized)
        c = a_share * self.floattype(self._m)
        r = crra_utility(c, gamma=self.GAMMA, floattype=self.floattype)

        # Post-consumption assets (normalized)
        a_post = self.floattype(self._m) - c

        # Shocks and transition
        theta, psi = self._draw_shocks()
        m_next = (self.R / (self.G * psi)) * a_post + theta

        # numerical safety: keep m >= 0
        self._m = self.floattype(max(m_next, 0.0))
        self._last_theta, self._last_psi = theta, psi

        obs = self._get_obs()
        terminated = False
        truncated = False

        info = self._get_info()
        info.update({
            "consumption_frac": a_share,
            "consumption": c,
            "assets_post": a_post,
            "reward": r,
        })

        return obs, float(r), terminated, truncated, info
    

def crra_utility(consumption: float, gamma: float, floattype=np.float32) -> float:
    """
    CRRA utility u(c) with guard for strictly positive c.
    gamma == 1 -> log utility.
    """
    c = float(np.maximum(consumption, TOL)) # guard against non-positive consumption
    if gamma == 1:
        u = np.log(c)
    else:
        u = (c ** (1 - gamma)) / (1 - gamma)
    return floattype(u)
