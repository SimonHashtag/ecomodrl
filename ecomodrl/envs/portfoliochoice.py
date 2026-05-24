# portfoliochoice.py

from typing import Tuple, Optional
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from ecomodrl.envs.parameters import PortfolioChoiceParams  # expected dataclass, see notes below

__all__ = ["PortfolioChoiceEnv"]

# ---- Defaults (can be overridden via __init__) ----
RESET_M_DEFAULT = 5.0      # normalized market resources at reset
TOL = 1e-3                 # clamp for positive consumption & shares
TOL_W = 1e-8               # tiny floor for portfolio weights in softmax mapping
LOGIT_CAP = 12.0           # numeric cap for action logits mapping to simplex


class PortfolioChoiceEnv(gym.Env):
    """
    Portfolio-Choice + Buffer-Stock Savings environment in normalized units.

    ## State
    s_t = (m_t, z_{1t}, ..., z_{Nt})
      - m_t: normalized market resources (cash-on-hand / P_t), m_t >= 0.
      - z_{jt}: return predictor for risky asset j (AR(1)).

    ## Action
    a_t = (α_t, ℓ_{0t}, ℓ_{1t}, ..., ℓ_{Nt})
      - α_t ∈ (TOL, 1-TOL): consumption share; c_t = α_t * m_t.
      - ℓ_t: (N+1) logits mapped to portfolio weights ω_t via softmax.
        Weights satisfy ω_i ≥ 0 and ∑_i ω_i = 1 (no shorting, fully invested).
        Asset i=0 is the risk-free asset; i=1..N are risky assets.

    ## Shocks each period
      - Permanent ψ_{t+1} ~ lognormal with E[ψ]=1 (log-mean = -½ σ_perm²).
      - Transitory income θ_{t+1}: unemployment mass at θ_low w.p. p, else lognormal with mean adjusted for E[θ]=1.
      - Predictors evolve as z_{j,t+1} = ρ_j z_{j,t} + σ_{z,j} ε_{j,t+1}, ε ~ N(0,1) iid.
      - Risky returns: ln R_{j,t+1} = κ_j + ξ_j z_{j,t} + σ_{r,j} η_{j,t+1}, η ~ N(0,1).
        Risk-free return R_f is constant.

    ## Transition (normalized)
      rs_t = (1 - α_t) m_t
      R^p_{t+1} = ω_{0t} R_f + ∑_{j=1}^N ω_{jt} R_{j,t+1}
      m_{t+1}   = (R^p_{t+1} / (G * ψ_{t+1})) * rs_t + θ_{t+1}

    ## Reward
      r_t = u(c_t) where u is CRRA with parameter gamma (log-utility if gamma=1).

    Notes
    -----
    * All variables are normalized by current permanent income P_t.
    * The mapping from logits to the simplex stabilizes numerics (keeps weights ≥ TOL_W after softmax and re-normalization). 
    TODO: Softmax should possibly be implemented in the neural network instead.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        *,
        model_params: PortfolioChoiceParams,
        fixed_start: bool = False,            # if True, start from RESET_M_DEFAULT (or provided reset_state)
        reset_state: float = RESET_M_DEFAULT,
        state_min: float = 1.0,               # min random start m (if not fixed)
        state_max: float = 15.0,              # max random start m (if not fixed)
        z0: Optional[np.ndarray] = None,      # optional initial z vector (length N)
        floattype: type = np.float32
    ):
        super().__init__()

        # --- parameters ---
        # Preferences & growth
        self.GAMMA = model_params.gamma
        self.BETA  = model_params.beta
        self.G     = 1.0 + model_params.g

        # Income process
        self.SIGMA_TRAN = model_params.sigma_tran
        self.SIGMA_PERM = model_params.sigma_perm
        self.PROB_UNEMP = model_params.unemployment_prob
        self.THETA_LOW  = model_params.theta_low

        # mean of log θ for E[θ]=1 (given point mass at θ_low with prob p)
        # μ_z = log((1 - p*θ_low)/(1 - p)) - 0.5 σ_tran^2
        self.MU_Z = float(
            np.log((1.0 - self.PROB_UNEMP * self.THETA_LOW) / max(1e-12, (1.0 - self.PROB_UNEMP)))
            - 0.5 * (self.SIGMA_TRAN ** 2)
        )
        self.psi_log_mean = -0.5 * self.SIGMA_PERM ** 2

        # Assets
        self.N = int(model_params.n_risky)               # number of risky assets
        self.RF = 1.0 + model_params.r                   # gross risk-free return

        # Predictors & risky returns parameters (length-N arrays)
        self.RHO_Z   = np.asarray(model_params.rho_z,   dtype=float).reshape(self.N)
        self.SIGMA_Z = np.asarray(model_params.sigma_z, dtype=float).reshape(self.N)

        self.KAPPA   = np.asarray(model_params.kappa_R, dtype=float).reshape(self.N)
        self.XI      = np.asarray(model_params.xi_R,    dtype=float).reshape(self.N)
        self.SIGMA_R = np.asarray(model_params.sigma_R, dtype=float).reshape(self.N)

        # --- start / rng / dtype ---
        self._fixed_start = fixed_start
        self.reset_state = float(reset_state)
        self.state_min, self.state_max = float(state_min), float(state_max)
        self.floattype = floattype

        # action: [consumption share α] + [logits ℓ_0,...,ℓ_N] mapped to simplex
        # α in (TOL, 1-TOL), logits in [-LOGIT_CAP, LOGIT_CAP] for stability
        low  = np.concatenate(([TOL],  -LOGIT_CAP * np.ones(self.N + 1)))
        high = np.concatenate(([1.0 - TOL], LOGIT_CAP * np.ones(self.N + 1)))
        self.action_space = Box(low=low.astype(self.floattype),
                                high=high.astype(self.floattype),
                                shape=(1 + self.N + 1,),
                                dtype=self.floattype)

        # observation: [m, z_1,...,z_N]
        self.observation_space = Box(
            low=np.concatenate(([0.0], -np.inf * np.ones(self.N))).astype(self.floattype),
            high=np.concatenate(([np.finfo(self.floattype).max], np.inf * np.ones(self.N))).astype(self.floattype),
            shape=(1 + self.N,),
            dtype=self.floattype,
        )

        # RNG
        self._seed: Optional[int] = None
        self._rng = np.random.default_rng()

        # internal counters / state holders
        self._t: int = 0
        self._m: Optional[float] = None
        self._z: Optional[np.ndarray] = None  # shape (N,)

        # last shocks (exposed in info for debugging/analysis)
        self._last_theta: Optional[float] = None
        self._last_psi: Optional[float] = None
        self._last_eps: Optional[np.ndarray] = None  # z innovations
        self._last_eta: Optional[np.ndarray] = None  # return innovations
        self._last_R_risky: Optional[np.ndarray] = None  # realized risky gross returns (N,)
        self._last_Rp: Optional[float] = None         # realized gross portfolio return
        self._last_w: Optional[np.ndarray] = None # (N+1,)

        # optional provided starting z # TODO: Think about z initialization
        if z0 is not None:
            z0 = np.asarray(z0, dtype=self.floattype).reshape(self.N)
        self._z0 = z0

    # ---------- helpers ----------
    def _softmax_simplex(self, logits: np.ndarray) -> np.ndarray:
        """Map real logits to the (N+1)-simplex with tiny floor for numerics."""
        x = np.clip(logits.astype(float), -LOGIT_CAP, LOGIT_CAP)
        x = x - np.max(x)  # stabilize
        e = np.exp(x)
        w = e / np.sum(e)
        # tiny floor + renormalize to avoid exact zeros (keeps KKT direction well-defined in practice)
        w = np.maximum(w, TOL_W)
        w = w / np.sum(w)
        return w.astype(self.floattype)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(([self._m], self._z), axis=0).astype(self.floattype)

    def _get_info(self) -> dict:
        return {
            "market_resources": float(self._m),
            "predictors": None if self._z is None else self._z.copy(),
            "seed": self._seed,
            # preferences & growth
            "BETA": self.BETA, "GAMMA": self.GAMMA, "G": self.G,
            # income
            "SIGMA_TRAN": self.SIGMA_TRAN, "SIGMA_PERM": self.SIGMA_PERM,
            "PROB_UNEMP": self.PROB_UNEMP, "THETA_LOW": self.THETA_LOW, "MU_Z": self.MU_Z,
            # assets
            "RF": self.RF,
            "KAPPA": self.KAPPA.copy(), "XI": self.XI.copy(),
            "SIGMA_R": self.SIGMA_R.copy(), "RHO_Z": self.RHO_Z.copy(), "SIGMA_Z": self.SIGMA_Z.copy(),
            # last period realizations
            "theta": self._last_theta, "psi": self._last_psi,
            "eps_z": None if self._last_eps is None else self._last_eps.copy(),
            "eta_r": None if self._last_eta is None else self._last_eta.copy(),
            "R_risky": None if self._last_R_risky is None else self._last_R_risky.copy(),
            "R_portfolio": self._last_Rp,
            "w": None if self._last_w is None else self._last_w.copy(),
        }

    def _draw_income_shocks(self) -> Tuple[float, float]:
        """
        Draw (θ, ψ).
        θ has unemployment mass at θ_low with prob p; else lognormal with mean-adjusted μ_z for E[θ]=1.
        ψ is lognormal with mean 1 (log-mean = -½ σ_perm^2).
        """
        if self._rng.uniform(0.0, 1.0) < self.PROB_UNEMP:
            theta = self.THETA_LOW
        else:
            theta = float(np.exp(self._rng.normal(self.MU_Z, self.SIGMA_TRAN)))

        psi = float(np.exp(self._rng.normal(self.psi_log_mean, self.SIGMA_PERM)))
        self._last_theta, self._last_psi = self.floattype(theta), self.floattype(psi)
        return theta, psi

    def _draw_risky_returns(self) -> np.ndarray:
        """
        Draw η shocks and produce risky gross returns R_j = exp(κ_j + ξ_j z_j + σ_{r,j} η_j).
        Then, draw ε shocks and update predictors in place z_{t+1} = ρ ⊙ z_t + σ_z ⊙ ε
        """
        eta = self._rng.standard_normal(self.N)
        lnR = self.KAPPA + self.XI * self._z + self.SIGMA_R * eta
        R = np.exp(lnR).astype(self.floattype)
        eps = self._rng.standard_normal(self.N)
        z_next = self.RHO_Z * self._z + self.SIGMA_Z * eps
        self._z = z_next.astype(self.floattype)
        self._last_eta = eta.astype(self.floattype)
        self._last_R_risky = R.copy()
        self._last_eps = eps.astype(self.floattype)
        return R

    # ---------- Gym API ----------
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
        self._last_theta = self._last_psi = None
        self._last_eps = self._last_eta = None
        self._last_R_risky = None
        self._last_Rp = None
        self._last_w = None

        # start m
        if self._fixed_start or (options and options.get("fixed_start", False)):
            self._m = self.floattype(self.reset_state)
        else:
            self._m = self.floattype(self._rng.uniform(self.state_min, self.state_max))

        # start z
        if self._z0 is not None:
            self._z = self._z0.astype(self.floattype).copy()
        else:
            # initialize from stationary distribution of AR(1): Var = σ_z^2 / (1-ρ^2)
            std = np.abs(self.SIGMA_Z / np.sqrt(np.maximum(1e-12, 1.0 - self.RHO_Z**2)))
            self._z = self._rng.normal(loc=0.0, scale=std, size=self.N).astype(self.floattype)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        """
        Execute one period:
          1) Parse action: α (consumption share) and logits ℓ → weights ω via softmax.
          2) Consume c = α * m; reward r = u(c).
          3) Invest a_post = (1-α) * m across assets with ω.
          4) Realize returns:
             - Risky: ln R_j = κ_j + ξ_j z_j + σ_{r,j} η_j
             - Risk-free: R_f
             - Portfolio: R^p = ω_0 R_f + Σ_{j=1..N} ω_j R_j
          5) Draw income shocks (θ, ψ) and predictors z' ; transition:
             m' = (R^p / (G * ψ)) * a_post + θ
        """
        assert self._m is not None and self._z is not None, "Call reset() before step()."

        # parse action
        a = np.asarray(action, dtype=self.floattype).reshape(-1)
        # consumption share
        alpha = self.floattype(np.clip(a[0], TOL, 1.0 - TOL))
        # portfolio weights (from logits)
        logits = a[1: 1 + self.N + 1]
        if logits.shape[0] != self.N + 1:
            raise ValueError(f"Expected {self.N+1} logits; got {logits.shape[0]}")
        w = self._softmax_simplex(logits)

        self._t += 1

        # Consumption & reward
        c = alpha * self.floattype(self._m)
        r = crra_utility(float(c), gamma=self.GAMMA, floattype=self.floattype)

        # Post-consumption assets
        a_post = self.floattype(self._m) - c

        # Returns realization (use current z for returns; then advance z for next period state)
        R_risky = self._draw_risky_returns()  # shape (N,)
        Rp = float(w[0] * self.RF + np.dot(w[1:], R_risky))

        # Income shocks and predictor transition
        theta, psi = self._draw_income_shocks()

        # Transition
        m_next = (Rp / (self.G * psi)) * a_post + theta

        # numerical safety
        self._m = self.floattype(max(m_next, 0.0))
        self._last_Rp = self.floattype(Rp)
        self._last_w = w.astype(self.floattype)

        obs = self._get_obs()
        terminated = False
        truncated = False

        info = self._get_info()
        info.update({
            "consumption_frac": float(alpha),
            "consumption": float(c),
            "assets_post": float(a_post),
            "reward": float(r),
        })

        return obs, float(r), terminated, truncated, info


def crra_utility(consumption: float, gamma: float, floattype=np.float32) -> float:
    """
    CRRA utility u(c) with guard for strictly positive c.
    gamma == 1 -> log utility.
    """
    c = float(max(consumption, TOL))
    if gamma == 1:
        u = np.log(c)
    else:
        u = (c ** (1 - gamma)) / (1 - gamma)
    return floattype(u)
