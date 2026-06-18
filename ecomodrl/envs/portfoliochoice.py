# portfoliochoice.py

from typing import Tuple, Optional
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from ecomodrl.envs.parameters import PortfolioChoiceParams

__all__ = ["PortfolioChoiceEnv"]

# ---- Defaults (can be overridden via __init__) ----
RESET_M_DEFAULT = 5.0      # normalized market resources at reset
TOL = 1e-3                 # clamp for positive consumption
LOGIT_CAP = 12.0           # numeric cap for action logits mapping to simplex


class PortfolioChoiceEnv(gym.Env):
    """
    Portfolio-Choice + Buffer-Stock Savings environment in normalized units.

    ## State
    s_t = (m_t, z_{1t}, ..., z_{Nt})
      - m_t: normalized market resources (cash-on-hand / P_t), m_t >= 0.
      - z_{jt}: return predictor for risky asset j (AR(1)).

    ## Action
    a_t = (α_t, ℓ_{1t}, ..., ℓ_{Nt})
      - α_t ∈ (TOL, 1.0]: consumption share; c_t = α_t * m_t.
      - ℓ_t: N logits mapped to (N+1)-simplex weights via the additive log-ratio
        (ALR) inverse:  ω_i = exp(ℓ_i) / (1 + Σ_j exp(ℓ_j)) for i=1..N,
                        ω_{N+1} = 1 / (1 + Σ_j exp(ℓ_j)).
        With this convention the *last* component (here: index N, i.e. risky
        asset N) plays the role of the ALR reference / safe pivot in
        log-ratio space.  We then *reorder* the (N+1)-vector so that index 0
        is the risk-free asset and indices 1..N are risky, matching the EGM
        solver's `[safe, risky_1, ..., risky_N]` convention.
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
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        *,
        model_params: PortfolioChoiceParams,
        fixed_start: bool = False,
        reset_state: float = RESET_M_DEFAULT,
        state_min: float = 0.2,               # aligned with EGM m_grid span (0.2, 30.0)
        state_max: float = 30.0,              # aligned with EGM m_grid span (0.2, 30.0)
        z0: Optional[np.ndarray] = None,      # default: zeros, matches EGM evaluator
        floattype: type = np.float64,         # (a) float64 by default
    ):
        super().__init__()

        # --- preferences & growth ---
        self.GAMMA = float(model_params.gamma)
        self.BETA  = float(model_params.beta)
        self.G     = 1.0 + float(model_params.g)

        # --- income process ---
        self.SIGMA_TRAN = float(model_params.sigma_tran)
        self.SIGMA_PERM = float(model_params.sigma_perm)
        self.PROB_UNEMP = float(model_params.unemployment_prob)
        self.THETA_LOW  = float(model_params.theta_low)

        self.MU_Z = float(
            np.log((1.0 - self.PROB_UNEMP * self.THETA_LOW)
                   / max(1e-12, (1.0 - self.PROB_UNEMP)))
            - 0.5 * (self.SIGMA_TRAN ** 2)
        )
        self.psi_log_mean = -0.5 * self.SIGMA_PERM ** 2

        # --- assets ---
        self.N  = int(model_params.n_risky)
        self.RF = 1.0 + float(model_params.r)

        # Predictors & risky returns parameters (length-N arrays, float64)
        self.RHO_Z   = np.asarray(model_params.rho_z,   dtype=np.float64).reshape(self.N)
        self.SIGMA_Z = np.asarray(model_params.sigma_z, dtype=np.float64).reshape(self.N)
        self.KAPPA   = np.asarray(model_params.kappa_R, dtype=np.float64).reshape(self.N)
        self.XI      = np.asarray(model_params.xi_R,    dtype=np.float64).reshape(self.N)
        self.SIGMA_R = np.asarray(model_params.sigma_R, dtype=np.float64).reshape(self.N)

        # --- start / rng / dtype ---
        self._fixed_start = fixed_start
        self.reset_state  = float(reset_state)
        self.state_min, self.state_max = float(state_min), float(state_max)
        self.floattype = floattype

        # (c) ALR action layout: α ∈ (TOL, 1.0], plus N logits (NOT N+1)
        low  = np.concatenate(([TOL],       -LOGIT_CAP * np.ones(self.N)))
        high = np.concatenate(([1.0],        LOGIT_CAP * np.ones(self.N)))
        self.action_space = Box(
            low=low.astype(self.floattype),
            high=high.astype(self.floattype),
            shape=(1 + self.N,),
            dtype=self.floattype,
        )

        # observation: [m, z_1,...,z_N]
        self.observation_space = Box(
            low=np.concatenate(([0.0], -np.inf * np.ones(self.N))).astype(self.floattype),
            high=np.concatenate(([np.finfo(self.floattype).max],
                                 np.inf * np.ones(self.N))).astype(self.floattype),
            shape=(1 + self.N,),
            dtype=self.floattype,
        )

        # RNG
        self._seed: Optional[int] = None
        self._rng = np.random.default_rng()

        # internal state
        self._t: int = 0
        self._m: Optional[float] = None
        self._z: Optional[np.ndarray] = None

        # last shocks (for info)
        self._last_theta = self._last_psi = None
        self._last_eps   = self._last_eta = None
        self._last_R_risky = None
        self._last_Rp      = None
        self._last_w       = None

        # (b) z0: default to zeros to match EGM evaluator's median start
        if z0 is None:
            self._z0 = np.zeros(self.N, dtype=self.floattype)
        else:
            self._z0 = np.asarray(z0, dtype=self.floattype).reshape(self.N)

    # ---------- helpers ----------
    def _alr_to_weights(self, logits: np.ndarray) -> np.ndarray:
        """
        (c) ALR inverse: N logits → (N+1)-simplex weights.

        Convention here:
          - The ALR pivot (the "1" in the denominator) corresponds to the
            risk-free asset, which is then placed at index 0 to match the
            EGM solver's [safe, risky_1, ..., risky_N] ordering.

        This is bijective and removes the translation-invariance redundancy
        of a softmax over N+1 logits.
        """
        x = np.clip(logits.astype(np.float64), -LOGIT_CAP, LOGIT_CAP)
        # subtract max for stability without changing the ratio structure
        # (only the relative magnitudes of the N logits and the pivot 0 matter)
        m = max(0.0, float(np.max(x)))
        e = np.exp(x - m)
        pivot = np.exp(-m)              # the pivot "1" rescaled
        denom = pivot + np.sum(e)
        w_risky = e / denom             # length N (risky weights)
        w_safe  = pivot / denom         # scalar  (safe weight, ALR pivot)
        w = np.concatenate(([w_safe], w_risky))
        return w.astype(self.floattype)

    def _get_obs(self) -> np.ndarray:
        # Log-scale z and m
        assert self._z is not None and self._m is not None
        return np.concatenate((np.log([self._m]), self._z), axis=0).astype(self.floattype)

    def _get_info(self) -> dict:
        return {
            "market_resources": float(self._m),
            "predictors": None if self._z is None else self._z.copy(),
            "seed": self._seed,
            "BETA": self.BETA, "GAMMA": self.GAMMA, "G": self.G,
            "SIGMA_TRAN": self.SIGMA_TRAN, "SIGMA_PERM": self.SIGMA_PERM,
            "PROB_UNEMP": self.PROB_UNEMP, "THETA_LOW": self.THETA_LOW, "MU_Z": self.MU_Z,
            "RF": self.RF,
            "KAPPA": self.KAPPA.copy(), "XI": self.XI.copy(),
            "SIGMA_R": self.SIGMA_R.copy(), "RHO_Z": self.RHO_Z.copy(), "SIGMA_Z": self.SIGMA_Z.copy(),
            "theta": self._last_theta, "psi": self._last_psi,
            "eps_z":   None if self._last_eps     is None else self._last_eps.copy(),
            "eta_r":   None if self._last_eta     is None else self._last_eta.copy(),
            "R_risky": None if self._last_R_risky is None else self._last_R_risky.copy(),
            "R_portfolio": self._last_Rp,
            "w":       None if self._last_w       is None else self._last_w.copy(),
        }

    def _draw_income_shocks(self) -> Tuple[float, float]:
        if self._rng.uniform(0.0, 1.0) < self.PROB_UNEMP:
            theta = self.THETA_LOW
        else:
            theta = float(np.exp(self._rng.normal(self.MU_Z, self.SIGMA_TRAN)))

        psi = float(np.exp(self._rng.normal(self.psi_log_mean, self.SIGMA_PERM)))
        self._last_theta = self.floattype(theta)
        self._last_psi   = self.floattype(psi)
        return theta, psi

    def _draw_risky_returns(self) -> np.ndarray:
        eta = self._rng.standard_normal(self.N)
        lnR = self.KAPPA + self.XI * self._z.astype(np.float64) + self.SIGMA_R * eta
        R = np.exp(lnR).astype(self.floattype)
        eps = self._rng.standard_normal(self.N)
        z_next = self.RHO_Z * self._z.astype(np.float64) + self.SIGMA_Z * eps
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
        self._last_eps   = self._last_eta = None
        self._last_R_risky = None
        self._last_Rp = None
        self._last_w  = None

        # start m, z
        if self._fixed_start or (options and options.get("fixed_start", False)):
            self._m = self.floattype(self.reset_state)
            self._z = self._z0.astype(self.floattype).copy()
        else:
            self._m = self.floattype(self._rng.uniform(self.state_min, self.state_max))
            # initialize from stationary distribution of AR(1): Var = σ_z^2 / (1-ρ^2)
            std = np.abs(self.SIGMA_Z / np.sqrt(np.maximum(1e-12, 1.0 - self.RHO_Z**2)))
            self._z = self._rng.normal(loc=0.0, scale=std, size=self.N).astype(self.floattype)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        assert self._m is not None and self._z is not None, "Call reset() before step()."

        a = np.asarray(action, dtype=np.float64).reshape(-1)

        # consumption share
        alpha = self.floattype(np.clip(a[0], TOL, 1.0))

        # (c) ALR: exactly N logits → (N+1)-simplex weights, safe at index 0
        logits = a[1: 1 + self.N]
        if logits.shape[0] != self.N:
            raise ValueError(
                f"Expected {self.N} logits (ALR for N+1 assets); got {logits.shape[0]}"
            )
        w = self._alr_to_weights(logits)

        self._t += 1

        # consumption & reward
        c = alpha * self.floattype(self._m)
        r = crra_utility(float(c), gamma=self.GAMMA, floattype=self.floattype)

        # post-consumption assets
        a_post = self.floattype(self._m) - c

        # returns realization
        R_risky = self._draw_risky_returns()  # shape (N,)
        Rp = float(w[0] * self.RF + np.dot(w[1:].astype(np.float64),
                                           R_risky.astype(np.float64)))

        # income shocks
        theta, psi = self._draw_income_shocks()

        # transition
        m_next = (Rp / (self.G * psi)) * float(a_post) + theta

        self._m = self.floattype(max(m_next, 0.0))
        self._last_Rp = self.floattype(Rp)
        self._last_w  = w.astype(self.floattype)

        obs = self._get_obs()
        terminated = False
        truncated  = False

        info = self._get_info()
        info.update({
            "consumption_frac": float(alpha),
            "consumption":      float(c),
            "assets_post":      float(a_post),
            "reward":           float(r),
        })

        return obs, float(r), terminated, truncated, info


def crra_utility(consumption: float, gamma: float, floattype=np.float64) -> float:
    c = float(max(consumption, TOL))
    if gamma == 1:
        u = np.log(c)
    else:
        u = (c ** (1 - gamma)) / (1 - gamma)
    return floattype(u)