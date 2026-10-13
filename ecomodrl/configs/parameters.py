from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

@dataclass(frozen=True)
class CassKoopmansParams:
    """
    Canonical, frozen configuration for the deterministic Cass-Koopmans growth model.

    Attributes
    ----------
    gamma : float
        Coefficient of relative risk aversion σ.
        Must be > 0. The σ=1 case corresponds to log utility as a limit.
    beta : float
        Subjective discount factor β. Must lie in (0, 1).
    delta : float
        Capital depreciation rate δ per period in [0, 1].
    alpha : float
        Capital share α (output elasticity of capital) in (0, 1).
    A : float
        Productivity / technology level A > 0 in y = A * k^α.
    """
    gamma: float = 2.0
    beta: float = 0.96
    delta: float = 0.03
    alpha: float = 0.36
    A: float = 1.0

    # -------- Validation --------
    def __post_init__(self):
        if not (self.gamma > 0):
            raise ValueError("crra (σ) must be > 0")
        if not (0 < self.beta < 1):
            raise ValueError("beta (β) must be in (0, 1)")
        if not (0 <= self.delta <= 1):
            raise ValueError("delta (δ) must be in [0, 1]")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha (α) must be in (0, 1)")
        if not (self.A > 0):
            raise ValueError("A must be > 0")


@dataclass(frozen=True)
class BrockMirmanParams:
    """
    Canonical, frozen configuration for the (stochastic) Brock–Mirman model.

    Technology:
        y_t = z_t * k_t^α ,   log z_{t+1} = ρ log z_t + σ ε_{t+1},   ε ~ N(0,1)

    Attributes
    ----------
    gamma : float
        Coefficient of relative risk aversion σ (CRRA). Must be > 0.
        The σ=1 case corresponds to log utility as a limit.
    beta : float
        Subjective discount factor β. Must lie in (0, 1).
    delta : float
        Capital depreciation rate δ per period in [0, 1].
        (In the classic Brock–Mirman setup δ=1 is common.)
    alpha : float
        Capital share α (output elasticity of capital) in (0, 1).
    rho : float
        AR(1) persistence ρ of log z. Must satisfy |ρ| < 1.
    sigma : float
        Innovation std. dev. σ of log z. Must be ≥ 0.
    """
    gamma: float = 2.0
    beta: float = 0.96
    delta: float = 0.03
    alpha: float = 0.36
    A: float = 1.0
    rho: float = 0.95
    sigma: float = 0.03

    # -------- Validation --------
    def __post_init__(self):
        if not (self.gamma > 0):
            raise ValueError("gamma (σ) must be > 0")
        if not (0 < self.beta < 1):
            raise ValueError("beta (β) must be in (0, 1)")
        if not (0 <= self.delta <= 1):
            raise ValueError("delta (δ) must be in [0, 1]")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha (α) must be in (0, 1)")
        if not (self.A > 0):
            raise ValueError("A must be > 0")
        if not (abs(self.rho) < 1):
            raise ValueError("rho (ρ) must satisfy |ρ| < 1")
        if not (self.sigma >= 0):
            raise ValueError("sigma (σ, shock std) must be ≥ 0")


@dataclass(frozen=True)
class BufferStockParams:
    """
    Canonical configuration for the (normalized) Buffer-Stock Savings model (Carroll, 1997).

    Variables are normalized by current permanent labor income. Dynamics (normalized):
        m_{t+1} = (R / (G * ψ_{t+1})) * (m_t - c_t) + θ_{t+1}

    Attributes
    ----------
    gamma : float
        CRRA coefficient σ (> 0). σ=1 corresponds to log utility as a limit.
    beta : float
        Subjective discount factor β in (0, 1).
    r : float
        Net interest rate r per period (R = 1 + r). Must satisfy r > -1.
    g : float
        Growth rate of permanent income g per period (G = 1 + g). Must satisfy g > -1.
    sigma_tran : float
        Std. dev. of transitory shock in logs (σ_θ) ≥ 0.
    sigma_perm : float
        Std. dev. of permanent shock in logs (σ_ψ) ≥ 0.
    unemp_prob : float
        Probability p of unemployment (point mass at θ_low). Must be in [0, 1).
    theta_low : float
        Transitory income when unemployed θ_low, normalized. Must lie in [0, 1].
    gh_order : int
        Order of Gauss-Hermite quadrature for approximating lognormal shocks.
    """
    gamma: float = 2.0
    beta: float = 0.96
    r: float = 0.04
    g: float = 0.03
    sigma_tran: float = 0.1
    sigma_perm: float = 0.1
    unemp_prob: float = 0.05
    theta_low: float = 0.15
    gh_order: int = 7

    # -------- Validation --------
    def __post_init__(self):
        if not (self.gamma > 0):
            raise ValueError("gamma (σ) must be > 0")
        if not (0 < self.beta < 1):
            raise ValueError("beta (β) must be in (0, 1)")
        if not (self.r > -1.0):
            raise ValueError("r must be > -1 so that R = 1 + r > 0")
        if not (self.g > -1.0):
            raise ValueError("g must be > -1 so that G = 1 + g > 0")
        if not (self.sigma_tran >= 0):
            raise ValueError("sigma_tran (transitory log-std) must be ≥ 0")
        if not (self.sigma_perm >= 0):
            raise ValueError("sigma_perm (permanent log-std) must be ≥ 0")
        if not (0 <= self.unemp_prob < 1):
            raise ValueError("unemp_prob must be in [0, 1)")
        if not (0 <= self.theta_low <= 1):
            raise ValueError("theta_low must lie in [0, 1]")
        if not (self.gh_order >= 1 and isinstance(self.gh_order, int)):
            raise ValueError("gh_order must be a positive integer (≥ 1)")


@dataclass(frozen=True)
class PortfolioChoiceParams:
    """
    Params for the normalized buffer-stock model with portfolio choice.

    Assets:
      - One risk-free asset with net rate r (R_f = 1 + r).
      - N risky assets with predictor-driven log-returns:
            z_{j,t+1} = rho_z[j] * z_{j,t} + sigma_z[j] * eps_{j,t+1}
            ln R_{j,t+1} = kappa_R[j] + xi_R[j] * z_{j,t} + sigma_R[j] * eta_{j,t+1}
        eps, eta ~ N(0,1), iid across time and assets.

    Income shocks (normalized):
      - Permanent ψ ~ lognormal with mean one (handled via log-mean shift).
      - Transitory θ: unemployment mass at theta_low with prob unemployment_prob,
        else lognormal with mean one (shift computed in the env).
    """

    # ---- preferences & growth ----
    gamma: float = 2.0
    beta: float = 0.96
    r: float = 0.04  # risk-free net rate
    g: float = 0.03


    # ---- income shocks ----
    sigma_tran: float = 0.10
    sigma_perm: float = 0.10
    unemployment_prob: float = 0.05
    theta_low: float = 0.15

    # ---- assets & predictors ----
    rho_z: Union[Sequence[float], float] = (0.90,)         # AR(1) coefficient(s) |rho|<1
    sigma_z: Union[Sequence[float], float] = (0.08,)       # predictor innovation std(s) >= 0
    kappa_R: Union[Sequence[float], float] = (0.04,)       # intercept(s) in ln R
    xi_R: Union[Sequence[float], float] = (0.4,)           # slope(s) on z in ln R
    sigma_R: Union[Sequence[float], float] = (0.18,)       # std(s) in ln R innovations >= 0

    n_risky: Optional[int] = 1

    # -------- Validation & coercion --------
    def __post_init__(self):
        # --- scalar checks ---
        if not (self.gamma > 0):
            raise ValueError("gamma must be > 0")
        if not (0 < self.beta < 1):
            raise ValueError("beta must be in (0, 1)")
        if not (self.r > -1.0):
            raise ValueError("r must be > -1 (so 1 + r > 0)")
        if not (self.g > -1.0):
            raise ValueError("g must be > -1 (so 1 + g > 0)")
        if not (self.sigma_tran >= 0):
            raise ValueError("sigma_tran must be ≥ 0")
        if not (self.sigma_perm >= 0):
            raise ValueError("sigma_perm must be ≥ 0")
        if not (0 <= self.unemployment_prob < 1):
            raise ValueError("unemployment_prob must be in [0, 1)")

        # --- helper: coerce to 1D float array of length n_risky ---
        def to_1d(val: Union[Sequence[float], float], name: str) -> np.ndarray:
            arr = np.asarray(val, dtype=float).ravel()
            if arr.size == 0:
                raise ValueError(f"{name} cannot be empty")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
            return arr

        rho_z   = to_1d(self.rho_z,   "rho_z")
        sigma_z = to_1d(self.sigma_z, "sigma_z")
        kappa_R = to_1d(self.kappa_R, "kappa_R")
        xi_R    = to_1d(self.xi_R,    "xi_R")
        sigma_R = to_1d(self.sigma_R, "sigma_R")

        # --- infer K and assert equal dimensions ---
        lengths = {rho_z.size, sigma_z.size, kappa_R.size, xi_R.size, sigma_R.size}
        if len(lengths) != 1:
            raise ValueError(
                f"Per-asset arrays must have the same length; got sizes {sorted(lengths)}"
            )
        K = int(rho_z.size)

        if self.n_risky is not None and self.n_risky != K:
            raise ValueError(f"n_risky={self.n_risky} does not match inferred length K={K}")

        # --- per-asset constraints ---
        if not np.all(np.abs(rho_z) < 1.0):
            raise ValueError("All |rho_z[j]| must be < 1 for AR(1) stationarity")
        if not np.all(sigma_z >= 0):
            raise ValueError("All sigma_z[j] must be ≥ 0")
        if not np.all(sigma_R >= 0):
            raise ValueError("All sigma_R[j] must be ≥ 0")

        # --- write back coerced arrays + derived n_risky (dataclass is frozen) ---
        object.__setattr__(self, "rho_z", rho_z)
        object.__setattr__(self, "sigma_z", sigma_z)
        object.__setattr__(self, "kappa_R", kappa_R)
        object.__setattr__(self, "xi_R", xi_R)
        object.__setattr__(self, "sigma_R", sigma_R)
        object.__setattr__(self, "n_risky", K)

@dataclass(frozen=True)
class DurableGoodsParams:
    """
    Canonical configuration for the durable-goods model of Fella (2014), Table 1.

    The household chooses a durable stock d' from a finite set D and next-period
    assets a' (wealth plus borrowing capacity, a = w + xi d):

        max E sum_t beta^t [theta log c_t + (1 - theta) log(kappa d_{t+1})]
        c + a' = z(d'; a, d, y) = y + (1 + r) a - (1 - xi)(d' - d) - h(d, d') - r xi d
        h(d, d') = 1{d' != d} phi d',     a' >= 0,     d' in D

    with log income y = p + e, p a persistent AR(1) and e iid, both discretized by
    Tauchen (1986).

    Attributes
    ----------
    gamma : float
        Kept at 1.0 (log utility); used by the CEV formula.
    beta : float
        Discount factor β in (0, 1).
    r : float
        Net interest rate r (> -1).
    theta : float
        Weight of non-durable consumption in utility, in (0, 1).
    kappa : float
        Service flow per unit of durable stock (> 0).
    phi : float
        Fixed adjustment cost: pay phi * d' whenever d' != d (≥ 0).
    xi : float
        Collateralizable share of the durable, in [0, 1).
    rho, sigma_eta : float
        Persistence and innovation std. dev. of the AR(1) component p.
    sigma_eps : float
        Std. dev. of the iid component e.
    n_p, n_e : int
        Tauchen nodes for p and e.
    tauchen_width : float
        Tauchen grids cover +/- tauchen_width unconditional std. devs.
    n_d : int
        Number of durable levels (equally spaced on [d_min, d_max]).
    d_min, d_max : float
        Durable grid bounds (d_min > 0 keeps log(kappa d') finite).
    """
    gamma: float = 1.0
    beta: float = 0.93
    r: float = 0.06
    theta: float = 0.77
    kappa: float = 0.075
    phi: float = 0.06
    xi: float = 0.8
    rho: float = 0.977
    sigma_eta: float = 0.024
    sigma_eps: float = 0.063
    n_p: int = 7
    n_e: int = 7
    tauchen_width: float = 3.0
    n_d: int = 7
    d_min: float = 0.01
    d_max: float = 10.0

    # -------- Validation --------
    def __post_init__(self):
        if self.gamma != 1.0:
            raise ValueError("gamma must be 1.0 (the model has log utility)")
        if not (0 < self.beta < 1):
            raise ValueError("beta (β) must be in (0, 1)")
        if not (self.r > -1.0):
            raise ValueError("r must be > -1")
        if not (0 < self.theta < 1):
            raise ValueError("theta must be in (0, 1)")
        if not (self.kappa > 0):
            raise ValueError("kappa must be > 0")
        if not (self.phi >= 0):
            raise ValueError("phi must be ≥ 0")
        if not (0 <= self.xi < 1):
            raise ValueError("xi must be in [0, 1)")
        if not (abs(self.rho) < 1):
            raise ValueError("|rho| must be < 1")
        if not (self.sigma_eta > 0 and self.sigma_eps > 0):
            raise ValueError("sigma_eta and sigma_eps must be > 0")
        if not (self.n_p >= 2 and self.n_e >= 2 and self.n_d >= 2):
            raise ValueError("n_p, n_e and n_d must be ≥ 2")
        if not (self.tauchen_width > 0):
            raise ValueError("tauchen_width must be > 0")
        if not (0 < self.d_min < self.d_max):
            raise ValueError("durable grid needs 0 < d_min < d_max")
