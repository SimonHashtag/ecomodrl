from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import qmc, norm

from ecomodrl.models.base_model import EconomicModel
from ecomodrl.configs.parameters import PortfolioChoiceParams


class PortfolioChoice(EconomicModel):
    """Normalized buffer-stock model with portfolio choice.

    State
    -----
    state = (m, z_1, ..., z_N)

    where z_j is a persistent predictor for risky asset j.

    Controls
    --------
    c_t:
        consumption

    w_t:
        portfolio weights over assets ordered as

            [safe, risky_1, ..., risky_N]

    Notes
    -----
    This model is solved by a nested EGM/root method, not by BaseEGM.

    The normalized Bellman equation contains the same permanent-income
    normalization kernel as BufferStock:

        v(m,z) = u(c) + beta E[(G psi')^(1-gamma) v(m',z')]

    For RL, `transition()` uses the Esscher-tilted permanent shock and the
    correct constant discount is `beta_eff`.
    """

    PARAMS_CLS = PortfolioChoiceParams

    min_action_share = 1e-6
    max_action_share = 1.0
    logit_cap = 3.0

    def __init__(
        self,
        params=None,
        m_grid_size: int = 61,
        rs_grid_size: int = 61,
        z_grid_size: int = 21,
        m_span: tuple[float, float] = (0.2, 10.0),
        rs_span: tuple[float, float] = (0.0, 10.0),
        z_width: float = 3.0,
        rqmc_nodes: int | None = None,
        rqmc_seed: int | None = None,
    ):
        super().__init__(params)

        p = self.params

        self.R = 1.0 + float(p.r)
        self.G = 1.0 + float(p.g)

        self.sigma_tran = float(p.sigma_tran)
        self.sigma_perm = float(p.sigma_perm)

        # Support both naming conventions during transition.
        self.unemp_prob = float(
            getattr(p, "unemp_prob", getattr(p, "unemployment_prob"))
        )
        self.theta_low = float(p.theta_low)

        self.rho_z = np.atleast_1d(p.rho_z).astype(np.float64)
        self.sigma_z = np.atleast_1d(p.sigma_z).astype(np.float64)
        self.kappa_R = np.atleast_1d(p.kappa_R).astype(np.float64)
        self.xi_R = np.atleast_1d(p.xi_R).astype(np.float64)
        self.sigma_R = np.atleast_1d(p.sigma_R).astype(np.float64)

        self.n_risky = int(len(self.rho_z))
        self.n_assets = self.n_risky + 1

        for name, arr in {
            "sigma_z": self.sigma_z,
            "kappa_R": self.kappa_R,
            "xi_R": self.xi_R,
            "sigma_R": self.sigma_R,
        }.items():
            if len(arr) != self.n_risky:
                raise ValueError(
                    f"{name} must have length {self.n_risky}, got {len(arr)}."
                )

        self.m_grid_size = int(m_grid_size)
        self.rs_grid_size = int(rs_grid_size)
        self.z_grid_size = int(z_grid_size)
        self.m_span = tuple(map(float, m_span))
        self.rs_span = tuple(map(float, rs_span))
        self.z_width = float(z_width)

        self.rqmc_nodes = int(
            rqmc_nodes if rqmc_nodes is not None else getattr(p, "M", 4096)
        )
        self.rqmc_seed = int(
            rqmc_seed if rqmc_seed is not None else getattr(p, "seed", 1996)
        )

        # Mean of employed transitory shock so unconditional E[theta] = 1.
        self.mu_theta = (
            np.log((1.0 - self.unemp_prob * self.theta_low) / (1.0 - self.unemp_prob))
            - 0.5 * self.sigma_tran**2
        )

        self._mean_perm = -0.5 * self.sigma_perm**2
        self._build_tilted_discount()

        self.m_grid = self.make_m_grid(
            points=self.m_grid_size,
            span=self.m_span,
            dense_low=True,
        )
        self.rs_grid = self.make_m_grid(
            points=self.rs_grid_size,
            span=self.rs_span,
            dense_low=True,
        )
        self._z_grids = tuple(self.make_z_grids(points=self.z_grid_size, width=self.z_width))

        # `a_grid` is not used by BaseEGM here, but some generic code expects it.
        self.a_grid = self.rs_grid

        # Useful observation scale.
        self.m_ref = min(max(5.0, self.m_grid[0]), self.m_grid[-1])

        self.build_rqmc(self.rqmc_nodes, self.rqmc_seed)

        # True-distribution permanent growth used by solver and true Bellman.
        self.Gamma_next = (self.G * self._psi_vals).astype(np.float64)
        self.SDF_factor = (1.0 / (self.Gamma_next**self.params.gamma)).astype(np.float64)

        # Equal RQMC weights. Note: unemployment is handled analytically in the
        # SDF moment kernel, not by adding a second shock axis.
        self.weights = np.full(self._M_rqmc, 1.0 / self._M_rqmc)

        # Solved policies. Consumption share has shape (n_m, n_z1, ..., n_zN).
        # Portfolio weights have shape (n_m, n_z1, ..., n_zN, n_assets).
        self.share_policy_grid: np.ndarray | None = None
        self.portfolio_policy_grid: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def exogenous_grids(self) -> tuple[np.ndarray, ...]:
        return self._z_grids

    @property
    def action_dim(self) -> int:
        # Consumption share action + one ALR logit for each risky asset.
        return 1 + self.n_risky

    @property
    def beta_eff(self) -> float:
        return self._beta_eff

    # ------------------------------------------------------------------
    # Normalization / tilted RL representation
    # ------------------------------------------------------------------

    def _build_tilted_discount(self):
        theta = 1.0 - self.params.gamma

        self._mean_perm_tilt = self._mean_perm + theta * self.sigma_perm**2

        M_psi = np.exp(
            0.5 * theta * (theta - 1.0) * self.sigma_perm**2
        )

        self._beta_eff = self.params.beta * self.G**theta * M_psi

    def value_kernel(self):
        """True-distribution normalized Bellman continuation kernel."""
        return (self.G * self._psi_vals) ** (1.0 - self.params.gamma)

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------

    def make_m_grid(
        self,
        points: int,
        span: tuple[float, float],
        dense_low: bool = True,
        p: int = 3,
    ) -> np.ndarray:
        lo, hi = map(float, span)

        if not dense_low:
            return np.linspace(lo, hi, int(points))

        u = np.linspace(0.0, 1.0, int(points))
        return lo + (hi - lo) * (u**p)

    def make_z_grids(self, points: int, width: float = 3.0) -> list[np.ndarray]:
        sd = self.sigma_z / np.sqrt(np.maximum(1e-12, 1.0 - self.rho_z**2))
        return [
            np.linspace(-width * sdi, width * sdi, int(points))
            for sdi in sd
        ]

    def make_multilinear(self, values: np.ndarray, axes=None) -> RegularGridInterpolator:
        if axes is None:
            axes = self.policy_axes

        return RegularGridInterpolator(
            tuple(axes),
            values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    # ------------------------------------------------------------------
    # RQMC nodes for solver / true expectations
    # ------------------------------------------------------------------

    def build_rqmc(self, M: int = 4096, seed: int = 1996) -> None:
        """Build scrambled Sobol nodes with antithetics for true expectations."""
        d = 2 * self.n_risky + 2

        engine = qmc.Sobol(d=d, scramble=True, seed=int(seed))
        m_exp = int(np.ceil(np.log2(max(1, int(M)))))

        U = engine.random_base2(m_exp)
        U = np.vstack([U, 1.0 - U])

        eps = np.finfo(float).eps
        U = np.clip(U, eps, 1.0 - eps)

        X = norm.ppf(U).astype(np.float64)

        n = self.n_risky
        self._M_rqmc = X.shape[0]
        self._eps_nodes = X[:, :n]
        self._eta_nodes = X[:, n : 2 * n]

        nu_nodes = X[:, 2 * n]
        zeta_nodes = X[:, 2 * n + 1]

        self._psi_vals = np.exp(
            self._mean_perm + self.sigma_perm * nu_nodes
        ).astype(np.float64)

        self._theta_vals = np.exp(
            self.mu_theta + self.sigma_tran * zeta_nodes
        ).astype(np.float64)

    # ------------------------------------------------------------------
    # Observation transform for signed predictors
    # ------------------------------------------------------------------

    def obs_from_state(self, state):
        """Model-specific observation transform.

        Unlike productivity states in Brock-Mirman, portfolio predictors z can
        be negative. Therefore we do not log-transform z.
        """
        state = np.asarray(state, dtype=np.float64)
        m = np.maximum(state[..., 0], 1e-8)
        m_obs = np.log(m / max(self.m_ref, 1e-8))
        z_obs = state[..., 1:]
        return np.concatenate([m_obs[..., None], z_obs], axis=-1).astype(np.float64)

    # ------------------------------------------------------------------
    # Policy interpolation
    # ------------------------------------------------------------------

    def _require_solved(self):
        if self.share_policy_grid is None or self.portfolio_policy_grid is None:
            raise RuntimeError("PortfolioChoice policy not solved or assigned.")

    def _policy_coords(self, state):
        state = np.asarray(state, dtype=np.float64)
        scalar = state.ndim == 1
        coords = state[None, :] if scalar else state
        coords = self._clip_policy_coords(coords)
        return coords, scalar

    def share(self, state):
        self._require_solved()

        coords, scalar = self._policy_coords(state)
        interp = self.make_multilinear(self.share_policy_grid)
        out = interp(coords)

        out = np.clip(out, self.min_action_share, 1.0)

        return float(out[0]) if scalar else out

    def consumption(self, state):
        state = np.asarray(state, dtype=np.float64)
        return self.share(state) * state[..., 0]

    def portfolio_weights(self, state):
        self._require_solved()

        coords, scalar = self._policy_coords(state)
        interp = self.make_multilinear(self.portfolio_policy_grid)
        w = np.asarray(interp(coords), dtype=np.float64)

        w = np.clip(w, 0.0, None)
        denom = np.maximum(w.sum(axis=-1, keepdims=True), 1e-14)
        w = w / denom

        return w[0] if scalar else w

    # ------------------------------------------------------------------
    # Action mapping for RL
    # ------------------------------------------------------------------

    def controls_from_action(self, action, state=None) -> dict[str, np.ndarray]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        share = self.min_action_share + 0.5 * (
            self.max_action_share - self.min_action_share
        ) * (action[..., 0] + 1.0)

        logits = self.logit_cap * action[..., 1 : 1 + self.n_risky]
        weights = self.alr_safe_pivot_to_weights(logits)

        out = {
            "share": share,
            "portfolio_weights": weights,
        }

        if state is not None:
            m = np.asarray(state, dtype=np.float64)[..., 0]
            out["consumption"] = share * m

        return out

    @staticmethod
    def alr_safe_pivot_to_weights(logits: np.ndarray) -> np.ndarray:
        """Map risky-asset ALR logits to weights [safe, risky_1, ..., risky_N]."""
        x = np.asarray(logits, dtype=np.float64)
        x = np.clip(x, -PortfolioChoice.logit_cap, PortfolioChoice.logit_cap)

        m = np.maximum(0.0, np.max(x, axis=-1, keepdims=True))
        e = np.exp(x - m)
        pivot = np.exp(-m)

        denom = pivot + np.sum(e, axis=-1, keepdims=True)

        w_safe = pivot / denom
        w_risky = e / denom

        return np.concatenate([w_safe, w_risky], axis=-1)

    # ------------------------------------------------------------------
    # Shock simulation and transitions
    # ------------------------------------------------------------------

    def sample_shocks(self, rng, shape):
        """Raw shocks for RL simulation.

        Last dimension layout:
            eps_z[0:N],
            eta_R[0:N],
            nu_perm,
            zeta_tran,
            unemployment_indicator
        """
        shape = tuple(shape)

        eps_z = rng.normal(size=shape + (self.n_risky,))
        eta_R = rng.normal(size=shape + (self.n_risky,))
        nu_perm = rng.normal(size=shape + (1,))
        zeta_tran = rng.normal(size=shape + (1,))
        unemp = (rng.random(size=shape + (1,)) < self.unemp_prob).astype(np.float64)

        return np.concatenate([eps_z, eta_R, nu_perm, zeta_tran, unemp], axis=-1)

    def transition(self, state, controls: dict[str, Any], shocks):
        """RL transition using tilted permanent-income shocks."""
        state = np.asarray(state, dtype=np.float64)

        m = state[..., 0]
        z = state[..., 1:]

        c = np.asarray(controls["consumption"], dtype=np.float64)
        w = np.asarray(controls["portfolio_weights"], dtype=np.float64)

        n = self.n_risky
        eps_z = shocks[..., :n]
        eta_R = shocks[..., n : 2 * n]
        nu_perm = shocks[..., 2 * n]
        zeta_tran = shocks[..., 2 * n + 1]
        unemp = shocks[..., 2 * n + 2].astype(bool)

        z_next = self.rho_z * z + self.sigma_z * eps_z

        R_risky = np.exp(
            self.kappa_R + self.xi_R * z + self.sigma_R * eta_R
        )

        Rp = w[..., 0] * self.R + np.sum(w[..., 1:] * R_risky, axis=-1)

        # Tilted permanent shock for normalized RL representation.
        psi = np.exp(self._mean_perm_tilt + self.sigma_perm * nu_perm)

        theta_emp = np.exp(self.mu_theta + self.sigma_tran * zeta_tran)
        theta = np.where(unemp, self.theta_low, theta_emp)

        rs = np.maximum(m - c, 0.0)
        m_next = (Rp / (self.G * psi)) * rs + theta

        return np.concatenate([m_next[..., None], z_next], axis=-1)

    def zs_R_next(self, z):
        """True-distribution next predictors and asset returns for RQMC nodes."""
        z = np.asarray(z, dtype=np.float64)

        if self.n_risky == 1 and z.shape == ():
            z = z.reshape(1)

        if z.shape != (self.n_risky,):
            raise ValueError(f"z must have shape ({self.n_risky},), got {z.shape}.")

        z_next = self.rho_z[None, :] * z[None, :] + self.sigma_z[None, :] * self._eps_nodes

        R_risky = np.exp(
            (self.kappa_R + self.xi_R * z)[None, :]
            + self.sigma_R[None, :] * self._eta_nodes
        )

        R_safe = np.full((self._M_rqmc, 1), self.R, dtype=np.float64)

        return z_next, np.concatenate([R_safe, R_risky], axis=1)

    def m_next_from_portfolio_return(self, rs, Rp_next):
        base = (Rp_next / self.Gamma_next) * float(rs)
        return base + self._theta_vals, base + self.theta_low

    # ------------------------------------------------------------------
    # Solver-facing methods that BaseEGM does not use here
    # ------------------------------------------------------------------

    def euler_rhs(self, c_next, **transition_kwargs):
        raise NotImplementedError(
            "PortfolioChoice uses PortfolioChoiceSolver, not BaseEGM.euler_rhs."
        )

    def transition_grid_kwargs(self):
        raise NotImplementedError(
            "PortfolioChoice uses PortfolioChoiceSolver, not BaseEGM."
        )

    def next_state_distribution(self, state, a):
        raise NotImplementedError(
            "Generic next_state_distribution is not implemented for "
            "PortfolioChoice because the transition depends on portfolio weights. "
            "Use portfolio-specific diagnostics or rollout evaluation."
        )

    # ------------------------------------------------------------------
    # Initial and evaluation states
    # ------------------------------------------------------------------

    def sample_initial_state(self, rng):
        m0 = rng.uniform(self.m_grid[0], self.m_grid[-1])

        sd = self.sigma_z / np.sqrt(np.maximum(1e-12, 1.0 - self.rho_z**2))
        z0 = rng.normal(0.0, sd, size=self.n_risky)

        return np.concatenate([[m0], z0]).astype(np.float64)

    def eval_start_states(self, M: int):
        """Evaluation states.

        To avoid tensor-product explosion in high dimension, use an m-grid and
        set predictors to their unconditional mean zero.
        """
        m = self.log_grid(self.m_grid[0], self.m_grid[-1], int(M))
        z = np.zeros((len(m), self.n_risky), dtype=np.float64)
        return np.concatenate([m[:, None], z], axis=1)
