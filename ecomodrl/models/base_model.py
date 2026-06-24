# -----------------------------------------------------------------------------
# Economic models and EGM solvers.
#
# State convention:
#     state = (m, x1, ..., xd)
#
# Policy convention:
#     policy.shape = (n_m, n_x1, ..., n_xd)
#     policy_axes  = (m_grid, x1_grid, ..., xd_grid)
# -----------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

from ecomodrl.analysis import accuracy_metrics
from ecomodrl.configs.parameters import (
    CassKoopmansParams,
    BrockMirmanParams,
    BufferStockParams,
)


# =============================================================================
# Quadrature helpers
# =============================================================================

def gauss_hermite_nodes(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and weights for eps ~ N(0, 1)."""
    x, w = np.polynomial.hermite.hermgauss(int(n_nodes))
    return x * np.sqrt(2.0), w / np.sqrt(np.pi)


def tensor_product_nodes(
    nodes_1d: np.ndarray,
    weights_1d: np.ndarray,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product quadrature for independent standard normal shocks."""
    dim = int(dim)

    if dim == 0:
        return np.empty((1, 0)), np.ones(1)

    node_meshes = np.meshgrid(*([nodes_1d] * dim), indexing="ij")
    weight_meshes = np.meshgrid(*([weights_1d] * dim), indexing="ij")

    nodes = np.stack([x.ravel() for x in node_meshes], axis=-1)
    weights = np.prod(np.stack(weight_meshes, axis=0), axis=0).ravel()

    return nodes, weights


# =============================================================================
# Base economic model
# =============================================================================

class EconomicModel(ABC):
    """Base class for models with state `(m, x1, ..., xd)`."""

    PARAMS_CLS = None

    min_action_share: float = 1e-3
    max_action_share: float = 1.0 - 1e-3

    def __init__(self, params: Any | None = None):
        if params is None:
            if self.PARAMS_CLS is None:
                raise ValueError("Either pass params or set PARAMS_CLS.")
            params = self.PARAMS_CLS()

        self.params = params

        self.weights: np.ndarray | None = None
        self.policy: np.ndarray | None = None

        self.m_grid: np.ndarray | None = None
        self.a_grid: np.ndarray | None = None

        self.m_next: np.ndarray | None = None
        self.x_next: np.ndarray | None = None

        self.m_ref: float = 1.0

    @property
    def exogenous_grids(self) -> tuple[np.ndarray, ...]:
        """Tuple of exogenous state grids. Empty for one-state models."""
        return ()

    @property
    def exogenous_dim(self) -> int:
        return len(self.exogenous_grids)

    @property
    def exogenous_shape(self) -> tuple[int, ...]:
        return tuple(len(g) for g in self.exogenous_grids)

    @property
    def state_dim(self) -> int:
        return 1 + self.exogenous_dim

    @property
    def policy_shape(self) -> tuple[int, ...]:
        return (len(self.m_grid),) + self.exogenous_shape

    @property
    def policy_axes(self) -> tuple[np.ndarray, ...]:
        """Interpolation axes in the same order as the economic state."""
        return (self.m_grid,) + self.exogenous_grids

    @property
    def action_dim(self) -> int:
        """Gym action dimension. Non-portfolio models use one action."""
        return 1

    @staticmethod
    def log_grid(lo: float, hi: float, n: int) -> np.ndarray:
        """Log-spaced grid on [lo, hi]. Requires lo > 0."""
        if lo <= 0:
            raise ValueError("log_grid requires lo > 0.")
        return np.exp(np.linspace(np.log(lo), np.log(hi), int(n)))

    @staticmethod
    def log_grid_from_zero(hi: float, n: int, shift: float = 1.0) -> np.ndarray:
        """Curved grid on [0, hi], dense near zero."""
        return EconomicModel.log_grid(shift, hi + shift, n) - shift

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def utility(self, c):
        c = np.asarray(c)
        if self.params.gamma == 1.0:
            return np.log(c)
        return c ** (1.0 - self.params.gamma) / (1.0 - self.params.gamma)

    def marginal_utility(self, c):
        return np.asarray(c) ** (-self.params.gamma)

    def inv_marginal_utility(self, mu):
        return np.asarray(mu) ** (-1.0 / self.params.gamma)

    # ------------------------------------------------------------------
    # Integration and Bellman normalization
    # ------------------------------------------------------------------

    def integrate(self, x):
        """Integrate over the last axis when quadrature weights are available."""
        if self.weights is None:
            return x
        return np.asarray(x) @ self.weights

    @property
    def beta_eff(self) -> float:
        return self.params.beta

    def value_kernel(self):
        return 1.0

    # ------------------------------------------------------------------
    # Policy evaluation
    # ------------------------------------------------------------------

    def consumption(self, state):
        """Evaluate the solved consumption policy at raw states.

        State convention:
            state = (m, x1, ..., xd)

        Policy convention:
            policy.shape = (n_m, n_x1, ..., n_xd)
            policy_axes  = (m_grid, x1_grid, ..., xd_grid)
        """
        if self.policy is None:
            raise RuntimeError("No policy attached. Solve first or set model.policy.")

        state = np.asarray(state, dtype=float)

        if self.exogenous_dim == 0:
            m = state[..., 0]
            return np.interp(m, self.m_grid, self.policy)

        scalar_input = state.ndim == 1

        coords = state[None, :] if scalar_input else state
        coords = self._clip_policy_coords(coords)

        interp = RegularGridInterpolator(
            self.policy_axes,
            self.policy,
            bounds_error=False,
            fill_value=None,
        )

        values = interp(coords)

        return values[0] if scalar_input else values

    def share(self, state):
        state = np.asarray(state, dtype=float)
        return self.consumption(state) / state[..., 0]

    def _clip_policy_coords(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float).copy()

        for j, grid in enumerate(self.policy_axes):
            coords[..., j] = np.clip(coords[..., j], grid[0], grid[-1])

        return coords

    # ------------------------------------------------------------------
    # EGM hooks
    # ------------------------------------------------------------------

    def initial_policy(self):
        """Initial policy c(m, x) = m."""
        if self.exogenous_dim == 0:
            return self.m_grid.copy()

        m = self.m_grid.reshape((len(self.m_grid),) + (1,) * self.exogenous_dim)
        return np.broadcast_to(m, self.policy_shape).copy()

    def interpolation_kink(self) -> tuple[float, float] | None:
        """Optional EGM map-back kink. BufferStock returns `(0, 0)`."""
        return None

    @abstractmethod
    def euler_rhs(self, c_next, **transition_kwargs):
        ...

    @abstractmethod
    def transition_grid_kwargs(self) -> dict[str, np.ndarray]:
        """Return transition arrays for EGM grid evaluation. Used in `egm_step`."""
        ...

    @abstractmethod
    def next_state_distribution(self, state, a):
        """Return next-state distribution given current state and action."""
        ...

    # ------------------------------------------------------------------
    # RL hooks
    # ------------------------------------------------------------------

    def controls_from_action(self, action, state=None) -> dict[str, np.ndarray]:
        """Map Gym action in [-1, 1] to consumption.
        """
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        share = self.min_action_share + 0.5 * (
            self.max_action_share - self.min_action_share
        ) * (action[..., 0] + 1.0)

        out = {"share": share}

        if state is not None:
            m = np.asarray(state)[..., 0]
            out["consumption"] = share * m

        return out

    @abstractmethod
    def sample_shocks(self, rng, shape):
        ...

    @abstractmethod
    def transition(self, state, controls: dict[str, Any], shocks):
        ...

    @abstractmethod
    def sample_initial_state(self, rng):
        ...

    @abstractmethod
    def eval_start_states(self, M: int) -> np.ndarray:
        ...


# =============================================================================
# Cass-Koopmans
# =============================================================================

class CassKoopmans(EconomicModel):
    """Deterministic Cass-Koopmans growth model."""

    PARAMS_CLS = CassKoopmansParams

    k_min_factor = 0.2
    k_max_factor = 5.0

    def __init__(self, params=None, grid_size: int = 5000):
        super().__init__(params)

        self.grid_size = int(grid_size)

        self.m_ref = self.cash_on_hand(self.k_steady, 1.0)

        self.a_grid = self._build_a_grid(self.grid_size)

        self.m_grid = self.log_grid(
            self.cash_on_hand(self.a_grid[0]),
            self.cash_on_hand(self.a_grid[-1]),
            self.grid_size,
        )

        self.m_next = self.cash_on_hand(self.a_grid)
        self.x_next = None
        self.R_next = self.gross_interest_rate(self.a_grid)

    @property
    def k_steady(self):
        p = self.params
        return ((1.0 / p.beta - 1.0 + p.delta) / (p.A * p.alpha)) ** (
            1.0 / (p.alpha - 1.0)
        )

    def production(self, k, z=1.0):
        return z * self.params.A * np.asarray(k) ** self.params.alpha

    def cash_on_hand(self, k, z=1.0):
        return self.production(k, z) + (1.0 - self.params.delta) * np.asarray(k)

    def gross_interest_rate(self, a, z=1.0):
        return (
            z
            * self.params.A
            * self.params.alpha
            * np.asarray(a) ** (self.params.alpha - 1.0)
            + (1.0 - self.params.delta)
        )

    def _build_a_grid(self, n):
        return self.log_grid(
            self.k_min_factor * self.k_steady,
            self.k_max_factor * self.k_steady,
            n,
        )

    def transition_grid_kwargs(self):
        return {"R_next": self.R_next}

    def euler_rhs(self, c_next, R_next):
        return self.params.beta * self.integrate(self.marginal_utility(c_next) * R_next)

    def next_state_distribution(self, state, a):
        m_next = self.cash_on_hand(a)
        return np.asarray(m_next)[..., None], {"R_next": self.gross_interest_rate(a)}

    def sample_shocks(self, rng, shape):
        return None

    def transition(self, state, controls, shocks):
        c = controls["consumption"]
        a = np.asarray(state)[..., 0] - c
        return self.cash_on_hand(a)[..., None]

    def sample_initial_state(self, rng):
        return np.array([rng.choice(self.m_grid)], dtype=float)

    def eval_start_states(self, M):
        return self.log_grid(self.m_grid[0], self.m_grid[-1], M)[:, None]


# =============================================================================
# Brock-Mirman with AR(1) technology innovation
# =============================================================================

class BrockMirman(CassKoopmans):
    """Stochastic Brock-Mirman growth model with one AR(1) technology shock.

    State:
        state = (m, z)

    Production:
        y = z A k^alpha
    """

    PARAMS_CLS = BrockMirmanParams

    n_gh_nodes = 7

    eval_z_quantiles = (
        0.05, 0.15, 0.25, 0.35, 0.45,
        0.55, 0.65, 0.75, 0.85, 0.95,
    )

    def __init__(
        self,
        params=None,
        m_grid_size: int = 500,
        a_grid_size: int = 500,
        z_grid_size: int = 100,
    ):
        EconomicModel.__init__(self, params)

        self.m_grid_size = int(m_grid_size)
        self.a_grid_size = int(a_grid_size)
        self.z_grid_size = int(z_grid_size)

        self.m_ref = self.cash_on_hand(self.k_steady, 1.0)

        x, w = gauss_hermite_nodes(self.n_gh_nodes)
        self.shock_nodes = self.params.sigma * x
        self.weights = w

        self.z_grid = self._build_z_grid()
        self.a_grid = self._build_a_grid(self.a_grid_size)

        self._build_transition_arrays()

    @property
    def exogenous_grids(self):
        return (self.z_grid,)

    def _build_z_grid(self):
        p = self.params

        if p.rho == 0.0:
            sigma_z = p.sigma
        else:
            sigma_z = p.sigma / np.sqrt(1.0 - p.rho ** 2)

        return self.log_grid(
            np.exp(-3.0 * sigma_z),
            np.exp(3.0 * sigma_z),
            self.z_grid_size,
        )

    def _z_next(self, z, eps):
        return np.exp(self.params.rho * np.log(z) + eps)

    def _build_transition_arrays(self):
        """Build transition arrays."""
        a = self.a_grid[:, None, None]
        z = self.z_grid[None, :, None]
        eps = self.shock_nodes[None, None, :]

        z_next = self._z_next(z, eps)

        self.m_next = self.cash_on_hand(a, z_next)
        self.R_next = self.gross_interest_rate(a, z_next)
        self.x_next = np.broadcast_to(z_next, self.m_next.shape)[..., None]

        m_lo = self.cash_on_hand(self.a_grid[0], self.z_grid[0])
        m_hi = self.cash_on_hand(self.a_grid[-1], self.z_grid[-1])

        self.m_grid = self.log_grid(m_lo, m_hi, self.m_grid_size)

    def transition_grid_kwargs(self):
        return {"R_next": self.R_next}

    def next_state_distribution(self, state, a):
        state = np.asarray(state, dtype=float)

        z = state[..., 1]
        z_next = self._z_next(z[..., None], self.shock_nodes)

        a_bc = np.asarray(a)[..., None]
        m_next = self.cash_on_hand(a_bc, z_next)

        next_state = np.stack([m_next, z_next], axis=-1)

        return next_state, {
            "R_next": self.gross_interest_rate(a_bc, z_next),
        }

    def sample_shocks(self, rng, shape):
        return rng.normal(0.0, self.params.sigma, size=shape)

    def transition(self, state, controls, shocks):
        state = np.asarray(state, dtype=float)

        c = controls["consumption"]
        a = state[..., 0] - c

        z_next = self._z_next(state[..., 1], shocks)
        m_next = self.cash_on_hand(a, z_next)

        return np.stack([m_next, z_next], axis=-1)

    def _stationary_z_anchors(self):
        p = self.params

        if p.rho == 0.0:
            sigma_z = p.sigma
        else:
            sigma_z = p.sigma / np.sqrt(1.0 - p.rho ** 2)

        q = np.asarray(self.eval_z_quantiles)
        return np.exp(norm.ppf(q) * sigma_z)

    def sample_initial_state(self, rng):
        m0 = rng.choice(self.m_grid)
        z0 = rng.choice(self.z_grid)
        return np.array([m0, z0], dtype=float)

    def eval_start_states(self, M):
        z_anchors = self._stationary_z_anchors()

        M_z = len(z_anchors)
        M_m = max(1, int(M) // M_z)

        m_anchors = self.log_grid(self.m_grid[0], self.m_grid[-1], M_m)

        m_mesh, z_mesh = np.meshgrid(m_anchors, z_anchors, indexing="ij")

        return np.stack([m_mesh.ravel(), z_mesh.ravel()], axis=-1)


# =============================================================================
# Buffer-stock model
# =============================================================================

class BufferStock(EconomicModel):
    """Normalized buffer-stock model with an occasionally binding constraint.

    State:
        state = (m,)

    Grid initialization:
        Uses a policy-independent, theory-anchored scale based on the
        growth-impatience gap and income-risk/prudence.
    """

    PARAMS_CLS = BufferStockParams

    n_gh_nodes = 7

    min_m_max = 20.0
    grid_upper_factor = 5.0
    impatience_gap_floor = 1e-3

    max_action_share = 1.0

    def __init__(
        self,
        params=None,
        m_grid_size: int = 5000,
        m_max: float | None = None,
    ):
        super().__init__(params)

        self.m_grid_size = int(m_grid_size)

        self.R = 1.0 + self.params.r
        self.G = 1.0 + self.params.g

        self._build_shocks()
        self._build_tilted_discount()

        if m_max is None:
            self.m_ref, m_max = self._buffer_stock_grid_scale()
        else:
            m_max = float(m_max)
            self.m_ref = max(1.0, m_max / self.grid_upper_factor)

        self.m_grid = self.log_grid_from_zero(m_max, self.m_grid_size)
        self.a_grid = self.m_grid.copy()

        self.m_next, self.perm_next = self._transition_arrays(self.a_grid)
        self.x_next = None

    @property
    def beta_eff(self):
        return self._beta_eff

    def _build_shocks(self):
        x, w = gauss_hermite_nodes(self.n_gh_nodes)
        p = self.params

        self._mean_perm = -0.5 * p.sigma_perm ** 2

        self._mean_tran = (
            np.log((1.0 - p.unemp_prob * p.theta_low) / (1.0 - p.unemp_prob))
            - 0.5 * p.sigma_tran ** 2
        )

        perm_nodes = np.exp(self._mean_perm + p.sigma_perm * x)
        tran_nodes = np.exp(self._mean_tran + p.sigma_tran * x)

        perm_unemp = perm_nodes
        tran_unemp = np.full_like(perm_nodes, p.theta_low)
        weights_unemp = p.unemp_prob * w

        perm_mesh, tran_mesh = np.meshgrid(perm_nodes, tran_nodes, indexing="ij")

        perm_emp = perm_mesh.ravel()
        tran_emp = tran_mesh.ravel()
        weights_emp = (1.0 - p.unemp_prob) * np.outer(w, w).ravel()

        self.perm_shocks = np.concatenate([perm_unemp, perm_emp])
        self.tran_shocks = np.concatenate([tran_unemp, tran_emp])
        self.weights = np.concatenate([weights_unemp, weights_emp])

        self.inv_perm_e = float(np.dot(1.0 / self.perm_shocks, self.weights))
        self.tran_e = float(np.dot(self.tran_shocks, self.weights))

    def _build_tilted_discount(self):
        theta = 1.0 - self.params.gamma

        self._mean_perm_tilt = (
            self._mean_perm + theta * self.params.sigma_perm ** 2
        )

        M_psi = np.exp(
            0.5 * theta * (theta - 1.0) * self.params.sigma_perm ** 2
        )

        self._beta_eff = self.params.beta * self.G ** theta * M_psi

    def _buffer_stock_grid_scale(self) -> tuple[float, float]:
        """Theory-anchored buffer-stock grid scale.

        This function chooses a fixed m-grid before solving the model.

        It does not compute the exact stochastic target m*. Instead it builds a
        deterministic proxy with the correct buffer-stock comparative statics.

        Main object (perfect-foresight consumption growth factor implied by the
        Euler equation):
            thorn = C_{t+1} / C_t = (beta * R) ** (1 / gamma)
        The Growth Impatience Condition is: thorn / G < 1.

        The precautionary motive is increasing in patience, prudence and income risk.
        """
        p = self.params

        thorn = (p.beta * self.R) ** (1.0 / p.gamma)

        growth_impatience_gap = max(1.0 - thorn / self.G, self.impatience_gap_floor)
        unemployment_risk = p.unemp_prob * (1.0 - p.theta_low) ** 2
        income_risk = p.sigma_perm ** 2 + p.sigma_tran ** 2 + unemployment_risk
        prudence_risk_term = 0.5 * p.gamma * (p.gamma + 1.0) * income_risk

        precautionary_assets = prudence_risk_term / growth_impatience_gap

        m_ref = max(1.0, 1.0 + precautionary_assets)

        m_max = max(
            self.min_m_max,
            self.grid_upper_factor * m_ref,
        )

        return float(m_ref), float(m_max)

    def interpolation_kink(self):
        return 0.0, 0.0

    def _transition_arrays(self, a):
        a = np.asarray(a)[..., None]

        m_next = (self.R / (self.G * self.perm_shocks)) * a + self.tran_shocks
        perm_next = np.broadcast_to(self.perm_shocks, m_next.shape)

        return m_next, perm_next

    def transition_grid_kwargs(self):
        return {"perm_next": self.perm_next}

    def euler_rhs(self, c_next, perm_next):
        return (
            self.R
            * self.params.beta
            * self.G ** (-self.params.gamma)
            * self.integrate(self.marginal_utility(c_next * perm_next))
        )

    def next_state_distribution(self, state, a):
        m_next, perm_next = self._transition_arrays(a)
        return m_next[..., None], {"perm_next": perm_next}

    def sample_shocks(self, rng, shape):
        p = self.params

        return np.stack(
            [
                rng.normal(size=shape),
                rng.normal(size=shape),
                (rng.random(size=shape) < p.unemp_prob).astype(float),
            ],
            axis=-1,
        )

    def value_kernel(self):
        return (self.G * self.perm_shocks) ** (1.0 - self.params.gamma)

    def transition(self, state, controls, shocks):
        c = controls["consumption"]
        a = np.asarray(state)[..., 0] - c

        psi = np.exp(
            self._mean_perm_tilt + self.params.sigma_perm * shocks[..., 0]
        )

        theta_emp = np.exp(
            self._mean_tran + self.params.sigma_tran * shocks[..., 1]
        )

        theta = np.where(
            shocks[..., 2].astype(bool),
            self.params.theta_low,
            theta_emp,
        )

        m_next = (self.R / (self.G * psi)) * a + theta

        return m_next[..., None]

    def sample_initial_state(self, rng):
        positive = self.m_grid[self.m_grid > 0.0]
        return np.array([rng.choice(positive)], dtype=float)

    def eval_start_states(self, M):
        positive = self.m_grid[self.m_grid > 0.0]
        return self.log_grid(positive[0], positive[-1], M)[:, None]


# =============================================================================
# Generic EGM solver
# =============================================================================

class BaseEGM:
    """Generic EGM solver for one continuous control: consumption."""

    def __init__(
        self,
        econ_model: EconomicModel,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        self.model = econ_model
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.policy: np.ndarray | None = None
        self.solve_time: float | None = None

    def solve(self, verbose: bool = True):
        start = time.time()

        policy_old = self.model.initial_policy()
        self.policy = self._egm_loop(policy_old, verbose)

        self.model.policy = self.policy
        self.solve_time = time.time() - start

        if verbose:
            print(f"EGM finished in {self.solve_time:.2f} seconds.")

        return self.policy

    def _egm_loop(self, policy_old, verbose):
        diff = np.inf

        for iteration in range(1, self.max_iter + 1):
            policy_new = self._egm_step(policy_old)

            diff = float(np.max(np.abs(policy_new - policy_old)))

            if diff < self.tol:
                if verbose:
                    print(
                        f"\nConverged after {iteration} iterations "
                        f"(sup-norm diff = {diff:.2e})"
                    )
                return policy_new

            policy_old = policy_new

            if verbose:
                print(
                    f"Iteration {iteration}: sup-norm diff = {diff:.2e}",
                    end="\r",
                    flush=True,
                )

        if verbose:
            print(
                f"\nDid not converge after {self.max_iter} iterations "
                f"(sup-norm diff = {diff:.2e})"
            )

        return policy_old

    def _egm_step(self, policy_old):
        c_next = self._interp_policy(
            policy_old,
            self.model.m_next,
            self.model.x_next,
        )

        rhs = self.model.euler_rhs(
            c_next,
            **self.model.transition_grid_kwargs(),
        )

        c_endo = self.model.inv_marginal_utility(rhs)

        a = self._a_grid_for_shape(c_endo.shape)
        m_endo = a + c_endo

        return self._interp_back(m_endo, c_endo)

    def _a_grid_for_shape(self, target_shape):
        if self.model.exogenous_dim == 0:
            return self.model.a_grid

        return self.model.a_grid.reshape(
            (len(self.model.a_grid),) + (1,) * self.model.exogenous_dim
        )

    def _interp_policy(self, policy, m_next, x_next):
        if self.model.exogenous_dim == 0:
            out = np.interp(
                np.ravel(m_next),
                self.model.m_grid,
                policy,
            )
            return out.reshape(np.shape(m_next))

        interp = RegularGridInterpolator(
            self.model.policy_axes,
            policy,
            bounds_error=False,
            fill_value=None,
        )

        m_next = np.asarray(m_next)
        x_next = np.asarray(x_next)

        lead_shape = np.broadcast_shapes(m_next.shape, x_next.shape[:-1])

        m_coord = np.broadcast_to(m_next, lead_shape)[..., None]
        x_coord = np.broadcast_to(x_next, lead_shape + (self.model.exogenous_dim,))

        coords = np.concatenate([m_coord, x_coord], axis=-1)
        coords = self.model._clip_policy_coords(coords)

        return interp(coords)

    def _interp_back(self, m_endo, c_endo):
        if self.model.exogenous_dim == 0:
            return self._interp_one(m_endo, c_endo)

        policy = np.empty(self.model.policy_shape)

        for x_idx in np.ndindex(self.model.exogenous_shape):
            idx = (slice(None),) + x_idx
            policy[idx] = self._interp_one(m_endo[idx], c_endo[idx])

        return policy

    def _interp_one(self, m_endo, c_endo):
        m_endo = np.asarray(m_endo, dtype=float)
        c_endo = np.asarray(c_endo, dtype=float)

        kink = self.model.interpolation_kink()

        if kink is not None:
            m_kink, c_kink = kink
            m_endo = np.concatenate([[m_kink], m_endo])
            c_endo = np.concatenate([[c_kink], c_endo])

        order = np.argsort(m_endo)
        m_sorted = m_endo[order]
        c_sorted = c_endo[order]

        m_unique, unique_idx = np.unique(m_sorted, return_index=True)
        c_unique = c_sorted[unique_idx]

        return np.interp(self.model.m_grid, m_unique, c_unique)


CassKoopmansEGM = BaseEGM
BrockMirmanEGM = BaseEGM
BufferStockEGM = BaseEGM


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    from ecomodrl.analysis.policy_interface import EGMPolicy

    specs = [
        ("Cass-Koopmans", CassKoopmans(CassKoopmansParams())),
        ("Brock-Mirman", BrockMirman(BrockMirmanParams())),
        ("Buffer stock", BufferStock(BufferStockParams())),
    ]

    solved = []

    for label, model in specs:
        print(label)
        solver = BaseEGM(model)
        solver.solve(verbose=True)
        egm_policy = EGMPolicy(model)
        solved.append((label, model, solver, egm_policy))

    header = f"{'Model':<35} {'Mean log10|Euler error|':>24} {'Solve time (s)':>16}"
    print(header)
    print("=" * len(header))

    for label, model, solver, egm_policy in solved:
        ee = accuracy_metrics.euler_errors(model, egm_policy)
        mean_err = np.mean(ee) if ee.size else np.nan

        print(f"{label:<35} {mean_err:>24.4f} {solver.solve_time:>16.2f}")

    print("=" * len(header))