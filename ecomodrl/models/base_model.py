# -----------------------------------------------------------------------------
# Base Economic model 
# (includes all necessary logic for EGM and RL training and evaluation)
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
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator



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
