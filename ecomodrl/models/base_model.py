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

from ecomodrl.models.array_ops import ops as _ops



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

    def model_spec_kwargs(self) -> dict[str, Any]:
        """Behavior-affecting constructor kwargs stored in ModelSpec."""
        return {}

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
        """CRRA utility. Accepts NumPy arrays or Torch tensors (see array_ops.py)."""
        xp = _ops(c)
        c = xp.asarray(c)
        if self.params.gamma == 1.0:
            return xp.log(c)
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

        Accepts numpy arrays or torch tensors (see array_ops.py); the torch path
        is differentiable in `action`.
        """
        xp = _ops(action)
        action = xp.clip(xp.asarray(action, dtype=float), -1.0, 1.0)

        share = self.min_action_share + 0.5 * (
            self.max_action_share - self.min_action_share
        ) * (action[..., 0] + 1.0)

        out = {"share": share}

        if state is not None:
            m = xp.asarray(state)[..., 0]
            out["consumption"] = share * m

        return out

    def obs_from_state(self, state):
        """Map raw state to the default log Gym observation.

        Accepts NumPy arrays or Torch tensors. Models with non-positive
        exogenous states override this together with state_from_obs.
        """
        xp = _ops(state)
        state = xp.asarray(state, dtype=np.float64)

        m = xp.maximum(state[..., 0], 1e-8)
        m_obs = xp.log(m / max(float(self.m_ref), 1e-8))

        if self.exogenous_dim == 0:
            return m_obs[..., None]

        x_obs = xp.log(xp.maximum(state[..., 1:], 1e-8))
        return xp.concatenate([m_obs[..., None], x_obs], axis=-1)

    def state_from_obs(self, obs):
        """Inverse of obs_from_state for NumPy arrays or Torch tensors.

        obs = (log(m / m_ref), log(x1), ..., log(xd)); the 1e-8 floor is the
        model's observation floor.
        """
        xp = _ops(obs)
        obs = xp.asarray(obs, dtype=np.float64)

        m = xp.exp(obs[..., 0]) * max(float(self.m_ref), 1e-8)

        if self.exogenous_dim == 0:
            return m[..., None]

        return xp.concatenate([m[..., None], xp.exp(obs[..., 1:])], axis=-1)

    # ------------------------------------------------------------------
    # Post-decision hooks (ecomodrl/algorithms/post_decision)
    # ------------------------------------------------------------------

    @property
    def post_decision_dim(self) -> int:
        """Dimension of `post_decision_features`."""
        raise NotImplementedError(f"{type(self).__name__} does not implement post_decision_dim.")

    def post_decision_features(self, state, controls: dict[str, Any]):
        """Features the continuation network W is evaluated on, shape (..., post_decision_dim).

        Accepts numpy arrays or torch tensors; the torch path is differentiable
        in `controls`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement post_decision_features.")

    def quadrature_shocks(self) -> tuple[np.ndarray | None, np.ndarray]:
        """Quadrature nodes `(shocks, weights)` of the RL transition.

        `shocks` has shape (K, ...) with the same last-axis layout that
        `sample_shocks()` produces, so `transition()` can be reused verbatim;
        None for deterministic models. `weights` has shape (K,) and sums to one.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement quadrature_shocks.")

    def quadrature_next_states(self, state, controls: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Next states at every quadrature node via `transition()`. numpy only.

        state: (B, state_dim); controls: values with leading dimension B.
        Returns next states of shape (B, K, state_dim) and weights of shape (K,).
        """
        shocks, weights = self.quadrature_shocks()
        weights = np.asarray(weights, dtype=np.float64)

        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 2:
            raise ValueError(f"state must have shape (B, state_dim), got {state.shape}.")
        B, D = state.shape
        K = int(weights.shape[0])

        controls_b = {
            k: np.asarray(v, dtype=np.float64)[:, None, ...] for k, v in controls.items()
        }
        shocks_b = None if shocks is None else np.asarray(shocks, dtype=np.float64)[None, ...]

        next_state = np.asarray(
            self.transition(state[:, None, :], controls_b, shocks_b), dtype=np.float64
        )
        return np.broadcast_to(next_state, (B, K, D)).copy(), weights

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
