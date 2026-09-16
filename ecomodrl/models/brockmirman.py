from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ecomodrl.configs.parameters import BrockMirmanParams
from ecomodrl.models.casskoopmans import CassKoopmans
from ecomodrl.models.base_model import EconomicModel, gauss_hermite_nodes
from ecomodrl.models.backend import ops as _ops

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

    # ------------------------------------------------------------------
    # Post-decision hooks
    # ------------------------------------------------------------------

    @property
    def post_decision_dim(self) -> int:
        return 2

    def post_decision_features(self, state, controls):
        """W takes (log a, log z) with the post-decision state a = m - c."""
        xp = _ops(state)
        state = xp.asarray(state, dtype=np.float64)
        c = xp.asarray(controls["consumption"], dtype=np.float64)
        log_a = xp.log(xp.maximum(state[..., 0] - c, 1e-12))
        return xp.concatenate([log_a[..., None], xp.log(state[..., 1:])], axis=-1)

    def quadrature_shocks(self):
        """Gauss-Hermite nodes of the technology innovation, same layout as `sample_shocks`."""
        return self.shock_nodes.copy(), self.weights.copy()