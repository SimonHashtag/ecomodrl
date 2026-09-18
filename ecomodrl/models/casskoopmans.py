from __future__ import annotations

import numpy as np

from ecomodrl.configs.parameters import (
    CassKoopmansParams,
)
from ecomodrl.models.base_model import EconomicModel
from ecomodrl.models.array_ops import ops as _ops

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

    def model_spec_kwargs(self):
        return {"grid_size": self.grid_size}

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

    # ------------------------------------------------------------------
    # Post-decision hooks
    # ------------------------------------------------------------------

    @property
    def post_decision_dim(self) -> int:
        return 1

    def post_decision_features(self, state, controls):
        """W takes log a with the post-decision state a = m - c."""
        xp = _ops(state)
        m = xp.asarray(state, dtype=np.float64)[..., 0]
        c = xp.asarray(controls["consumption"], dtype=np.float64)
        return xp.log(xp.maximum(m - c, 1e-12))[..., None]

    def quadrature_shocks(self):
        """Deterministic growth: one node (transition ignores shocks)."""
        return None, np.ones(1)
