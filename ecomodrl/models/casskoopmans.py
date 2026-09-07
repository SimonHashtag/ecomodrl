from __future__ import annotations

import numpy as np

from ecomodrl.configs.parameters import (
    CassKoopmansParams,
)
from ecomodrl.models.base_model import EconomicModel

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

    def post_decision_spec(self, reward_floor: float) -> dict:
        """Deterministic growth: one node, post-decision state a = m - c.

        m' = A a^alpha + (1 - delta) a. Brock-Mirman overrides the node set
        and the persistence `rho` (here z is fixed at 1, so rho and eps are 0).
        """
        p = self.params
        spec = self._post_decision_common(reward_floor, family="growth", pd_dim=1)
        spec.update({
            "A": float(p.A),
            "alpha": float(p.alpha),
            "delta": float(p.delta),
            "rho": 0.0,
            "eps_nodes": [0.0],
            "weights": [1.0],
        })
        return spec