from __future__ import annotations

import numpy as np

from ecomodrl.configs.parameters import (
    BufferStockParams,
)
from ecomodrl.models.casskoopmans import EconomicModel
from ecomodrl.models.base_model import gauss_hermite_nodes
from ecomodrl.models.backend import ops as _ops


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

    M_MAX = 10.0
    grid_upper_factor = 2.5
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
            m_max = self.M_MAX

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

    # ------------------------------------------------------------------
    # Post-decision hooks
    # ------------------------------------------------------------------

    @property
    def post_decision_dim(self) -> int:
        return 1

    def post_decision_features(self, state, controls):
        """W takes log(1 + a) with the post-decision state a = m - c >= 0."""
        xp = _ops(state)
        m = xp.asarray(state, dtype=np.float64)[..., 0]
        c = xp.asarray(controls["consumption"], dtype=np.float64)
        return xp.log1p(xp.maximum(m - c, 0.0))[..., None]

    def quadrature_shocks(self):
        """The model's 56 quadrature nodes in the raw-shock layout of `sample_shocks`:
        (nu, zeta, unemployment indicator). Node order mirrors `_build_shocks`
        (unemployed nodes first, then the employed tensor product with the
        permanent shock varying slowest), so the weights are `self.weights`.
        `transition()` applies the tilted permanent-shock mean itself.
        """
        x, _ = gauss_hermite_nodes(self.n_gh_nodes)

        unemp = np.stack([x, np.zeros_like(x), np.ones_like(x)], axis=-1)

        nu, zeta = np.meshgrid(x, x, indexing="ij")
        emp = np.stack([nu.ravel(), zeta.ravel(), np.zeros(nu.size)], axis=-1)

        return np.concatenate([unemp, emp], axis=0), self.weights.copy()
