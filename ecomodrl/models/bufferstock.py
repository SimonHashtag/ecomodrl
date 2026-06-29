from __future__ import annotations

import numpy as np

from ecomodrl.configs.parameters import (
    BufferStockParams,
)
from ecomodrl.models.casskoopmans import EconomicModel
from ecomodrl.models.base_model import gauss_hermite_nodes


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
