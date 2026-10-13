from __future__ import annotations

import numpy as np
from numba import njit
from scipy.stats import norm

from ecomodrl.configs.parameters import DurableGoodsParams
from ecomodrl.models.base_model import EconomicModel
from ecomodrl.models.array_ops import ops as _ops


# =============================================================================
# Grids (as in Fella2014/VFI.py, with the parameters as arguments)
# =============================================================================

def tauchen(n, rho, sigma, width):
    """Tauchen (1986): n-point Markov chain approximating x' = rho x + sigma eps."""
    std_x = sigma / np.sqrt(1 - rho**2)
    x = np.linspace(-width * std_x, width * std_x, n)
    half_step = (x[1] - x[0]) / 2
    # P[i, j] = Prob(x' in [x_j - half_step, x_j + half_step] | x = x_i)
    upper = norm.cdf((x[None, :] + half_step - rho * x[:, None]) / sigma)
    lower = norm.cdf((x[None, :] - half_step - rho * x[:, None]) / sigma)
    upper[:, -1] = 1.0  # last interval extends to +infinity
    lower[:, 0] = 0.0   # first interval extends to -infinity
    return x, upper - lower


def asset_grid(n_a, a_max):
    """Double-exponential grid on [0, a_max]: dense near the borrowing limit a = 0."""
    x = np.linspace(0.0, np.log(np.log(a_max + 1) + 1), n_a)
    return np.exp(np.exp(x) - 1) - 1


def stationary_distribution(P):
    """Stationary distribution of the Markov matrix P (left unit eigenvector)."""
    vals, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return pi / pi.sum()


# =============================================================================
# Benchmark policy off the grid (Fella2014/VFI.py `policy`)
# =============================================================================

@njit(cache=True)
def vfi_interp(x, xp, fp):
    """Linear interpolation of fp(xp) at x, extrapolating linearly at both ends."""
    i = min(max(np.searchsorted(xp, x) - 1, 0), len(xp) - 2)
    weight = (x - xp[i]) / (xp[i + 1] - xp[i])
    return fp[i] + weight * (fp[i + 1] - fp[i])


@njit(cache=True)
def vfi_policy_batch(a, i_d, i_y, affordable, a_grid, V_cond, a_cond):
    """Durable index d' and savings a' at assets a: interpolate V^{d'} in a, pick the
    best affordable d' (Eq. 14), then interpolate its savings policy a'^{d'}."""
    n = a.shape[0]
    n_d = V_cond.shape[3]
    k_out = np.empty(n, dtype=np.int64)
    a_out = np.empty(n)
    for s in range(n):
        i_best, v_best = i_d[s], -np.inf  # keeping d is always affordable
        for k in range(n_d):
            if not affordable[s, k]:
                continue
            v = vfi_interp(a[s], a_grid, V_cond[:, i_d[s], i_y[s], k])
            if v > v_best:
                i_best, v_best = k, v
        k_out[s] = i_best
        a_out[s] = vfi_interp(a[s], a_grid, a_cond[:, i_d[s], i_y[s], i_best])
    return k_out, a_out


# =============================================================================
# Durable-goods model (Fella 2014)
# =============================================================================

class DurableGoods(EconomicModel):
    """Durable and non-durable consumption with a fixed adjustment cost (Fella, 2014).

    State:
        state = (a, d, p, e): assets incl. borrowing capacity a = w + xi d, the
        durable stock d (one of n_d levels) and the two income components on
        their Tauchen nodes, log y = p + e.

    Action (Box(-1, 1, (2,))):
        action[0]: durable choice, n_d equal-width ordinal bins (bin k -> level k);
                   an unaffordable level is replaced by the nearest affordable one.
        action[1]: consumption share of cash-on-hand z(d'), c = share * z.

    Benchmark:
        Fella's VFI1 (`DurableGoodsVFI` in solvers.py) attaches the d'-contingent
        solution V_cond, a_cond on its own asset grid (vfi_grid_size points);
        m_grid (m_grid_size points) is the evaluation grid of the analysis.
    """

    PARAMS_CLS = DurableGoodsParams

    max_action_share = 1.0  # c = z, i.e. a' = 0, is the borrowing limit
    z_floor = 1e-8          # a durable level is affordable iff z(d') > z_floor
    init_a_max = 10.0       # initial / evaluation assets cover [0, init_a_max]

    def __init__(
        self,
        params=None,
        m_grid_size: int = 200,
        vfi_grid_size: int = 1000,
        a_max: float = 25.0,
    ):
        super().__init__(params)
        p = self.params

        self.m_grid_size = int(m_grid_size)
        self.vfi_grid_size = int(vfi_grid_size)
        self.a_max = float(a_max)

        self.m_grid = asset_grid(self.m_grid_size, self.a_max)
        self.a_grid = self.m_grid

        self.d_grid = np.linspace(p.d_min, p.d_max, p.n_d)
        self.p_grid, self.P_p = tauchen(p.n_p, p.rho, p.sigma_eta, p.tauchen_width)
        self.e_grid, self.P_e = tauchen(p.n_e, 0.0, p.sigma_eps, p.tauchen_width)
        self.pi_e = self.P_e[0].copy()  # e is iid: every row of P_e is the same
        self.pi_p = stationary_distribution(self.P_p)

        # Joint income chain, state index i_y = n_e * i_p + i_e (as in VFI.py)
        self.y_grid = np.exp(self.p_grid[:, None] + self.e_grid[None, :]).ravel()
        self.P = np.kron(self.P_p, self.P_e)
        self._P_cum = np.cumsum(self.P, axis=1)
        self._p_nodes = np.repeat(self.p_grid, p.n_e)
        self._e_nodes = np.tile(self.e_grid, p.n_p)

        self.sd_p = p.sigma_eta / np.sqrt(1.0 - p.rho**2)
        self.sd_e = p.sigma_eps
        self._d_mid = 0.5 * (p.d_min + p.d_max)
        self._d_half = 0.5 * (p.d_max - p.d_min)

        self.discrete_action_bins = {0: int(p.n_d)}

        # Euler errors only "off corners": a' at or above the first positive node
        # of the benchmark grid (Fella's criterion).
        self.euler_a_min = float(asset_grid(self.vfi_grid_size, self.a_max)[1])

        # Benchmark solution, attached by DurableGoodsVFI.solve()
        self.vfi_a_grid: np.ndarray | None = None
        self.V_cond: np.ndarray | None = None
        self.a_cond: np.ndarray | None = None

    def model_spec_kwargs(self):
        return {
            "m_grid_size": self.m_grid_size,
            "vfi_grid_size": self.vfi_grid_size,
            "a_max": self.a_max,
        }

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    def exogenous_grids(self):
        return (self.d_grid, self.p_grid, self.e_grid)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def durable_levels(self) -> np.ndarray:
        return self.d_grid

    # ------------------------------------------------------------------
    # Grid indices (numpy)
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest(x, grid):
        """Index of the nearest grid node (robust to round-off in observations)."""
        return np.abs(np.asarray(x, dtype=np.float64)[..., None] - grid).argmin(axis=-1)

    def income_index(self, state):
        """Joint income index i_y = n_e * i_p + i_e of raw states."""
        s = np.asarray(state, dtype=np.float64)
        i_p = self._nearest(s[..., 2], self.p_grid)
        i_e = self._nearest(s[..., 3], self.e_grid)
        return self.params.n_e * i_p + i_e

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def cash_on_hand_levels(self, state):
        """z(d'; a, d, y) for every durable level d', shape (..., n_d), Eq. (6).

        Accepts NumPy arrays or Torch tensors. d, p and e are snapped to their
        grid nodes, and the adjustment cost is charged by comparing level
        indices, never floats.
        """
        p = self.params
        xp = _ops(state)
        s = xp.asarray(state, dtype=np.float64)
        a = s[..., 0]

        def snap(x, grid):
            nodes = xp.take(grid, xp.arange(0, len(grid), like=a))
            return xp.argmin(xp.abs(x[..., None] - nodes))

        levels = xp.arange(0, p.n_d, like=a)
        d_row = xp.take(self.d_grid, levels)
        i_d = snap(s[..., 1], self.d_grid)
        d_now = xp.take(self.d_grid, i_d)
        y = xp.exp(xp.take(self.p_grid, snap(s[..., 2], self.p_grid)) + xp.take(self.e_grid, snap(s[..., 3], self.e_grid)))

        adjust = levels != i_d[..., None]
        return (
            (y + (1 + p.r) * a)[..., None]
            - (1 - p.xi) * (d_row - d_now[..., None])
            - xp.where(adjust, p.phi * d_row, 0.0)
            - (p.r * p.xi * d_now)[..., None]
        )

    # ------------------------------------------------------------------
    # RL hooks
    # ------------------------------------------------------------------

    def controls_from_action(self, action, state=None):
        """Map the Gym action in [-1, 1]^2 to (d', c, a').

        Accepts NumPy arrays or Torch tensors. The torch path is differentiable
        in action[..., 1] (consumption share); the durable bin is a hard choice,
        so the gradient with respect to action[..., 0] is exactly zero.
        """
        n_d = self.params.n_d
        xp = _ops(action, state)
        act = xp.clip(xp.asarray(action, dtype=np.float64), -1.0, 1.0)

        code = xp.clip(xp.floor(0.5 * (act[..., 0] + 1.0) * n_d), 0, n_d - 1)
        share = self.min_action_share + 0.5 * (
            self.max_action_share - self.min_action_share
        ) * (act[..., 1] + 1.0)

        out = {"share": share, "durable_code": code}
        if state is None:
            return out

        # Nearest affordable level to the chosen bin (ties -> the lower level)
        z_all = self.cash_on_hand_levels(state)
        levels = xp.arange(0, n_d, like=z_all)
        dist = xp.where(z_all > self.z_floor, xp.abs(levels - code[..., None]), float(n_d))
        k = xp.argmin(dist)

        z = xp.take_along_axis(z_all, k[..., None])[..., 0]
        c = share * z

        out.update(
            consumption=c,
            savings=z - c,
            cash_on_hand=z,
            durable=xp.take(self.d_grid, k),
            durable_index=k,
        )
        return out

    def utility(self, c):
        raise NotImplementedError(
            "DurableGoods utility depends on the durable choice; use utility_from_controls."
        )

    def utility_from_controls(self, controls):
        """u(c, d') = theta log c + (1 - theta) log(kappa d'), Eq. (22)."""
        p = self.params
        xp = _ops(controls["consumption"], controls["durable"])
        c = xp.asarray(controls["consumption"], dtype=np.float64)
        d = xp.asarray(controls["durable"], dtype=np.float64)
        return p.theta * xp.log(c) + (1 - p.theta) * xp.log(p.kappa * d)

    def marginal_utility(self, c):
        return self.params.theta / np.asarray(c)

    def obs_from_state(self, state):
        """(log(1 + a), d scaled to [-1, 1], p / sd_p, e / sd_e)."""
        xp = _ops(state)
        s = xp.asarray(state, dtype=np.float64)
        return xp.concatenate(
            [
                xp.log1p(xp.maximum(s[..., 0], 0.0))[..., None],
                ((s[..., 1] - self._d_mid) / self._d_half)[..., None],
                (s[..., 2] / self.sd_p)[..., None],
                (s[..., 3] / self.sd_e)[..., None],
            ],
            axis=-1,
        )

    def state_from_obs(self, obs):
        """Inverse of obs_from_state for NumPy arrays or Torch tensors."""
        xp = _ops(obs)
        o = xp.asarray(obs, dtype=np.float64)
        return xp.concatenate(
            [
                xp.expm1(o[..., 0])[..., None],
                (o[..., 1] * self._d_half + self._d_mid)[..., None],
                (o[..., 2] * self.sd_p)[..., None],
                (o[..., 3] * self.sd_e)[..., None],
            ],
            axis=-1,
        )

    def sample_shocks(self, rng, shape):
        """One uniform per period; transition() inverts the CDF of the income row."""
        return rng.random(tuple(shape) + (1,))

    def transition(self, state, controls, shocks):
        """(a', d', p', e') with y' drawn from row i_y of P by inverting its CDF."""
        n_y = self.P.shape[0]
        i_y = self.income_index(state)
        u = np.asarray(shocks, dtype=np.float64)[..., 0]

        # j = searchsorted(P_cum[i_y], u), as in VFI.py's simulate
        j = np.minimum((u[..., None] > self._P_cum[i_y]).sum(axis=-1), n_y - 1)

        a_next, d_next, p_next, e_next = np.broadcast_arrays(
            np.asarray(controls["savings"], dtype=np.float64),
            np.asarray(controls["durable"], dtype=np.float64),
            self._p_nodes[j],
            self._e_nodes[j],
        )
        return np.stack([a_next, d_next, p_next, e_next], axis=-1)

    def sample_initial_state(self, rng):
        p = self.params
        return np.array(
            [
                rng.uniform(0.0, self.init_a_max),
                self.d_grid[rng.integers(p.n_d)],
                self.p_grid[rng.choice(p.n_p, p=self.pi_p)],
                self.e_grid[rng.choice(p.n_e, p=self.pi_e)],
            ],
            dtype=np.float64,
        )

    def eval_start_states(self, M):
        """Deterministic grid: assets x durable levels x p nodes, e at its median node."""
        p = self.params
        n_a = max(1, int(M) // (p.n_d * p.n_p))
        a = self.log_grid_from_zero(self.init_a_max, n_a)
        A, D, P = np.meshgrid(a, self.d_grid, self.p_grid, indexing="ij")
        E = np.full_like(A, self.e_grid[p.n_e // 2])
        return np.stack([A.ravel(), D.ravel(), P.ravel(), E.ravel()], axis=-1)

    # ------------------------------------------------------------------
    # Post-decision hooks
    # ------------------------------------------------------------------

    @property
    def post_decision_dim(self) -> int:
        return 3

    def post_decision_features(self, state, controls):
        """W takes (log(1 + a'), d' scaled, p / sd_p): e is iid, so the
        continuation value depends on income only through p."""
        xp = _ops(state, controls["savings"])
        s = xp.asarray(state, dtype=np.float64)
        a_next = xp.asarray(controls["savings"], dtype=np.float64)
        d_next = xp.asarray(controls["durable"], dtype=np.float64)
        return xp.concatenate(
            [
                xp.log1p(xp.maximum(a_next, 0.0))[..., None],
                ((d_next - self._d_mid) / self._d_half)[..., None],
                (s[..., 2] / self.sd_p)[..., None],
            ],
            axis=-1,
        )

    def quadrature_shocks(self):
        raise NotImplementedError(
            "DurableGoods has a Markov income state; quadrature_next_states returns "
            "the exact transition rows as (B, K) weights."
        )

    @property
    def n_quadrature_nodes(self) -> int:
        return self.P.shape[0]

    def quadrature_next_states(self, state, controls):
        """Next states at all n_p * n_e income nodes, shape (B, K, 4), and the
        state-dependent transition probabilities P[i_y], shape (B, K)."""
        s = np.asarray(state, dtype=np.float64)
        if s.ndim != 2:
            raise ValueError(f"state must have shape (B, state_dim), got {s.shape}.")
        B, K = s.shape[0], self.P.shape[0]

        next_state = np.empty((B, K, 4), dtype=np.float64)
        next_state[..., 0] = np.asarray(controls["savings"], dtype=np.float64).reshape(B, 1)
        next_state[..., 1] = np.asarray(controls["durable"], dtype=np.float64).reshape(B, 1)
        next_state[..., 2] = self._p_nodes[None, :]
        next_state[..., 3] = self._e_nodes[None, :]

        return next_state, self.P[self.income_index(s)]

    # ------------------------------------------------------------------
    # Benchmark policy (VFI solution)
    # ------------------------------------------------------------------

    def benchmark_controls(self, state):
        """Controls of the attached VFI solution at raw states, same keys as the RL controls."""
        if self.V_cond is None:
            raise RuntimeError("No VFI solution attached. Run DurableGoodsVFI(model).solve() first.")

        s = np.asarray(state, dtype=np.float64)
        lead = s.shape[:-1]
        flat = s.reshape(-1, 4)
        n = flat.shape[0]

        z_all = self.cash_on_hand_levels(flat)
        k, a_next = vfi_policy_batch(
            np.ascontiguousarray(flat[:, 0]),
            self._nearest(flat[:, 1], self.d_grid),
            self.income_index(flat),
            z_all > self.z_floor,
            self.vfi_a_grid,
            self.V_cond,
            self.a_cond,
        )
        z = z_all[np.arange(n), k]
        c = z - a_next

        out = {
            "consumption": c,
            "share": c / z,
            "savings": a_next,
            "cash_on_hand": z,
            "durable": self.d_grid[k],
            "durable_index": k,
        }
        return {key: value.reshape(lead) for key, value in out.items()}

    def consumption(self, state):
        return self.benchmark_controls(state)["consumption"]

    def share(self, state):
        return self.benchmark_controls(state)["share"]

    # ------------------------------------------------------------------
    # Analysis hooks (used by analysis/accuracy_metrics.py)
    # ------------------------------------------------------------------

    def euler_errors_on_states(self, states, controls_fn, share_max: float = 0.999):
        """log10 |1 - c*/c| with theta / c* = beta (1 + r) E[theta / c' | y], Eqs. (24)-(25).

        c' is the same policy's consumption at the next state (including its next
        durable choice). Constrained states are dropped: a' below the first
        positive node of the benchmark grid, or a consumption share above share_max.
        """
        p = self.params
        s = np.asarray(states, dtype=np.float64).reshape(-1, 4)
        ctl = controls_fn(s)
        c = np.asarray(ctl["consumption"], dtype=np.float64).reshape(-1)
        z = np.asarray(ctl["cash_on_hand"], dtype=np.float64).reshape(-1)
        a_next = np.asarray(ctl["savings"], dtype=np.float64).reshape(-1)
        d_next = np.asarray(ctl["durable"], dtype=np.float64).reshape(-1)

        keep = (a_next >= self.euler_a_min) & (c > 0.0) & (c <= share_max * z)
        if not np.any(keep):
            return np.array([], dtype=np.float64)

        next_states, weights = self.quadrature_next_states(
            s[keep], {"savings": a_next[keep], "durable": d_next[keep]}
        )
        c_next = np.asarray(
            controls_fn(next_states.reshape(-1, 4))["consumption"], dtype=np.float64
        ).reshape(weights.shape)
        c_next = np.maximum(c_next, 1e-12)

        expected_mu = (weights * (p.theta / c_next)).sum(axis=1)
        c_star = p.theta / (p.beta * (1 + p.r) * expected_mu)

        err = np.abs(1.0 - c_star / c[keep])
        err = err[np.isfinite(err) & (err > 0.0)]
        return np.log10(err) if err.size else np.array([], dtype=np.float64)

    def evaluate_policy_value(self, controls_fn, tol=1e-8, max_iter=10_000):
        """Value of a fixed policy on the grid (m_grid, d, p, e).

        The policy is evaluated once; the iteration
            V = u + beta * sum_j P_p[i_p, j] sum_l pi_e[l] V(a', d', j, l)
        interpolates V linearly in a' only (d' and the income nodes are exact).
        """
        p = self.params
        axes = (self.m_grid,) + self.exogenous_grids
        shape = tuple(len(g) for g in axes)
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

        ctl = controls_fn(mesh.reshape(-1, 4))
        c = np.maximum(np.asarray(ctl["consumption"], dtype=np.float64), 1e-12)
        u = np.asarray(
            self.utility_from_controls({"consumption": c, "durable": ctl["durable"]}),
            dtype=np.float64,
        ).reshape(shape)

        grid = self.m_grid
        a_next = np.clip(np.asarray(ctl["savings"], dtype=np.float64).reshape(shape), grid[0], grid[-1])
        i_lo = np.clip(np.searchsorted(grid, a_next, side="right") - 1, 0, len(grid) - 2)
        w_hi = (a_next - grid[i_lo]) / (grid[i_lo + 1] - grid[i_lo])
        k = np.asarray(ctl["durable_index"]).reshape(shape).astype(np.int64)
        i_p = np.broadcast_to(np.arange(shape[2])[None, None, :, None], shape)

        V = np.zeros(shape, dtype=np.float64)
        for _ in range(int(max_iter)):
            EV = (V @ self.pi_e) @ self.P_p.T  # EV[a', d', i_p]
            V_new = u + p.beta * ((1.0 - w_hi) * EV[i_lo, k, i_p] + w_hi * EV[i_lo + 1, k, i_p])
            error = np.max(np.abs(V_new - V))
            V = V_new
            if error < tol:
                break

        return axes, V

    # ------------------------------------------------------------------
    # EGM hooks (not used: the benchmark is VFI)
    # ------------------------------------------------------------------

    def euler_rhs(self, c_next, **transition_kwargs):
        raise NotImplementedError("DurableGoods uses the VFI benchmark (DurableGoodsVFI).")

    def transition_grid_kwargs(self):
        raise NotImplementedError("DurableGoods uses the VFI benchmark (DurableGoodsVFI).")

    def next_state_distribution(self, state, a):
        raise NotImplementedError("DurableGoods uses euler_errors_on_states / evaluate_policy_value.")
