from __future__ import annotations

import time
import itertools

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from numba import njit, prange
from numba.typed import List
from scipy.optimize import root

from ecomodrl.models.base_model import EconomicModel
from ecomodrl.models.portfoliochoice import PortfolioChoice

# =============================================================================
# Generic EGM solver for one continuous control (consumption)
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
# PortfolioChoice solver (Mixed EGM/root) - Numba optimized
# =============================================================================

# 1) Multilinear, barycentric interpolation utilities

@njit(cache=True, fastmath=True)
def _binsearch1d(grid, x, inv_deltas):
    """
    Locate x on a 1D grid and return the left index and interpolation weight.
    Clamps x to the grid (extrapolation by setting to the nearest grid point).

    Parameters
    ----------
    grid : 1D ndarray (monotone increasing)
        Grid points.
    x : float
        Query value.
    inv_deltas : 1D ndarray
        Precomputed 1.0 / (grid[1:] - grid[:-1]).

    Returns
    -------
    i : int
        Left bracket index such that grid[i] <= x <= grid[i+1],
        clamped to the interval [0, n-2].
    t : float
        Linear weight in [0, 1]; the interpolated value is
        (1 - t) * f[i] + t * f[i+1].
    """
    n = grid.shape[0]

    # Find insertion point of x in grid (right side of the left neighbor)
    j = np.searchsorted(grid, x, side="right") - 1

    if j < 0:
        return 0, 0.0
    if j >= n - 1:
        return n - 2, 1.0

    t = (x - grid[j]) * inv_deltas[j]
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    return j, t


@njit(cache=True, fastmath=True)
def _c_strides(shape):
    """
    Compute C-order (row-major) logical strides for an N-D tensor shape.

    Used for indexing flattened N-D tensors.

    Parameters
    ----------
    shape : 1D ndarray of int
        Tensor shape, e.g. (Nm, Nz1, ..., NzK).

    Returns
    -------
    strides : 1D ndarray of int
        C-order strides so that flat_index = Σ idx[d] * strides[d].
    """
    D = len(shape)
    s = np.empty(D, np.int64)
    s[D - 1] = 1
    for d in range(D - 2, -1, -1):
        s[d] = s[d + 1] * shape[d + 1]
    return s


@njit(cache=True, fastmath=True)
def _z_corner_offsets(strides, K):
    """
    Precompute flat-index offsets for all 2^K z-corners in multilinear
    interpolation.

    Parameters
    ----------
    strides : 1D ndarray of int
        C-order strides for the full tensor (m, z1, ..., zK).
    K : int
        Number of z dimensions.

    Returns
    -------
    offsets : 1D ndarray of int, shape (2^K,)
        For each binary corner mask, the flat offset relative to the
        z base index.

    Notes
    -----
    Corner mask bit k = 1 selects the +1 corner in z_k; 0 selects the
    lower (0) corner.
    """
    C = 1 << K
    offs = np.zeros(C, np.int64)
    for mask in range(C):
        acc = 0
        for k in range(K):
            if (mask >> k) & 1:
                # z-dimensions start at axis=1 (axis 0 is m)
                acc += strides[1 + k]
        offs[mask] = acc
    return offs


@njit(cache=True, fastmath=True, parallel=True)
def _precompute_z_interp(zs_next, z_grids_list, z_inv_deltas_list, strides):
    """
    For each shock draw i, cache the z interpolation structure.

    Parameters
    ----------
    zs_next : 2D ndarray, shape (M, K)
        Next-period predictors for each risky asset and Sobol draw.
    z_grids_list : numba.typed.List of 1D ndarrays (length K)
        Grids for each predictor dimension.
    z_inv_deltas_list : numba.typed.List of 1D ndarrays (length K)
        For each dimension k: 1.0 / (z_grid_k[1:] - z_grid_k[:-1]).
    strides : 1D ndarray of int
        C-order strides for the target tensor (m, z1, ..., zK).

    Returns
    -------
    base_z_flat : 1D ndarray of int, shape (M,)
        Flat index for the z cell's lower-left corner (m-axis excluded).
    z_w : 2D ndarray of float64, shape (M, 2^K)
        Corner weights for multilinear interpolation over z at each draw.
    """
    M, K = zs_next.shape
    C = 1 << K      # number of corners
    base_z_flat = np.empty(M, np.int64)
    z_w = np.empty((M, C), np.float64)

    for i in prange(M):
        # 1) locate z-cell and 1D interpolation weights per dimension
        idxs = np.empty(K, np.int64)
        taus = np.empty(K, np.float64)

        for k in range(K):
            grid_k = z_grids_list[k]
            inv_deltas_k = z_inv_deltas_list[k]
            lo, t = _binsearch1d(grid_k, zs_next[i, k], inv_deltas_k)
            idxs[k] = lo
            taus[k] = t

        # 2) Convert K-D index vector (z1_lo, ..., zK_lo) to base flat index
        #    for m-row 0.
        acc = 0
        for k in range(K):
            acc += idxs[k] * strides[1 + k]
        base_z_flat[i] = acc

        # 3) Full multilinear weights over all 2^K z-corners
        for mask in range(C):
            w = 1.0
            for k in range(K):
                if (mask >> k) & 1:
                    w *= taus[k]
                else:
                    w *= 1.0 - taus[k]
            z_w[i, mask] = w

    return base_z_flat, z_w


@njit(cache=True, fastmath=True)
def _linear_interp_1d(a0, a1, t):
    """Simple linear interpolation between a0 and a1 with weight t in [0, 1]."""
    return (1.0 - t) * a0 + t * a1

# 2) Numba version of sdf_moments function

@njit(cache=True, fastmath=True, parallel=True)
def _sdf_moments_fast(
    rs,
    Rs_next,
    w,
    theta_vals,
    Gamma_next,
    SDF_factor,
    theta_low,
    unemp_prob,
    gamma,
    min_share,
    # policy grid info:
    a_values_flat,
    shape,
    strides,
    m_grid,
    inv_m_deltas,
    base_z_flat,
    z_corner_offs,
    z_w,
):
    """
    Compute SDF moments g ∈ R^{n_assets} at (rs, z) using cached z-structure.

    Economic object
    ---------------
    g_j = E[ u'(c_{t+1}) / Γ_{t+1}^γ * R_{j,t+1} ],  j = 0..n_assets-1

    Parameters
    ----------
    rs : float
        Resources saved today (m - c), scalar.
    Rs_next : 2D ndarray, shape (M, n_assets)
        Asset gross returns for each MC draw (col 0 = safe).
    w : 1D ndarray, shape (n_assets,)
        Portfolio weights on assets today (sum to 1).
    theta_vals : 1D ndarray, shape (M,)
        Transitory income draws Θ' for each MC draw.
    Gamma_next : 1D ndarray, shape (M,)
        Permanent growth factors Γ' for each MC draw.
    SDF_factor : 1D ndarray, shape (M,)
        Equals (Γ'^γ)^{-1} reused across calls.
    theta_low : float
        Unemployment income level.
    unemp_prob : float
        Probability of unemployment next period.
    gamma : float
        Risk aversion parameter.
    min_share : float
        Floor for consumption share a'(m', z') to stabilize interpolation.
    a_values_flat : 1D ndarray
        Flattened policy array a(m, z) in C-order.
    shape : 1D ndarray of int
        Shape of the a-policy tensor (Nm, Nz1, ..., NzK).
    strides : 1D ndarray of int
        C-order strides of the same tensor.
    m_grid : 1D ndarray
        Grid for m.
    inv_m_deltas : 1D ndarray
        Precomputed 1.0 / (m_grid[1:] - m_grid[:-1]).
    base_z_flat : 1D ndarray of int, shape (M,)
        Base flat index (z part) for each draw.
    z_corner_offs : 1D ndarray of int, shape (2^K,)
        Offsets of each z-corner relative to base_z_flat.
    z_w : 2D ndarray, shape (M, 2^K)
        Weights of each z-corner for each draw.

    Returns
    -------
    g : 1D ndarray, shape (n_assets,)
        SDF-weighted expected returns.
    """
    M, n_assets = Rs_next.shape
    K = len(shape) - 1
    C = 1 << K
    stride0 = strides[0]

    # Portfolio return for each draw
    Rp = np.empty(M)
    for i in prange(M):
        acc = 0.0
        for j in range(n_assets):
            acc += Rs_next[i, j] * w[j]
        Rp[i] = acc

    # Mixture of u'(c') * (Γ')^{-γ} across employment states
    upd_mix = np.empty(M)
    pE = 1.0 - unemp_prob

    for i in prange(M):
        base = (Rp[i] / Gamma_next[i]) * rs
        m_e = base + theta_vals[i]
        m_u = base + theta_low

        # --- employed branch ---
        im, t = _binsearch1d(m_grid, m_e, inv_m_deltas)
        i0 = im * stride0
        i1 = (im + 1) * stride0
        base_flat = base_z_flat[i]

        a_e = 0.0
        for c in range(C):
            off = z_corner_offs[c]
            wzc = z_w[i, c]
            a0 = a_values_flat[i0 + base_flat + off]
            a1 = a_values_flat[i1 + base_flat + off]
            a_e += wzc * _linear_interp_1d(a0, a1, t)

        if a_e < min_share:
            a_e = min_share

        c_e = a_e * m_e
        upd_e = (c_e ** (-gamma)) * SDF_factor[i]

        # --- unemployed branch ---
        im, t = _binsearch1d(m_grid, m_u, inv_m_deltas)
        i0 = im * stride0
        i1 = (im + 1) * stride0

        a_u = 0.0
        for c in range(C):
            off = z_corner_offs[c]
            wzc = z_w[i, c]
            a0 = a_values_flat[i0 + base_flat + off]
            a1 = a_values_flat[i1 + base_flat + off]
            a_u += wzc * _linear_interp_1d(a0, a1, t)

        if a_u < min_share:
            a_u = min_share

        c_u = a_u * m_u
        upd_u = (c_u ** (-gamma)) * SDF_factor[i]

        upd_mix[i] = pE * upd_e + (1.0 - pE) * upd_u

    # Average across shocks and multiply by returns
    g = np.empty(n_assets)
    for j in prange(n_assets):
        acc = 0.0
        for i in range(M):
            acc += upd_mix[i] * Rs_next[i, j]
        g[j] = acc / M

    return g


class PortfolioChoiceSolver:
    """Nested EGM/root solver for PortfolioChoice.

    Consumption is updated by EGM.
    Portfolio weights are updated by local root solving of SDF/KKT conditions.

    After solve:
        model.share_policy_grid
        model.portfolio_policy_grid
        model.policy

    are all populated.
    """

    def __init__(
        self,
        econ_model: PortfolioChoice,
        max_iter: int = 100,
        tol: float = 1e-5,
        conv: float = 1e-5,
        min_share: float = 1e-6,
        root_interval: int = 15,
        mu_fb: float = 1e-4,
    ):
        self.model = econ_model
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.conv = float(conv)
        self.min_share = float(min_share)
        self.root_interval = int(root_interval)
        self.mu_fb = float(mu_fb)

        self.share_policy: np.ndarray | None = None
        self.portfolio_policy: np.ndarray | None = None
        self.solve_time: float | None = None

    # ------------------------------------------------------------------
    # Simplex parameterization used internally by the root solver
    # ------------------------------------------------------------------

    @staticmethod
    def logits_to_weights(logits: np.ndarray) -> np.ndarray:
        """ALR inverse with last asset as pivot.

        Internal root-solver parameterization only. Returned weights are in
        model asset order [safe, risky_1, ..., risky_N].
        """
        x = np.asarray(logits, dtype=np.float64)
        e = np.exp(np.clip(x, -50.0, 50.0))
        denom = max(1.0 + float(np.sum(e)), 1e-300)
        return np.concatenate([e / denom, np.array([1.0 / denom])])

    @staticmethod
    def weights_to_logits(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)
        w = np.clip(w, 1e-15, None)
        w = w / w.sum()
        return np.log(w[:-1]) - np.log(w[-1])

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, verbose: bool = True):
        import time

        start = time.time()

        m = self.model
        m_grid = m.m_grid
        rs_grid = m.rs_grid
        z_grids = list(m.exogenous_grids)

        share = np.full(m.policy_shape, 0.2, dtype=np.float64)

        w_rs = np.full(
            (len(rs_grid),) + m.exogenous_shape + (m.n_assets,),
            1.0 / m.n_assets,
            dtype=np.float64,
        )

        z_indices = [range(len(g)) for g in z_grids]

        # Numba interpolation precomputations.
        shape = np.array(share.shape, dtype=np.int64)
        strides = _c_strides(shape)
        z_corner_offs = _z_corner_offsets(strides, len(z_grids))

        m_grid_c = np.ascontiguousarray(m_grid, dtype=np.float64)
        inv_m_deltas = 1.0 / (m_grid_c[1:] - m_grid_c[:-1])

        z_typed = List()
        z_inv_typed = List()

        for zg in z_grids:
            zg_c = np.ascontiguousarray(zg, dtype=np.float64)
            z_typed.append(zg_c)
            z_inv_typed.append(1.0 / (zg_c[1:] - zg_c[:-1]))

        c_buf = np.empty(len(rs_grid), dtype=np.float64)
        endogenous_m_buf = np.empty(len(rs_grid), dtype=np.float64)

        for iteration in range(self.max_iter):
            share_old = share.copy()
            w_old = w_rs.copy()

            share_c = np.ascontiguousarray(share, dtype=np.float64)
            share_flat = share_c.ravel()

            mu_now = max(1e-8, self.mu_fb * (0.5 ** (iteration // 2)))

            for z_idx in itertools.product(*z_indices):
                z_vec = np.array(
                    [z_grids[j][idx] for j, idx in enumerate(z_idx)],
                    dtype=np.float64,
                )

                z_next, R_next = m.zs_R_next(z_vec)

                z_next_c = np.ascontiguousarray(z_next, dtype=np.float64)
                R_next_c = np.ascontiguousarray(R_next, dtype=np.float64)

                base_z_flat, z_w = _precompute_z_interp(
                    z_next_c,
                    z_typed,
                    z_inv_typed,
                    strides,
                )

                for i_rs, rs in enumerate(rs_grid):
                    if rs <= self.min_share:
                        w_star = np.full(m.n_assets, 1.0 / m.n_assets)
                    else:
                        w0 = w_rs[(i_rs, *z_idx, slice(None))]

                        if iteration % self.root_interval == 0:
                            w_star = self._solve_portfolio_weights(
                                rs=float(rs),
                                w0=w0,
                                R_next=R_next_c,
                                share_flat=share_flat,
                                shape=shape,
                                strides=strides,
                                m_grid=m_grid_c,
                                inv_m_deltas=inv_m_deltas,
                                base_z_flat=base_z_flat,
                                z_corner_offs=z_corner_offs,
                                z_w=z_w,
                                mu_now=mu_now,
                            )
                        else:
                            w_star = w0

                    g = _sdf_moments_fast(
                        float(rs),
                        R_next_c,
                        w_star,
                        m._theta_vals,
                        m.Gamma_next,
                        m.SDF_factor,
                        m.theta_low,
                        m.unemp_prob,
                        m.params.gamma,
                        self.min_share,
                        share_flat,
                        shape,
                        strides,
                        m_grid_c,
                        inv_m_deltas,
                        base_z_flat,
                        z_corner_offs,
                        z_w,
                    )

                    gbar = float(np.dot(w_star, g))

                    w_rs[(i_rs, *z_idx, slice(None))] = w_star

                    c = max(
                        m.inv_marginal_utility(m.params.beta * gbar),
                        self.min_share,
                    )

                    endogenous_m_buf[i_rs] = float(rs) + c
                    c_buf[i_rs] = c

                order = np.argsort(endogenous_m_buf)
                m_endo = endogenous_m_buf[order]
                c_endo = c_buf[order]

                share_endo = np.clip(c_endo / np.maximum(m_endo, 1e-14), self.min_share, 1.0)

                share[(slice(None), *z_idx)] = np.interp(
                    m_grid,
                    np.concatenate([[0.0], m_endo]),
                    np.concatenate([[1.0], share_endo]),
                )

            if iteration % self.root_interval == 0:
                da = float(np.max(np.abs(share - share_old)))
                dw = float(np.max(np.abs(w_rs - w_old)))
                converged = da <= self.conv and dw <= self.conv

                if verbose:
                    print(
                        f"[PortfolioChoice iter {iteration}] "
                        f"max|Δshare|={da:.3e}, max|Δw|={dw:.3e}, "
                        f"converged={converged}",
                        flush=True,
                    )

                if converged:
                    break

        w_mz = self._weights_rs_to_mz(
            rs_grid=rs_grid,
            m_grid=m_grid,
            z_grids=z_grids,
            share=share,
            w_rs=w_rs,
        )

        self.share_policy = share
        self.portfolio_policy = w_mz

        m.share_policy_grid = share
        m.portfolio_policy_grid = w_mz

        # Compatibility with EconomicModel.consumption convention.
        m_axis = m_grid.reshape((len(m_grid),) + (1,) * m.exogenous_dim)
        m.policy = share * m_axis

        self.policy = m.policy
        self.solve_time = time.time() - start

        if verbose:
            print(f"PortfolioChoice solver finished in {self.solve_time:.2f} seconds.")

        return share, w_mz

    def _solve_portfolio_weights(
        self,
        *,
        rs,
        w0,
        R_next,
        share_flat,
        shape,
        strides,
        m_grid,
        inv_m_deltas,
        base_z_flat,
        z_corner_offs,
        z_w,
        mu_now,
    ):
        m = self.model

        g0 = _sdf_moments_fast(
            float(rs),
            R_next,
            w0,
            m._theta_vals,
            m.Gamma_next,
            m.SDF_factor,
            m.theta_low,
            m.unemp_prob,
            m.params.gamma,
            self.min_share,
            share_flat,
            shape,
            strides,
            m_grid,
            inv_m_deltas,
            base_z_flat,
            z_corner_offs,
            z_w,
        )

        x0 = np.concatenate([self.weights_to_logits(w0), [float(np.max(g0))]])

        def residuals(x):
            logits = x[:-1]
            xi = x[-1]
            w = self.logits_to_weights(logits)

            g = _sdf_moments_fast(
                float(rs),
                R_next,
                w,
                m._theta_vals,
                m.Gamma_next,
                m.SDF_factor,
                m.theta_low,
                m.unemp_prob,
                m.params.gamma,
                self.min_share,
                share_flat,
                shape,
                strides,
                m_grid,
                inv_m_deltas,
                base_z_flat,
                z_corner_offs,
                z_w,
            )

            return self.smooth_fisher_burmeister(w, xi - g, mu=mu_now)

        sol = root(residuals, x0, method="hybr", tol=self.tol)

        if not sol.success:
            return np.asarray(w0, dtype=np.float64)

        w = self.logits_to_weights(sol.x[:-1])
        w = np.clip(w, 0.0, None)
        return w / np.maximum(w.sum(), 1e-14)

    def _weights_rs_to_mz(self, *, rs_grid, m_grid, z_grids, share, w_rs):
        """Map weights from (rs,z) to (m,z) using rs = m * (1-share(m,z))."""
        m = self.model

        w_interp = m.make_multilinear(
            w_rs,
            axes=(rs_grid, *z_grids),
        )

        grids = [m_grid, *z_grids]
        mesh = np.meshgrid(*grids, indexing="ij")
        pts_mz = np.stack([x.ravel() for x in mesh], axis=1)

        share_interp = m.make_multilinear(share)
        share_vals = np.asarray(share_interp(pts_mz), dtype=np.float64)

        pts_rsz = pts_mz.copy()
        pts_rsz[:, 0] = pts_mz[:, 0] * (1.0 - share_vals)

        w_flat = np.asarray(w_interp(pts_rsz), dtype=np.float64)
        w_flat = np.clip(w_flat, 0.0, None)
        w_flat /= np.maximum(w_flat.sum(axis=1, keepdims=True), 1e-14)

        return w_flat.reshape(tuple(len(g) for g in grids) + (m.n_assets,))

    @staticmethod
    def smooth_fisher_burmeister(a, b, mu: float = 1e-6):
        return np.sqrt(a**2 + b**2 + 2.0 * mu) - (a + b)
    

# =============================================================================
# Durable goods (Fella 2014): VFI1 = grid search + MONOTONE + REFINE
# =============================================================================
# Literal port of Fella2014/VFI.py with the parameters as arguments; the
# Bellman update runs the independent (y, d) blocks in parallel (prange), which
# leaves every block's arithmetic, and therefore the solution, unchanged.

_DG_PENALTY = -1e7      # value of an unaffordable durable choice (z <= 0)
_DG_REFINE_TOL = 1e-8   # stop refining once the bracket is this narrow


@njit(cache=True)
def _dg_utility(c, d_next, theta, kappa):
    """u(c, d') = theta log c + (1 - theta) log(kappa d'), Eq. (22)."""
    return theta * np.log(c) + (1 - theta) * np.log(kappa * d_next)


@njit(cache=True)
def _dg_cash_on_hand(a, d, y, d_next, r, xi, phi):
    """z(d'; a, d, y), Eq. (6): resources left for c + a' after choosing d'."""
    adjustment_cost = phi * d_next * (d_next != d)
    return y + (1 + r) * a - (1 - xi) * (d_next - d) - adjustment_cost - r * xi * d


@njit(cache=True)
def _dg_interp(x, xp, fp):
    """Linear interpolation of fp(xp) at x, extrapolating linearly at both ends."""
    i = min(max(np.searchsorted(xp, x) - 1, 0), len(xp) - 2)
    weight = (x - xp[i]) / (xp[i + 1] - xp[i])
    return fp[i] + weight * (fp[i + 1] - fp[i])


@njit(cache=True)
def _dg_refine(lo, hi, best, f_best, z, d_next, a_grid, V_tilde, theta, kappa):
    """Fella's bracket search for u(z - a', d') + V~(a') between the grid neighbours."""
    while hi - lo > _DG_REFINE_TOL:
        mid = 0.5 * (lo + hi)
        f_mid = _dg_utility(z - mid, d_next, theta, kappa) + _dg_interp(mid, a_grid, V_tilde)
        if f_mid > f_best:  # mid is better: the max lies on mid's side of best
            if mid < best:
                hi = best
            else:
                lo = best
            best, f_best = mid, f_mid
        else:  # mid is worse: the max lies on best's side of mid
            if mid < best:
                lo = mid
            else:
                hi = mid
    return best, f_best


@njit(cache=True, parallel=True)
def _dg_bellman_update(a_grid, d_grid, y_grid, V_tilde, theta, kappa, r, xi, phi):
    """New V at every state (a, d, y), Eq. (15), and V^{d'}, a'^{d'} on G_a x D x Y x D'."""
    n_a, n_d, n_y = V_tilde.shape
    V = np.empty((n_a, n_d, n_y))
    V_cond = np.empty((n_a, n_d, n_y, n_d))
    a_cond = np.empty((n_a, n_d, n_y, n_d))
    for idx in prange(n_y * n_d):
        i_y = idx // n_d
        i_d = idx % n_d
        # a' rises with z only for a given d' (Prop. 1), so each d' has its own start
        j_start = np.zeros(n_d, dtype=np.int64)
        for i_a in range(n_a):
            v_best = -np.inf
            for i_dn in range(n_d):
                d_next = d_grid[i_dn]
                z = _dg_cash_on_hand(a_grid[i_a], d_grid[i_d], y_grid[i_y], d_next, r, xi, phi)
                if z <= 0.0:  # d' is not affordable
                    v, a_next = _DG_PENALTY, 0.0
                else:
                    # Grid search over a' in G_a with c = z - a' > 0 (MONOTONE start)
                    j_best, v = j_start[i_dn], -np.inf
                    for j in range(j_start[i_dn], n_a):
                        if a_grid[j] >= z:
                            break
                        f = _dg_utility(z - a_grid[j], d_next, theta, kappa) + V_tilde[j, i_dn, i_y]
                        if f > v:
                            j_best, v = j, f
                    a_next = a_grid[j_best]
                    j_start[i_dn] = j_best
                    # REFINE between the grid neighbours of the best node
                    lo = a_grid[max(j_best - 1, 0)]
                    hi = min(a_grid[min(j_best + 1, n_a - 1)], z)
                    a_next, v = _dg_refine(
                        lo, hi, a_next, v, z, d_next, a_grid, V_tilde[:, i_dn, i_y], theta, kappa
                    )
                V_cond[i_a, i_d, i_y, i_dn] = v
                a_cond[i_a, i_d, i_y, i_dn] = a_next
                v_best = max(v_best, v)
            V[i_a, i_d, i_y] = v_best
    return V, V_cond, a_cond


class DurableGoodsVFI:
    """Fella's (2014) VFI1 for the DurableGoods model.

    `solve()` attaches the d'-contingent solution to the model:
        model.vfi_a_grid : asset grid of the solution (n_a points)
        model.V_cond     : V^{d'}(a, d, y),  shape (n_a, n_d, n_y, n_d)
        model.a_cond     : a'^{d'}(a, d, y), shape (n_a, n_d, n_y, n_d)
    which `DurableGoods.benchmark_controls` evaluates off the grid.
    """

    def __init__(self, econ_model, n_a: int | None = None, tol: float = 1e-5, max_iter: int = 1000):
        self.model = econ_model
        self.n_a = int(econ_model.vfi_grid_size if n_a is None else n_a)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.solve_time: float | None = None
        self.n_iter: int | None = None

    def solve(self, verbose: bool = False):
        from ecomodrl.models.durablegoods import asset_grid

        m = self.model
        p = m.params
        a_grid = asset_grid(self.n_a, m.a_max)
        d_grid, y_grid, P = m.d_grid, m.y_grid, m.P

        start = time.perf_counter()

        # Fella: initial guess V^0(a, d, y) = u(r a + y, d)
        a, d, y = np.meshgrid(a_grid, d_grid, y_grid, indexing="ij")
        V = _dg_utility(p.r * a + y, d, p.theta, p.kappa)

        for iteration in range(1, self.max_iter + 1):
            V_tilde = p.beta * V @ P.T  # beta E[V(a', d', y') | y], Eq. (11)
            V_new, V_cond, a_cond = _dg_bellman_update(
                a_grid, d_grid, y_grid, V_tilde, p.theta, p.kappa, p.r, p.xi, p.phi
            )
            # Fella: relative sup-norm distance
            distance = np.max(np.abs(V_new - V)) / np.max(np.abs(V))
            V = V_new
            if verbose:
                print(f"VFI iter {iteration:4d}: rel. sup-norm change {distance:.3e}")
            if distance < self.tol:
                break

        self.solve_time = time.perf_counter() - start
        self.n_iter = iteration

        m.vfi_a_grid = a_grid
        m.V_cond = V_cond
        m.a_cond = a_cond
        return V_cond, a_cond
