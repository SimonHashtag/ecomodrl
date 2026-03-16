import itertools
import logging
from typing import Tuple, Union

import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root
from scipy.stats import qmc, norm

from ecomodrl.utils.vpost_extrapolation import VPostExtrapolator


# ---------------------------------------------------------------------------
# Low-level Numba utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Portfolio Choice planning problem
# ---------------------------------------------------------------------------


class PlanningProblem:
    """
    Planning problem for a buffer-stock saver with portfolio choice.

    The model is written in normalized variables (by permanent income growth). 
    States are:
        - m_t: cash-on-hand (pre-decision), scalar
        - z_t: vector of return predictors for risky assets (AR(1) processes)

    Controls are:
        - c_t: consumption
        - ω_t: portfolio weights over [safe, risky_1, ..., risky_N], sum to 1
    """

    def __init__(
        self,
        gamma: float = 2.0,  # Coefficient of relative risk aversion
        beta: float = 0.95,  # Discount factor
        g: float = 0.01,  # Growth rate of permanent income
        sigma_tran: float = 0.1,  # Std. dev. of transitory shock (log)
        sigma_perm: float = 0.1,  # Std. dev. of permanent shock (log)
        unemp_prob: float = 0.005,  # Probability of unemployment
        theta_low: float = 0.3,  # Earnings in unemployment
        r: float = 0.02,  # Interest rate of risk-free asset
        sigma_z: Union[float, list, np.ndarray] = 0.1,  # Std dev of predictor shocks
        sigma_R: Union[float, list, np.ndarray] = 0.2,  # Std dev of return shocks
        rho_z: Union[float, list, np.ndarray] = 0.85,  # Persistence of predictors
        kappa_R: Union[float, list, np.ndarray] = 0.0,  # Intercept of asset return
        xi_R: Union[float, list, np.ndarray] = 0.025,  # Loading on predictor
        M: int = 4096,  # Target number of RQMC nodes (before antithetics)
        seed: int = 1996,  # Seed for RQMC
    ) -> None:
        # Preferences
        self.gamma = float(gamma)
        self.beta = float(beta)

        # Normalized income process
        self.r = float(r)
        self.g = float(g)
        self.R = 1.0 + self.r
        self.G = 1.0 + self.g
        self.sigma_tran = float(sigma_tran)
        self.sigma_perm = float(sigma_perm)

        # Unemployment
        self.unemp_prob = float(unemp_prob)
        self.theta_low = float(theta_low)

        # Mean adjustment so E[Theta] = 1
        self.mu_theta = (
            np.log(
                (1.0 - self.unemp_prob * self.theta_low) / (1.0 - self.unemp_prob)
            )
            - 0.5 * self.sigma_tran**2
        )

        # ----- Asset process params -----
        self.sigma_z = np.atleast_1d(sigma_z).astype(np.float64)
        self.sigma_R = np.atleast_1d(sigma_R).astype(np.float64)
        self.rho_z = np.atleast_1d(rho_z).astype(np.float64)
        self.kappa_R = np.atleast_1d(kappa_R).astype(np.float64)
        self.xi_R = np.atleast_1d(xi_R).astype(np.float64)

        self.n_risky = len(self.sigma_z)
        assert all(
            len(p) == self.n_risky
            for p in (self.sigma_R, self.rho_z, self.kappa_R, self.xi_R)
        ), "Asset parameters must have the same length."
        self.n_assets = self.n_risky + 1  # +1 safe asset

        # Seeding
        self.seed: int = int(seed)
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

        # Monte Carlo draws for random shocks
        self.build_rqmc(M, seed)

        # Precompute Γ' = G * Ψ' and Γ'^γ used in SDF
        self.Gamma_next = (self.G * self._psi_vals).astype(np.float64)
        self.SDF_factor = (1.0 / (self.Gamma_next**self.gamma)).astype(np.float64)

        # Buffers used to cut temporary allocations in sdf_moments
        self._pts_buf_shape = None
        self._pts_emp = None
        self._pts_un = None

    # ------------------------------------------------------------------
    # Random Shock Handling - RQMC
    # ------------------------------------------------------------------

    def build_rqmc(self, M: int = 4096, seed: int = 1996) -> None:
        """
        Build a scrambled Sobol' point set with antithetics for Randomized Quasi-Monte-Carlo Sampling (RQMC).

        Parameters
        ----------
        M : int
            Target number of Sobol points *before* antithetics. Internally we
            use the next power-of-two; final count is 2 * 2**ceil(log2 M) = M_rqmc.
        seed : int
            Seed for the scramble (change only if you want a new replicate).

        Stores
        ------
        self._eps_nodes : (M_rqmc, n_risky)
            ε for z' innovations.
        self._eta_nodes : (M_rqmc, n_risky)
            η for R' innovations.
        self._psi_vals : (M_rqmc,)
            Ψ' values (permanent income shocks).
        self._theta_vals : (M_rqmc,)
            Θ' values (transitory income shocks, lognormal branch).
        """
        d_base = 2 * self.n_risky + 1  # ε_1..N, η_1..N, ν (for Ψ)
        d = d_base + 1  # + ζ for Θ'

        eng = qmc.Sobol(d=d, scramble=True, seed=seed)

        # Use next power-of-two to avoid Sobol balance warnings
        m_exp = int(np.ceil(np.log2(max(1, M))))
        U = eng.random_base2(m_exp)  # (2**m_exp, d)

        # Antithetic points
        Ua = 1.0 - U
        U_all = np.vstack([U, Ua])  # (2 * 2**m_exp, d)

        # Numerical safety: avoid 0/1 before inverse-normal
        eps = np.finfo(float).eps
        U_all = np.clip(U_all, eps, 1.0 - eps)

        # Standard-normal nodes for CRN (reuse every expectation call)
        Xi = norm.ppf(U_all).astype(np.float64)

        n = self.n_risky
        self._M_rqmc = Xi.shape[0]
        self._eps_nodes = Xi[:, :n].astype(np.float64)
        self._eta_nodes = Xi[:, n : 2 * n].astype(np.float64)
        nu_nodes = Xi[:, 2 * n]  # Shocks for permanent income Ψ
        zeta_nodes = Xi[:, 2 * n + 1]  # Shocks for transitory income Θ

        # Ψ' = exp(-0.5 σ_Ψ^2 + σ_Ψ * ν)
        self._psi_vals = np.exp(
            -0.5 * self.sigma_perm**2 + self.sigma_perm * nu_nodes
        ).astype(np.float64)

        # Θ' (lognormal branch) = exp(μ_Θ + σ_Θ * ζ); μ_Θ set so E[Θ]=1
        self._theta_vals = np.exp(
            self.mu_theta + self.sigma_tran * zeta_nodes
        ).astype(np.float64)

    # ------------------------------------------------------------------
    # Grid setup and interpolation constructors
    # ------------------------------------------------------------------

    def make_m_grid(
        self,
        points: int = 30,
        span: Tuple[float, float] = (0.2, 5.0),
        dense_low: bool = True,
        p: int = 3,
    ) -> np.ndarray:
        """
        Construct the m-grid.

        Parameters
        ----------
        points : int
            Number of grid points.
        span : (float, float)
            [min, max] range of the grid.
        dense_low : bool
            If True, emphasize low m via a polynomial warp.
        p : int
            Exponent for the warp when dense_low is True.

        Returns
        -------
        1D ndarray of float
            Monotone m-grid.
        """
        if not dense_low:
            return np.linspace(span[0], span[1], points)
        u = np.linspace(0.0, 1.0, points)
        return span[0] + (span[1] - span[0]) * (u**p)

    def make_z_grids(self, points: int = 21, m: float = 3.0) -> list[np.ndarray]:
        """
        Construct predictor grids for each risky asset.

        Each z-grid is centered at 0 with ±m stationary std devs of the AR(1)
        process for that asset.
        """
        sd = self.sigma_z / np.sqrt(1.0 - self.rho_z**2)  # stationary std dev
        return [np.linspace(-m * sdi, m * sdi, points) for sdi in sd]

    def make_multilinear(
        self,
        m_grid: np.ndarray,
        z_grids: list[np.ndarray],
        values: np.ndarray,
    ) -> RegularGridInterpolator:
        """
        N-D linear interpolation on (m, z1, ..., zN).

        Parameters
        ----------
        m_grid : 1D ndarray
            Grid for m.
        z_grids : list of 1D ndarray
            Grids for z1, ..., zN.
        values : ndarray
            Values with shape
            (len(m_grid), len(z1_grid), ..., len(zN_grid)[, ...]).

        Returns
        -------
        RegularGridInterpolator
            Linear interpolator with out-of-bounds extrapolation enabled.
        """
        grid_tuple = (m_grid, *z_grids)
        return RegularGridInterpolator(
            grid_tuple,
            values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    # ------------------------------------------------------------------
    # Model primitives (preferences & technology)
    # ------------------------------------------------------------------

    def u(self, c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CRRA utility function."""
        if self.gamma == 1.0:
            return np.log(c)
        return c ** (1.0 - self.gamma) / (1.0 - self.gamma)

    def u_prime(self, c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Marginal utility of consumption."""
        return c ** (-self.gamma)

    def u_prime_inv(self, c: float) -> float:
        """Inverse marginal utility"""
        return c ** (-1 / self.gamma)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def zs_R_next(self, z: Union[float, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Map cached RQMC normal nodes to next-period predictors z' and returns R'.

        Parameters
        ----------
        z : float or 1D ndarray, shape (n_risky,)
            Current predictor state.

        Returns
        -------
        zprime : ndarray, shape (M_rqmc, n_risky)
            Next predictors for each asset.
        Rprime : ndarray, shape (M_rqmc, n_assets)
            Next gross returns for each asset: column 0 is risk-free,
            columns 1..N are risky.
        """
        z = np.asarray(z, dtype=np.float64)
        if self.n_risky == 1 and z.shape == ():
            z = z.reshape(1)
        assert z.shape == (self.n_risky,), "z must be scalar or shape (n_risky,)."

        zprime = (self.rho_z * z)[None, :] + self.sigma_z[None, :] * self._eps_nodes

        # Risky returns R_{j,t+1}
        locR = (self.kappa_R + self.xi_R * z)[None, :]
        R_risky = np.exp(locR + self.sigma_R[None, :] * self._eta_nodes)

        # Stack risk-free as column 0
        R_safe = np.full((R_risky.shape[0], 1), self.R, dtype=np.float64)
        Rprime = np.concatenate([R_safe, R_risky], axis=1)

        return zprime, Rprime

    def m_next(
        self,
        rs: Union[float, np.ndarray],
        Rp_next: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Next-period cash-on-hand.

        Employed:
            m_{t+1} = (R^p / Γ') * rs_t + Θ_{t+1}
        Unemployed:
            m_{t+1} = (R^p / Γ') * rs_t + θ_low
        """
        Rp_next = np.ascontiguousarray(Rp_next)
        M = Rp_next.shape[0]
        base = (Rp_next / self.Gamma_next[:M]) * float(rs)
        return base + self._theta_vals[:M], base + self.theta_low

    @staticmethod
    def consumption_from_share(a: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Map share a(m) to consumption level: c_t = a_t(m_t) * m_t.

        A small floor is imposed to avoid log/CRRA issues at c → 0.
        """
        return np.clip(a * m, 1e-10, None)

    # ------------------------------------------------------------------
    # Bellman equation
    # ------------------------------------------------------------------

    def bellman(
        self,
        c: float,
        w: np.ndarray,
        m: float,
        z_vec: np.ndarray,
        v: RegularGridInterpolator,
    ) -> float:
        """Calculate the Bellman equation at (m, z) given c, ω, and a value function."""
        rs = m - c
        zs_next, Rs_next = self.zs_R_next(z_vec)
        R_p_next = Rs_next @ w
        m_next_emp, m_next_unemp = self.m_next(rs, R_p_next)

        pts_emp = np.column_stack([m_next_emp, zs_next])
        pts_unemp = np.column_stack([m_next_unemp, zs_next])
        v_emp = v(pts_emp)
        v_unemp = v(pts_unemp)

        # Weighted expectation over employment states
        p_e = 1.0 - self.unemp_prob
        return self.u(c) + self.beta * (
            p_e * v_emp.mean() + self.unemp_prob * v_unemp.mean()
        )

    # ---------------------------
    # SDF moments and Euler Residuals
    # ---------------------------

    @staticmethod
    def _clamp_pts(policy: RegularGridInterpolator, pts: np.ndarray) -> np.ndarray:
        """
        Clamp query points to the domain of the interpolant in-place.
        """
        grids = getattr(policy, "grid", None) or getattr(policy, "points", None)
        if grids is not None:
            for j, g in enumerate(grids):
                pts[:, j] = np.clip(pts[:, j], float(g[0]), float(g[-1]))
        return pts

    @staticmethod
    def smooth_fisher_burmeister(
        a: np.ndarray,
        b: np.ndarray,
        mu: float = 1e-6,
    ) -> np.ndarray:
        """Smoothed Fisher–Burmeister function for complementarity conditions."""
        return np.sqrt(a**2 + b**2 + 2.0 * mu) - (a + b)

    def sdf_moments(
        self,
        rs: float,
        zs_next: np.ndarray,
        Rs_next: np.ndarray,
        w: np.ndarray,
        a_policy: RegularGridInterpolator,
        min_share: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute E[m_{t+1} * R_{i,t+1}] scaled by u'(c_t)/beta, i.e.

            g_i = E[ u'(c_{t+1}) / Γ_{t+1}^γ * R_{i,t+1} ],  i = 0..N.

        Parameters
        ----------
        rs : float
            Resources saved today (m - c).
        zs_next : ndarray, shape (M_rqmc, n_risky)
            Next return predictors.
        Rs_next : ndarray, shape (M_rqmc, n_assets)
            Next returns for each asset.
        w : ndarray, shape (n_assets,)
            Portfolio weights (safe first).
        a_policy : RegularGridInterpolator
            Policy for next-period consumption share a'(m', z').
        min_share : float
            Minimum share of m allocated to consumption (for stability).

        Returns
        -------
        g : 1D ndarray, shape (n_assets,)
            Marginal-utility–weighted returns (“SDF moments”) for each asset.
        """
        R_p_next = Rs_next @ w

        # Next states for employment and unemployment
        m_emp, m_un = self.m_next(rs, R_p_next)
        M, n_risky = zs_next.shape
        ncols = 1 + n_risky

        if self._pts_buf_shape != (M, ncols):
            self._pts_emp = np.empty((M, ncols), dtype=np.float64)
            self._pts_un = np.empty((M, ncols), dtype=np.float64)
            self._pts_buf_shape = (M, ncols)

        pts_emp, pts_un = self._pts_emp, self._pts_un
        pts_emp[:, 1:] = zs_next
        pts_un[:, 1:] = zs_next
        pts_emp[:, 0] = m_emp
        pts_un[:, 0] = m_un

        # Clamp and clip to ensure stability
        self._clamp_pts(a_policy, pts_emp)
        self._clamp_pts(a_policy, pts_un)
        a_emp = np.clip(a_policy(pts_emp), min_share, 1.0)
        a_un = np.clip(a_policy(pts_un), min_share, 1.0)
        c_emp = self.consumption_from_share(a_emp, m_emp)
        c_un = self.consumption_from_share(a_un, m_un)

        # SDFs
        upd_emp = self.u_prime(c_emp) * self.SDF_factor
        upd_un = self.u_prime(c_un) * self.SDF_factor
        weight = (1.0 - self.unemp_prob) * upd_emp + self.unemp_prob * upd_un

        return (weight[:, None] * Rs_next).mean(axis=0)

    def euler_residuals(
        self,
        rs: float,
        z: np.ndarray,
        c: float,
        w: np.ndarray,
        a_policy: RegularGridInterpolator,
        c_upper: float,
        xi: float | None = None,
        mu_fb: float = 1e-6,
    ) -> np.ndarray:
        """Residuals: [Euler, portfolio equalization on active face]."""
        zs_next, Rs_next = self.zs_R_next(z)
        g = self.sdf_moments(rs, zs_next, Rs_next, w, a_policy)
        gbar = float(np.dot(w, g))

        if xi is None:
            xi = gbar

        # Portfolio residuals: ω_i * (g_i - gbar) = 0
        fb_r_port = self.smooth_fisher_burmeister(w, xi - g, mu=mu_fb)

        # Euler residual
        r0 = self.u_prime(c) - self.beta * gbar
        r0 = self.smooth_fisher_burmeister(r0, c_upper - c, mu=0.0)
        return np.concatenate([[r0], fb_r_port])

    # ---------------------------
    # Policy evaluation & errors
    # ---------------------------

    def get_euler_errors(
        self,
        m_grid: np.ndarray,
        z_grids: list[np.ndarray],
        a_policy: RegularGridInterpolator,
        w_policy: RegularGridInterpolator,
        episode_len: int = 100,
    ):
        """
        Simulate Euler/KKT residuals along sample paths.

        For each starting m on the grid, and with initial predictor z set to
        the component-wise median of each z-grid, simulate for `episode_len`
        periods using the provided policies a(m, z) and ω(m, z).

        Returns
        -------
        errors : 1D ndarray
            log10(max-abs residual) per step (NaN in constrained region).
        states : dict
            'm' : ndarray (len(m_grid), T + 1 - burn_in)
            'z' : ndarray (len(m_grid), T + 1 - burn_in, n_risky)
        avg_return : float
            Average discounted utility over start nodes.
        """
        T = int(episode_len)
        n_risky = len(z_grids)

        # Component-wise median z as start state
        z_med = np.array([zg[len(zg) // 2] for zg in z_grids], dtype=float)

        n_nodes = len(m_grid)
        states_m = np.empty((n_nodes, T + 1), dtype=float)
        states_z = np.empty((n_nodes, T + 1, n_risky), dtype=float)
        euler_errors = np.empty((n_nodes, T), dtype=float)
        avg_return = 0.0

        def eval_w(m, z_vec):
            w = w_policy([[m, *z_vec]])[0]
            w = np.clip(w, 0.0, None)
            s = w.sum()
            if s > 0.0:
                return w / s
            return np.full(self.n_assets, 1.0 / self.n_assets)

        # Unpack params
        kappa = np.atleast_1d(self.kappa_R)
        xi = np.atleast_1d(self.xi_R)
        rho = np.atleast_1d(self.rho_z)
        sig_z = np.atleast_1d(self.sigma_z)  # persistent predictor innovation
        sig_R = np.atleast_1d(self.sigma_R)  # transitory return shock

        for i, m0 in enumerate(m_grid):
            m_path = np.empty(T + 1, dtype=float)
            z_path = np.empty((T + 1, n_risky), dtype=float)
            c_path = np.empty(T, dtype=float)

            m_path[0] = float(m0)
            z_path[0, :] = z_med

            U = 0.0
            disc = 1.0

            for t in range(T):
                m_t = float(m_path[t])
                z_t = z_path[t, :]

                # Policies
                a_t = a_policy([[m_t, *z_t]])[0]
                c_t = self.consumption_from_share(a_t, m_t)
                w_t = eval_w(m_t, z_t)

                # Returns: risky + safe
                eta = self.rng.standard_normal(n_risky)
                R_risky = np.exp(kappa + xi * z_t + sig_R * eta)
                R_safe = self.R
                R_all = np.concatenate(([R_safe], R_risky))
                Rp = float(np.dot(w_t, R_all))

                # Income shocks & permanent growth
                employed = self.rng.random() > self.unemp_prob
                if employed:
                    theta = np.exp(
                        self.mu_theta
                        + self.sigma_tran * self.rng.standard_normal()
                    )
                else:
                    theta = self.theta_low

                psi = np.exp(
                    -0.5 * self.sigma_perm**2
                    + self.sigma_perm * self.rng.standard_normal()
                )
                Gamma = self.G * psi

                # Next states
                rs = max(m_t - c_t, 0.0)
                m_next = (Rp / Gamma) * rs + theta
                eps_z = self.rng.standard_normal(n_risky)
                z_next = rho * z_t + sig_z * eps_z

                m_path[t + 1] = m_next
                z_path[t + 1, :] = z_next

                # Utility
                c_path[t] = c_t
                U += disc * self.u(c_t)
                disc *= self.beta

                # Euler/KKT residual at (m_t, z_t)
                c_upper = m_t  # consume-all cap
                if c_t >= c_upper - 1e-10:
                    # Filter out constrained region when computing statistics
                    euler_errors[i, t] = np.nan
                else:
                    R_vec = self.euler_residuals(
                        rs=rs,
                        z=z_t,
                        c=c_t,
                        w=w_t,
                        a_policy=a_policy,
                        c_upper=c_upper,
                        mu_fb=0.0,
                    )
                    euler_errors[i, t] = np.log10(
                        max(1e-16, float(np.max(np.abs(R_vec))))
                    )

            states_m[i, :] = m_path
            states_z[i, :, :] = z_path
            avg_return += U / n_nodes

        # Apply burn-in slicing only to distribution metrics.
        burn_in = int(T * 0.2)
        states = {
            "m": states_m[:, burn_in:],
            "z": states_z[:, burn_in:, :],
        }
        euler_errors_final = euler_errors[:, burn_in:]
        return euler_errors_final.ravel(), states, avg_return

    # -------------------------------------------------------------------
    # Policy evaluation: value function from policy
    # -------------------------------------------------------------------

    def V_from_a(
        self,
        m_grid: np.ndarray,
        z_grids: list[np.ndarray],
        a_policy: RegularGridInterpolator,
        w_policy: RegularGridInterpolator,
        conv: float = 1e-5,
    ) -> RegularGridInterpolator:
        """
        Compute the value function implied by a given (a, w) policy.

        Uses simple value iteration on the discrete grid and
        `VPostExtrapolator` to extrapolate in the rs dimension.
        """
        # Set up full grid
        grid_shape = (len(m_grid),) + tuple(len(g) for g in z_grids)
        z_indices = [range(len(g)) for g in z_grids]
        z_tuples = list(itertools.product(*z_indices))

        # Double-buffer for value iteration
        V_old = np.zeros(grid_shape, dtype=float)
        V_new = np.empty_like(V_old)

        while True:
            # Interior multilinear interpolator on the updated grid values
            v_interior = self.make_multilinear(m_grid, z_grids, V_old)

            # Fit CRRA-consistent (or log) tail at the top rs slice and wrap
            V_interp = VPostExtrapolator(
                v_post_grid=V_old,
                v_post_interior=v_interior,
                rs_grid=np.asarray(m_grid),
                z_grids=z_grids,
                gamma=self.gamma,
            )

            # Bellman update
            for z_idx_tuple in z_tuples:
                z_vec = np.array(
                    [z_grids[j][idx] for j, idx in enumerate(z_idx_tuple)],
                    dtype=float,
                )
                pt = np.empty((1, 1 + len(z_vec)), dtype=float)

                for im, m in enumerate(m_grid):
                    pt[0, 0] = m
                    pt[0, 1:] = z_vec
                    a_curr = float(a_policy(pt))
                    c = self.consumption_from_share(a_curr, m)
                    w = w_policy(pt)[0]
                    V_new[(im, *z_idx_tuple)] = self.bellman(c, w, m, z_vec, V_interp)

            max_diff = np.max(np.abs(V_new - V_old))
            if max_diff < conv:
                v_interior = self.make_multilinear(m_grid, z_grids, V_new)
                V_interp = VPostExtrapolator(
                    v_post_grid=V_new,
                    v_post_interior=v_interior,
                    rs_grid=np.asarray(m_grid),
                    z_grids=z_grids,
                    gamma=self.gamma,
                )
                return V_interp

            # If not converged, swap buffers
            V_old, V_new = V_new, V_old


# -------------------------------------------------------------------
# Solvers: standardized on share policy a(m)
# -------------------------------------------------------------------


class PlanningSolvers:
    def __init__(
        self,
        m_points: int = 61,
        rs_points: int = 61,
        z_points: int = 21,
        tol: float = 1e-5,
        conv: float = 1e-5,
        min_share: float = 1e-6,
    ) -> None:
        self.m_points = m_points
        self.rs_points = rs_points
        self.z_points = z_points
        self.tol = float(tol)
        self.conv = float(conv)
        self.min_share = float(min_share)

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _init_policy(
        self,
        model: PlanningProblem,
        warm_start: bool = False,
    ):
        """
        Initialize grids, consumption share array and portfolio weights.

        a_new stores α(m, z) = c / m (consumption share, not assets).
        w_new stores per-asset weights ω(m, z).
        """
        m_grid = model.make_m_grid(points=self.m_points, span=(0.2, 30.0))
        z_grids = model.make_z_grids(points=self.z_points)

        m_grid_shape = (len(m_grid),) + tuple(len(g) for g in z_grids)

        rs_grid = model.make_m_grid(points=self.rs_points, span=(0.0, 30.0))
        rs_grid_shape = (len(rs_grid),) + tuple(len(g) for g in z_grids)

        w_new = np.full(
            rs_grid_shape + (model.n_assets,),
            1.0 / model.n_assets,
            dtype=float,
        )
        w_policy = None

        if warm_start:
            try:
                from ecomodrl.benchmarks.bufferstock.problemclass import (
                    PlanningProblem as BufferStockProblem,
                )
                from ecomodrl.benchmarks.bufferstock.problemclass import (
                    PlanningSolvers as BufferStockSolvers,
                )

                buffer_stock = BufferStockProblem(
                    gamma=model.gamma,
                    beta=model.beta,
                    r=model.r,
                    g=model.g,
                    sigma_tran=model.sigma_tran,
                    sigma_perm=model.sigma_perm,
                    unemp_prob=model.unemp_prob,
                    theta_low=model.theta_low,
                )
                buffer_stock_solvers = BufferStockSolvers(
                    grid_points=100,
                    conv=1e-5,
                )
                a_bs_policy = buffer_stock_solvers.EGM(buffer_stock)
                a_bs = np.asarray(a_bs_policy(m_grid), float)

                a_new = np.broadcast_to(
                    a_bs.reshape((len(m_grid),) + (1,) * len(z_grids)),
                    m_grid_shape,
                ).copy()
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning(
                    "Warm start for portfolio choice failed with exception %s; "
                    "proceeding with naive initialization.",
                    exc,
                )
                a_new = np.full(m_grid_shape, 0.2, dtype=float)
        else:
            a_new = np.full(m_grid_shape, 0.2, dtype=float)

        a_policy = model.make_multilinear(m_grid, z_grids, a_new)

        return m_grid, rs_grid, z_grids, a_new, a_policy, w_new, w_policy

    def _check_convergence(
        self,
        a_old: np.ndarray,
        a_new: np.ndarray,
        w_old: np.ndarray,
        w_new: np.ndarray,
    ) -> Tuple[bool, float, float]:
        """
        Sup-norm check for both consumption share α and portfolio weights ω.

        Returns
        -------
        converged : bool
            True if both policies changed by at most self.conv.
        max_da : float
            sup-norm difference for α.
        max_dw : float
            sup-norm difference for ω.
        """
        max_da = float(np.max(np.abs(a_old - a_new)))
        max_dw = float(np.max(np.abs(w_old - w_new)))
        return (max_da <= self.conv and max_dw <= self.conv), max_da, max_dw

    def weights_to_mz_policy(
        self,
        model: PlanningProblem,
        rs_grid: np.ndarray,
        m_grid: np.ndarray,
        z_grids: list[np.ndarray],
        a_policy: RegularGridInterpolator,
        ws: np.ndarray,
    ) -> RegularGridInterpolator:
        """
        Map weights on (rs, z) grid to weights on (m, z) given a(m, z).

        This is a deterministic change of variables: for each (m, z) we
        compute rs = m * (1 - a(m, z)) and evaluate the weights policy
        defined on (rs, z).
        """
        w_policy_rsz = model.make_multilinear(rs_grid, z_grids, ws)

        grids = [m_grid] + z_grids
        mesh = np.meshgrid(*grids, indexing="ij")
        pts_mz = np.stack([g.ravel() for g in mesh], axis=1)

        a_vals = np.asarray(a_policy(pts_mz), float).ravel()
        m_vals = pts_mz[:, 0]
        rs_vals = m_vals * (1.0 - a_vals)

        pts_rsz = pts_mz.copy()
        pts_rsz[:, 0] = rs_vals

        w_flat = np.asarray(w_policy_rsz(pts_rsz), float)
        w_mz = w_flat.reshape(tuple(len(g) for g in grids) + (model.n_assets,))

        return model.make_multilinear(m_grid, z_grids, w_mz)

    @staticmethod
    def logits_to_weights(logits: np.ndarray) -> np.ndarray:
        """
        Additive log-ratio: logits in R^{N-1} → ω ∈ Δ^N.

        ALR inverse = softmax of [logits, 0].
        """
        e = np.exp(np.clip(logits, -50.0, 50.0))
        denom = max(1.0 + float(np.sum(e)), 1e-300)
        return np.concatenate([e / denom, np.array([1.0 / denom])])

    @staticmethod
    def weights_to_logits(w: np.ndarray) -> np.ndarray:
        """
        ω ∈ Δ^N  → logits in R^{N-1}.

        ALR forward = log(ω_1..N-1 / ω_N).
        """
        w = np.asarray(w, dtype=float)
        assert w.ndim == 1 and w.shape[0] >= 2, "w must be 1D with at least 2 elements."
        w = np.clip(w, 1e-15, None)
        w /= w.sum()
        logits = np.log(w[:-1]) - np.log(w[-1])
        return logits

    # ------------------------------------------------------------------
    # Nested Endogenous Gridpoint Method (NEGM)
    # ------------------------------------------------------------------

    def EGM(
        self,
        model: PlanningProblem,
        verbose: bool = False,
        max_iter: int = 100,
        warm_start: bool = True,
        mu_fb: float = 1e-4,
    ) -> Tuple[RegularGridInterpolator, RegularGridInterpolator]:
        """Compute optimal consumption share and portfolio choice policy by TI."""
        # --- Setup ----------------------------------------------------
        (
            m_grid,
            rs_grid,
            z_grids,
            a_new,
            a_policy,
            w_new,
            _,
        ) = self._init_policy(model, warm_start=warm_start)

        z_indices = [range(len(g)) for g in z_grids]

        # Precompute static policy grid structures for fast moments
        a_values = np.ascontiguousarray(a_new, dtype=np.float64) # make sure that a_values is in C-order and np.float64
        shape = np.array(a_values.shape, dtype=np.int64)

        # Strides to index flattened 1D-arrays
        strides = _c_strides(shape)
        m_grid_np = np.ascontiguousarray(m_grid, dtype=np.float64)

        # Flat-index/strides offsets for all hypercube corners
        z_corner_offs = _z_corner_offsets(strides, len(z_grids))

        # Scratch arrays for one z-slice
        c_buf = np.empty(len(rs_grid), dtype=float)
        m_buf = np.empty(len(rs_grid), dtype=float)

        # Typed lists for Numba kernels (z-grid collection and their Δ^{-1})
        z_typed = List()
        z_inv_deltas_typed = List()
        for zg in z_grids:
            zg_np = np.ascontiguousarray(zg, dtype=np.float64)
            z_typed.append(zg_np)
            z_inv_deltas_typed.append(1.0 / (zg_np[1:] - zg_np[:-1]))

        # Precompute Δm^{-1} once
        inv_m_deltas = 1.0 / (m_grid_np[1:] - m_grid_np[:-1])

        # --- Iteration ------------------------------------------------
        for it in range(max_iter):
            a_old = a_new.copy()
            w_old = w_new.copy()

            a_values = np.ascontiguousarray(a_new, dtype=np.float64)
            a_values_flat = a_values.ravel()

            # Simple μ homotopy for stability
            mu_now = max(1e-8, mu_fb * (0.5 ** (it // 2)))

            # Pointwise policy improvement: (c, ω) via local active-set solve
            for z_idx_tuple in itertools.product(*z_indices):
                z_vec = np.array(
                    [z_grids[j][idx] for j, idx in enumerate(z_idx_tuple)],
                    dtype=float,
                )
                zs_next, Rs_next = model.zs_R_next(z_vec)

                # Precompute z-structure for compiled moments at this z-slice
                zs_next_c = np.ascontiguousarray(zs_next, dtype=np.float64)
                Rs_next_c = np.ascontiguousarray(Rs_next, dtype=np.float64)

                base_z_flat, z_w = _precompute_z_interp(
                    zs_next_c,
                    z_typed,
                    z_inv_deltas_typed,
                    strides,
                )

                for irs, rs in enumerate(rs_grid):
                    # (A) ω-update at (rs, z)
                    if rs <= self.min_share:  # negligible savings ⇒ ω irrelevant; use uniform
                        w_star = np.full(
                            model.n_assets,
                            1.0 / model.n_assets,
                            dtype=float,
                        )
                    else:
                        # Warm-start
                        w0 = w_new[(irs, *z_idx_tuple, slice(None))]
                        if it % 15 == 0:
                            v0 = self.weights_to_logits(w0)  # logits in R^{N-1}

                            g0 = _sdf_moments_fast(
                                float(rs),
                                Rs_next_c,
                                w0,
                                model._theta_vals,
                                model.Gamma_next,
                                model.SDF_factor,
                                model.theta_low,
                                model.unemp_prob,
                                model.gamma,
                                self.min_share,
                                a_values_flat,
                                shape,
                                strides,
                                m_grid_np,
                                inv_m_deltas,
                                base_z_flat,
                                z_corner_offs,
                                z_w,
                            )
                            xi0 = float(np.max(g0))
                            x0 = np.concatenate([v0, [xi0]])

                            def residuals(x):
                                logits, xi = x[:-1], x[-1]
                                w_loc = self.logits_to_weights(logits)
                                g_loc = _sdf_moments_fast(
                                    float(rs),
                                    Rs_next_c,
                                    w_loc,
                                    model._theta_vals,
                                    model.Gamma_next,
                                    model.SDF_factor,
                                    model.theta_low,
                                    model.unemp_prob,
                                    model.gamma,
                                    self.min_share,
                                    a_values_flat,
                                    shape,
                                    strides,
                                    m_grid_np,
                                    inv_m_deltas,
                                    base_z_flat,
                                    z_corner_offs,
                                    z_w,
                                )
                                return model.smooth_fisher_burmeister(
                                    w_loc,
                                    xi - g_loc,
                                    mu=mu_now,
                                )

                            sol = root(
                                residuals,
                                x0,
                                method="hybr",
                                tol=self.tol,
                            )
                            w_star = self.logits_to_weights(sol.x[:-1])
                        else:
                            w_star = w0

                    # (B) consumption by Euler inversion at this (rs, z)
                    g = _sdf_moments_fast(
                        float(rs),
                        Rs_next_c,
                        w_star,
                        model._theta_vals,
                        model.Gamma_next,
                        model.SDF_factor,
                        model.theta_low,
                        model.unemp_prob,
                        model.gamma,
                        self.min_share,
                        a_values_flat,
                        shape,
                        strides,
                        m_grid_np,
                        inv_m_deltas,
                        base_z_flat,
                        z_corner_offs,
                        z_w,
                    )
                    gbar = float(w_star @ g)
                    w_new[(irs, *z_idx_tuple, slice(None))] = w_star

                    c = max(model.u_prime_inv(model.beta * gbar), self.min_share)
                    m_buf[irs] = rs + c
                    c_buf[irs] = c

                # ---- EGM mapping to the m-grid on this z slice ----
                order = np.argsort(m_buf)
                m_ord, c_ord = m_buf[order], c_buf[order]

                a_new[(slice(None), *z_idx_tuple)] = np.interp(
                    m_grid,
                    np.concatenate([[0.0], m_ord]),
                    np.concatenate([[1.0], c_ord / m_ord]),
                )

            # Update interpolant
            a_policy = model.make_multilinear(m_grid, z_grids, a_new)

            # --- Convergence check ---------------------------------------
            if it % 15 == 0:
                converged, max_diff_a, max_diff_w = self._check_convergence(
                    a_old,
                    a_new,
                    w_old,
                    w_new,
                )

                if verbose:
                    print(
                        f"[EGM iter {it}] max|Δa|={max_diff_a:.3e}, "
                        f"max|Δω|={max_diff_w:.3e}, converged={converged}",
                    )

                if converged:
                    break

        w_policy = self.weights_to_mz_policy(
            model,
            rs_grid,
            m_grid,
            z_grids,
            a_policy,
            w_new,
        )

        return a_policy, w_policy
