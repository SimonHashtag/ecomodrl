import numpy as np
from typing import Union
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.optimize import brentq
from scipy.stats import qmc, norm


# -------------------------------------------------------------------
# Portfolio Choice planning problem
# -------------------------------------------------------------------
class PlanningProblem:
    def __init__(
        self,
        gamma: float = 2.0,                                         # Coefficient of relative risk aversion
        beta: float = 0.95,                                         # Discount factor
        g: float = 0.01,                                            # Growth rate of permanent income
        sigma_tran: float = 0.1,                                    # Std. dev. of transitory shock (log)
        sigma_perm: float = 0.1,                                    # Std. dev. of permanent shock (log)
        unemp_prob: float = 0.005,                                  # Probability of unemployment
        theta_low: float = 0.3,                                     # Earnings in unemployment
        r: float = 0.02,                                            # Interest rate of risk-free asset
        sigma_perm_assets: Union[float, list, np.ndarray] = 0.1,    # Std. dev. of log permanent asset return shocks
        sigma_tran_assets: Union[float, list, np.ndarray] = 0.1,    # Std. dev. of log transitory asset return shocks
        rho_assets: Union[float, list, np.ndarray] = 0.99,          # Persistence of asset return predictor
        kappa_assets: Union[float, list, np.ndarray] = 0.0,         # Intercept of asset return predictor
        xi_assets: Union[float, list, np.ndarray] = 0.5,            # Loading on asset return predictor
        M: int = 4096,                                              # Number of RQMC nodes
        seed: int = 1996,                                           # Seed for RQMC
    ):
        # Preferences
        self.gamma, self.beta = gamma, beta

        # Normalized income process
        self.R, self.G = 1 + r, 1 + g
        self.sigma_tran, self.sigma_perm = sigma_tran, sigma_perm

        # Unemployment
        self.unemp_prob, self.theta_low = unemp_prob, theta_low

        # Mean adjustment so E[Theta]=1
        self.mu_z = np.log((1 - self.unemp_prob * self.theta_low) / (1 - self.unemp_prob)) - 0.5 * self.sigma_tran**2

        # ----- Asset process params -----
        self.sigma_perm_assets = np.atleast_1d(sigma_perm_assets).astype(float)
        self.sigma_tran_assets = np.atleast_1d(sigma_tran_assets).astype(float)
        self.rho_assets        = np.atleast_1d(rho_assets).astype(float)
        self.kappa_assets      = np.atleast_1d(kappa_assets).astype(float)
        self.xi_assets         = np.atleast_1d(xi_assets).astype(float)

        self.n_assets = len(self.sigma_perm_assets)
        assert all(len(p) == self.n_assets for p in
                   [self.sigma_tran_assets, self.rho_assets, self.kappa_assets, self.xi_assets]), \
               "Asset parameters must have same length."
        # TODO: enable common calibrations across assets, e.g. common std?

        self.build_rqmc(M, seed)

        self.Gamma_next = (self.G * self._psi_vals)     # Next-period permanent income growth factor
        self.SDF_denom = self.Gamma_next ** self.gamma  # Denominator of SDF
    
    def build_rqmc(self, M=4096, seed=1996):
        """
        Build a scrambled Sobol' point set with antithetics for Randomized Quasi-Monte-Carlo Sampling (RQMC).

        Parameters
        ----------
        M : int
            Number of Sobol' points *before* antithetics (final count is 2*M).
        seed : int
            Seed for the scramble (change only if you want a new randomized replicate).

        Stores:
        - self._eps_nodes : (2M, n_assets)      ε for z'
        - self._eta_nodes : (2M, n_assets)      η for R'
        - self._psi_vals  : (2M,)               Ψ' values
        - self._theta_vals: (2M,)               Θ' values (lognormal branch)
        - self._wL, self._wP: floats            per-node weights for Θ branches
        """

        d_base = 2 * self.n_assets + 1   # ε_1..N, η_1..N, ν (for Ψ)
        d = d_base + 1

        eng = qmc.Sobol(d=d, scramble=True, seed=seed)     # Owen-scrambled Sobol'
        U = eng.random(M)                                   # (M, d) in (0,1)
        Ua = 1.0 - U                                        # antithetic points
        U_all = np.vstack([U, Ua])                          # (2M, d)

        # numerical safety: avoid 0/1 before inverse-normal
        eps = np.finfo(float).eps
        U_all = np.clip(U_all, eps, 1.0 - eps)

        # Store standard-normal nodes for CRN (reuse every expectation call)
        Xi = norm.ppf(U_all).astype(np.float64)   # (2M, d)

        M2 = Xi.shape[0]
        n = self.n_assets
        eps_nodes = Xi[:, :n]         
        eta_nodes = Xi[:, n:2*n]   
        nu_nodes  = Xi[:, 2*n]                             # Shocks for permanent income Ψ
        zeta_nodes = Xi[:, 2*n + 1]                        # Shocks for transitory income Θ 

        # ---- Precompute Ψ' and Θ' values ----
        # Ψ' = exp(-0.5 σ_Ψ^2 + σ_Ψ * ν)
        psi_vals = np.exp(-0.5 * self.sigma_perm**2 + self.sigma_perm * nu_nodes)

        # Θ' (lognormal branch) = exp(μ_Z + σ_Θ * ζ); μ_Z already set to enforce E[Θ]=1
        theta_vals = np.exp(self.mu_z + self.sigma_tran * zeta_nodes)

        # ---- Store everything for CRN reuse ----
        self._M_rqmc = M2
        self._eps_nodes  = eps_nodes         # (2M, n_assets)
        self._eta_nodes  = eta_nodes         # (2M, n_assets)
        self._psi_vals   = psi_vals          # (2M,)
        self._theta_vals = theta_vals        # (2M,)

        # per-node weights for the two Θ branches (identical for all nodes)
        self._wL = (1.0 - self.unemp_prob) / M2
        self._wP = (self.unemp_prob) / M2

    def zs_R_next(self, z: Union[float, np.ndarray]):
        """
        Map cached RQMC normal nodes to next-period zs and returns Rprime.

        Returns
        -------
        zprime : (M, n_assets)     # next z for each asset
        Rprime : (M, n_assets)     # next gross returns for each asset
        """
        # map to next states/returns
        z = np.asarray(z, dtype=np.float64)
        assert z.shape == (self.n_assets,), "z must be scalar or shape (n_assets,)."

        zprime = (self.rho_assets * z)[None, :] + self.sigma_perm_assets[None, :] * self._eps_nodes

        locR   = (self.kappa_assets + self.xi_assets * z)[None, :]
        Rprime = np.exp(locR + self.sigma_tran_assets[None, :] * self._eta_nodes)

        return zprime, Rprime


    # ------------------------------------------------------------------
    # Grid setup, and interpolation constructors
    # ------------------------------------------------------------------

    def make_m_grid(self, points: int = 30, span=(0.2, 5.0)):
        return np.linspace(span[0], span[1], points)

    def make_z_grids(self, points: int = 21, m: float = 2.0):
        """
        Return list [z1_grid, ..., zN_grid], one per asset.
        Each grid is centered at 0 with ±m stationary std devs of the AR(1) process.
        """
        sd = self.sigma_perm_assets / np.sqrt(1.0 - self.rho_assets**2)  # stationary std dev
        return [np.linspace(-m*sdi, m*sdi, points) for sdi in sd]

    def make_spline(self, grid, y, extrapolate=True):
        """PCHIP spline (1D; used for m<->k inversions inside EGM)."""
        return PchipInterpolator(grid, y, extrapolate=extrapolate)

    def make_multilinear(self, m_grid, z_grids, values):
        """
        N-D linear interpolation on (m, z1, ..., zN).
        'values' must have shape (len(m_grid), len(z1_grid), ..., len(zN_grid)).
        """
        grid_tuple = (m_grid, *z_grids)
        expected = (len(m_grid),) + tuple(len(g) for g in z_grids)
        if values.shape != expected:
            raise ValueError(f"'values' has shape {values.shape}, expected {expected}.")
        return RegularGridInterpolator(
            grid_tuple, values, method='linear',
            bounds_error=False, fill_value=None
        )

    # ------------------------------------------------------------------
    # Model primitives (preferences & technology)
    # ------------------------------------------------------------------

    def u(self, c):
        """CRRA utility function"""
        return c ** (1 - self.gamma) / (1 - self.gamma) if self.gamma != 1 else np.log(c)

    def u_prime(self, c):
        """Marginal utility of consumption"""
        return c ** (-self.gamma)

    def u_prime_inv(self, c):
        """Inverse marginal utility"""
        return c ** (-1 / self.gamma)

    # ------------------------------------------------------------------
    # State transitions / resource accounting (single entrypoint: m_next)
    # ------------------------------------------------------------------

    def m_next(self, rs, R):
        """Next-period cash-on-hand: m_{t+1} = (R / (G Ψ_{t+1})) * rs_t + Θ_{t+1} with rs_t = m_t - c_t"""
        return R / self.Gamma_next * rs + self._theta_vals

    def consumption_from_share(self, a, m):
        """Map share a(m) to consumption level:c_t = a_t(m_t) * m_t."""
        return a * m

    # ---------------------------
    # Euler helpers
    # ---------------------------

    def demu_next(self, c_next: np.ndarray, R: np.ndarray) -> float:
        # β * E[ R / ((G*Ψ_{t+1})^γ) * u'(c_{t+1}) ]
        return self.beta * np.mean((R / self.SDF_denom) *  self.u_prime(c_next))

    def euler_residual(self, a: np.ndarray, m_t: float, z_t: Union[float, np.ndarray], 
                       policy: RegularGridInterpolator) -> float:
        c_t  = self.consumption_from_share(a[0], m_t)
        rs_t = m_t - c_t

        zs_next, Rs_next = self.zs_R_next(z_t)                  # (2M, n), (2M, n)
        R_eff = Rs_next @ a[1:]                                 # (2M,)

        m_next = self.m_next(rs_t, R_eff)                       # (2M,)
        pts    = np.column_stack([m_next, zs_next])             # (2M, 1+n)
        a_next = policy(pts)                                    # (2M,)

        c_next = self.consumption_from_share(a_next, m_next)    # (2M,)
        rhs    = self.demu_next(c_next, R_eff)                  # scalar
        return self.u_prime(c_t) - rhs
    
    def c_from_euler(self, c_next: np.ndarray) -> float:
        """Recover c_t from: u'(c_t) = (β R / G) E[ u'(c_{t+1}) / Ψ_{t+1} ]."""
        mu = self.demu_next(c_next)
        return self.u_prime_inv(mu)

    # ---------------------------
    # Policy evaluation & errors
    # ---------------------------
    def get_euler_errors(self, grid, policy, episodes=100):
        """
        Simulate Euler residuals along sample paths starting from each m on the grid,
        using the provided consumption-share policy a(·). Returns (errors, states, avg_return).
        """
        T = int(episodes)
        grid = np.asarray(grid, dtype=float)
        grid_len = grid.size

        states = np.empty((grid_len, T + 1), dtype=float)
        euler_errors = np.empty((grid_len, T), dtype=float)
        avg_return = 0.0

        for i, m0 in enumerate(grid):
            m_path = np.empty(T + 1, dtype=float)
            c_path = np.empty(T, dtype=float)
            m_path[0] = m0

            U = 0.0
            power = 1.0

            for t in range(T):
                a_t_share = float(policy(m_path[t]))  # consumption share
                c_t = self.consumption_from_share(a_t_share, m_path[t])

                # draw shocks
                if np.random.uniform(0.0, 1.0) < self.unemp_prob:
                    theta_draw = self.theta_low
                else:
                    theta_draw = np.exp(np.random.normal(self.mu_z, self.sigma_tran))
                psi_draw = np.exp(np.random.normal(-0.5 * self.sigma_perm**2, self.sigma_perm))

                m_path[t + 1] = self.R / (self.G * psi_draw) * (m_path[t] - c_t) + theta_draw

                c_path[t] = c_t
                U += power * self.u(c_t)
                power *= self.beta

            # Euler error along the realized path
            err = np.empty(T, dtype=float)
            for t, (c_t, m_t) in enumerate(zip(c_path, m_path[:-1])):
                a_t_share = c_t / m_t  # implied share
                err[t] = np.log10(abs(self.euler_residual(a_t_share, m_t, policy)))
            
            euler_errors[i, :] = err
            states[i, :] = m_path
            avg_return += U / grid_len

        return euler_errors.ravel(), states.ravel(), avg_return


#-------------------------------------------------------------------
# Solvers: standardized on share policy a(m)
#-------------------------------------------------------------------

class PlanningSolvers:
    def __init__(
        self,
        grid_points: int = 1000,
        tol: float = 1e-5,
        conv: float = 1e-5,
        min_share: float = 1e-6,
    ):
        self.grid_points = grid_points
        self.tol = tol
        self.conv = conv
        self.min_share = min_share


    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _init_policy(self, model: PlanningProblem):
        """Initialize m-grid, consumption-share array and spline policy a(m)."""
        grid = model.make_grid(self.grid_points)
        a_new = np.full_like(grid, 0.2)
        a_policy = model.make_spline(grid, a_new)
        return grid, a_new, a_policy

    def _check_convergence(self, a_old, a_policy, grid):
        """Check convergence of consumption-share policy on given grid."""
        a_new = a_policy(grid)
        return np.max(np.abs(a_old - a_new)) < self.conv, a_new


    # ------------------------------------------------------------------
    # Time iteration (TI)
    # ------------------------------------------------------------------
    
    def TI(self, model: PlanningProblem) -> PchipInterpolator:
        """Compute optimal consumption-share policy a(m) by time iteration."""
        # --- Setup -------------------------------------------------------
        grid, a_new, a_policy = self._init_policy(model)

        # --- Iteration ---------------------------------------------------
        while True:
            a_old = a_new.copy()

            # Pointwise policy improvement: root-finding on the Euler residual
            for i, m in enumerate(grid):

                # Define optimization objective
                euler_root = lambda a_candidate: model.euler_residual(a_candidate, m, a_policy)

                a_lo, a_hi = self.min_share, 1.0 - self.min_share
                f_lo, f_hi = euler_root(a_lo), euler_root(a_hi)

                if f_lo * f_hi > 0.0:
                    # No interior root: pick corner consistent with sign
                    a_new[i] = a_hi if f_lo > 0.0 else a_lo
                else:
                    # Sign change: brentq on the candidate share
                    a_new[i] = brentq(euler_root, a_lo, a_hi, xtol=self.tol)

            # Update policy
            a_policy = model.make_spline(grid, a_new)

            # --- Convergence check ---------------------------------------
            converged, a_new = self._check_convergence(a_old, a_policy, grid)
            if converged:
                return a_policy
