import numpy as np
import itertools
import logging
from typing import Union, Tuple
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root
from scipy.stats import qmc, norm

from ecomodrl.utils.vpost_extrapolation import VPostExtrapolator

# -------------------------------------------------------------------
# Portfolio Choice planning problem
# -------------------------------------------------------------------
class PlanningProblem:
    """
    Planning problem for a buffer-stock saver with portfolio choice.

    The model is written in normalized variables. States are:
     - m_t: cash-on-hand (pre-decision), scalar
     - z_t: vector of return predictors for risky assets (AR(1) processes)

     Controls are:
     - c_t: consumption
     - ω_t: portfolio weights over [safe, risky_1, ..., risky_N], sum to 1

      We keep the public API intact, but add seedable randomness, docstrings,
      and performance-oriented tweaks (vectorization & cached shocks).
      """
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
        sigma_z: Union[float, list, np.ndarray] = 0.1,              # Std. dev. of log permanent asset return shocks
        sigma_R: Union[float, list, np.ndarray] = 0.2,              # Std. dev. of log transitory asset return shocks
        rho_z: Union[float, list, np.ndarray] = 0.85,               # Persistence of asset return predictor
        kappa_R: Union[float, list, np.ndarray] = 0.0,              # Intercept of asset return predictor
        xi_R: Union[float, list, np.ndarray] = 0.025,               # Loading on asset return predictor
        M: int = 4096,                                              # Number of RQMC nodes
        seed: int = 1996,                                           # Seed for RQMC
    ):
        # Preferences
        self.gamma, self.beta = float(gamma), float(beta)

        # Normalized income process
        self.r, self.g = float(r), float(g)
        self.R, self.G = 1.0 + self.r, 1.0 + self.g
        self.sigma_tran, self.sigma_perm = float(sigma_tran), float(sigma_perm)

        # Unemployment
        self.unemp_prob, self.theta_low = float(unemp_prob), float(theta_low)

        # Mean adjustment so E[Theta]=1
        self.mu_theta = np.log((1 - self.unemp_prob * self.theta_low) / (1 - self.unemp_prob)) - 0.5 * self.sigma_tran**2

        # ----- Asset process params -----
        self.sigma_z = np.atleast_1d(sigma_z).astype(float)
        self.sigma_R = np.atleast_1d(sigma_R).astype(float)
        self.rho_z        = np.atleast_1d(rho_z).astype(float)
        self.kappa_R      = np.atleast_1d(kappa_R).astype(float)
        self.xi_R         = np.atleast_1d(xi_R).astype(float)

        self.n_risky = len(self.sigma_z)
        assert all(
            len(p) == self.n_risky
            for p in [
                self.sigma_R,
                self.rho_z,
                self.kappa_R,
                self.xi_R,
            ]
        ), "Asset parameters must have same length."
        self.n_assets = (
            self.n_risky + 1
        )

        # Seeding
        self.seed: int = int(seed)
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

        # Monte Carlo draws for random shocks
        self.build_rqmc(M, seed)

        # Precompute Γ' = G * Ψ' and Γ'^γ used in SDF
        self.Gamma_next = (self.G * self._psi_vals)     # Next-period permanent income growth factor
        self.SDF_denom = self.Gamma_next ** self.gamma  # Denominator of SDF

    # ------------------------------------------------------------------
    # Random Shock Handling - RQMC
    # ------------------------------------------------------------------
    
    def build_rqmc(self, M=4096, seed=1996):
        """
        Build a scrambled Sobol' point set with antithetics for Randomized Quasi-Monte-Carlo Sampling (RQMC).

        Parameters
        ----------
        M : int
            Target number of Sobol' points *before* antithetics. Internally we
            use the next power-of-two for balance; final count is 2 * 2**⌈log2 M⌉.
        seed : int
            Seed for the scramble (change only if you want a new randomized replicate).

        Stores:
        - self._eps_nodes : (2M, n_assets)      ε for z'
        - self._eta_nodes : (2M, n_assets)      η for R'
        - self._psi_vals  : (2M,)               Ψ' values
        - self._theta_vals: (2M,)               Θ' values (lognormal branch)
        """

        d_base = 2 * self.n_risky + 1   # ε_1..N, η_1..N, ν (for Ψ)
        d = d_base + 1                  # + ζ for Θ'

        eng = qmc.Sobol(d=d, scramble=True, seed=seed)     # Owen-scrambled Sobol'
        # Use next power-of-two to avoid Sobol’ balance warning
        m_exp = int(np.ceil(np.log2(max(1, M))))
        U = eng.random_base2(m_exp)                         # (2**m_exp, d)
        Ua = 1.0 - U                                        # antithetic points
        U_all = np.vstack([U, Ua])                          # (2M, d)

        # numerical safety: avoid 0/1 before inverse-normal
        eps = np.finfo(float).eps
        U_all = np.clip(U_all, eps, 1.0 - eps)

        # Store standard-normal nodes for CRN (reuse every expectation call)
        Xi = norm.ppf(U_all).astype(np.float64)   # (2M, d)

        n = self.n_risky
        eps_nodes = Xi[:, :n]         
        eta_nodes = Xi[:, n:2*n]   
        nu_nodes  = Xi[:, 2*n]                             # Shocks for permanent income Ψ
        zeta_nodes = Xi[:, 2*n + 1]                        # Shocks for transitory income Θ 

        # ---- Precompute Ψ' and Θ' values ----
        # Ψ' = exp(-0.5 σ_Ψ^2 + σ_Ψ * ν)
        psi_vals = np.exp(-0.5 * self.sigma_perm**2 + self.sigma_perm * nu_nodes)

        # Θ' (lognormal branch) = exp(μ_Z + σ_Θ * ζ); μ_Z already set to enforce E[Θ]=1
        theta_vals = np.exp(self.mu_theta + self.sigma_tran * zeta_nodes)

        # ---- Store everything for CRN reuse ----
        self._M_rqmc = Xi.shape[0]
        self._eps_nodes  = eps_nodes         # (2M, n_risky)
        self._eta_nodes  = eta_nodes         # (2M, n_risky)
        self._psi_vals   = psi_vals          # (2M,)
        self._theta_vals = theta_vals        # (2M,)


    # ------------------------------------------------------------------
    # Grid setup, and interpolation constructors
    # ------------------------------------------------------------------

    def make_m_grid(self, points: int = 30, span: tuple = (0.2, 5.0), dense_low: bool = True, p: int = 3):
        if dense_low == False:
            return np.linspace(span[0], span[1], points)
        u = np.linspace(0.0, 1.0, points)
        return span[0] + (span[1] - span[0]) * (u ** p)

    def make_z_grids(self, points: int = 21, m: float = 3.0):
        """
        Return list [z1_grid, ..., zN_grid], one per asset.
        Each grid is centered at 0 with ±m stationary std devs of the AR(1) process.
        """
        sd = self.sigma_z / np.sqrt(1.0 - self.rho_z**2)  # stationary std dev
        return [np.linspace(-m*sdi, m*sdi, points) for sdi in sd]

    def make_multilinear(self, m_grid, z_grids, values):
        """
        N-D linear interpolation on (m, z1, ..., zN).
        'values' must have shape (len(m_grid), len(z1_grid), ..., len(zN_grid)).
        """
        grid_tuple = (m_grid, *z_grids)
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
    # State transitions
    # ------------------------------------------------------------------

    def zs_R_next(self, z: Union[float, np.ndarray]):
        """
        Map cached RQMC normal nodes to next-period predictors z' and returns R'.

        Returns
        -------
        zprime : (M, n_assets)     # next z for each asset
        Rprime : (M, n_assets)     # next gross returns for each asset [0] = risk-free, [1:] = risky
        """
        # map to next states/returns
        z = np.asarray(z, dtype=np.float64)
        if self.n_risky == 1 and z.shape == ():
            z = z.reshape(1)
        assert z.shape == (self.n_risky,), "z must be scalar or shape (n_risky,)."

        zprime = (self.rho_z * z)[None, :] + self.sigma_z[None, :] * self._eps_nodes

        # Risky returns R_{j,t+1}
        locR   = (self.kappa_R + self.xi_R * z)[None, :]
        R_risky = np.exp(locR + self.sigma_R[None, :] * self._eta_nodes)

        # Stack risk-free as column 0
        R_safe = np.full((R_risky.shape[0], 1), self.R, dtype=np.float64)
        Rprime = np.concatenate([R_safe, R_risky], axis=1)

        return zprime, Rprime

    def m_next(self, rs: float, R: float):
        """Next-period cash-on-hand: 
        m_{t+1} = (R^p / Γ') * rs_t + Θ_{t+1} (employed)
        m_{t+1} = (R^p / Γ') * rs_t + θ_low   (unemployed)
        """
        base = R / self.Gamma_next * rs
        m_next_employed = base + self._theta_vals
        m_next_unemployed = base + self.theta_low
        return m_next_employed, m_next_unemployed
    
    def consumption_from_share(self, a, m):
        """Map share a(m) to consumption level:c_t = a_t(m_t) * m_t."""
        return np.clip(a * m, 1e-10, None)
    
    # ------------------------------------------------------------------
    # Bellman equation
    # ------------------------------------------------------------------
    
    def bellman(self, c: float, w: np.ndarray, m: float, z_vec: np.ndarray, v: RegularGridInterpolator):
        """Calculate the Bellman equation for the states (m,z) given a consumption level c, weights w, and a value function"""
        rs = m-c
        zs_next, Rs_next = self.zs_R_next(z_vec)
        R_p_next = Rs_next @ w
        m_next_emp, m_next_unemp = self.m_next(rs, R_p_next)    
        v_emp = v(np.column_stack([m_next_emp, zs_next]))
        v_unemp = v(np.column_stack([m_next_unemp, zs_next]))
        # Weighted expectation over employment states
        pE = (1.0 - self.unemp_prob)
        return self.u(c) + self.beta * (pE * v_emp.mean() + self.unemp_prob * v_unemp.mean())

    # ---------------------------
    # SDF moments and Euler Residuals
    # ---------------------------
    @staticmethod
    def _clamp_pts(policy: RegularGridInterpolator, pts: np.ndarray):
        grids = getattr(policy, "grid", None) or getattr(policy, "points", None)
        if grids is not None:
            for j, g in enumerate(grids):
                pts[:, j] = np.clip(pts[:, j], float(g[0]), float(g[-1]))
        return pts
    
    @staticmethod
    def smooth_fisher_burmeister(a, b, mu=1e-6):
        """Smoothed Fisher-Burmeister function for complementarity conditions."""
        return np.sqrt(a**2 + b**2 + 2.0*mu) - (a + b)
    

    def sdf_moments(
            self, 
            rs: float, 
            z: np.ndarray, 
            w: np.ndarray, 
            a_policy: RegularGridInterpolator, 
            min_share: float = 1e-6
            ) -> np.ndarray:
        """
        Return E[m_{t+1} * R_{i,t+1}] scaled by u'(c_t)/beta, i.e.
        g_i = E[ u'(c_{t+1}) / Gamma_{t+1}^gamma * R_{i,t+1} ] for i=0..N.
        Parameters
        ----------
        rs : float
            Resources saved today (m - c).
        z : (n_risky,) array
            Current return predictors.
        w : (n_assets,) array
            Portfolio weights (safe first).
        a_policy : RegularGridInterpolator
            Policy for next-period consumption share a'(m', z').
        min-share: float
            Minimum share of consumption (for optimizer stability)

        Returns
        -------
        g : (n_assets,) ndarray
            Marginal-utility–weighted returns (“SDF moments”) for each asset.
        """
        # Shocks -> (z', R') and R^{p'}
        zs_next, Rs_next = self.zs_R_next(z)                           # (2M, n_risky), (2M, n_assets)
        R_p_next = Rs_next @ w                                       # (2M,)

        # m' branches
        m_emp, m_unemp = self.m_next(rs, R_p_next)                     # (2M,), (2M,)

        # Build point arrays for interpolation
        # Clamp to ensure TI stability (extrapolation slopes can be nasty)
        pts_emp   = self._clamp_pts(a_policy, np.column_stack([m_emp,  zs_next]))
        pts_unemp = self._clamp_pts(a_policy, np.column_stack([m_unemp, zs_next]))


        # Next-period consumption via current policy a'(m',z'), then c' = a' * m'
        a_next_emp   = np.clip(a_policy(pts_emp),   min_share, 1.0)
        a_next_unemp = np.clip(a_policy(pts_unemp), min_share, 1.0)
        c_next_emp   = self.consumption_from_share(a_next_emp,   m_emp)
        c_next_unemp = self.consumption_from_share(a_next_unemp, m_unemp)

        # Common factor 1/(Γ'^γ)
        sdf_part = 1.0 / self.SDF_denom                                  # (2M,)

        upd_emp   = self.u_prime(c_next_emp)   * sdf_part                # (2M,)
        upd_unemp = self.u_prime(c_next_unemp) * sdf_part                # (2M,)

        # Weighted expectation over employment states
        pE = (1.0 - self.unemp_prob)
        g = pE * (upd_emp[:, None] * Rs_next).mean(axis=0) + \
            self.unemp_prob * (upd_unemp[:, None] * Rs_next).mean(axis=0)
        return g  # (n_assets,)
    
    def euler_residuals(
            self, 
            rs: float, 
            z: np.ndarray, 
            c: float, 
            w: np.ndarray, 
            a_policy: RegularGridInterpolator, 
            c_upper: float, 
            xi: float | None = None, 
            mu_fb: float = 1e-6
            ) -> np.ndarray:
        """Residuals: [Euler, portfolio equalization on active face]."""

        g = self.sdf_moments(rs, z, w, a_policy)       
        gbar = float(np.dot(w, g))

        if xi is None:
            xi = gbar

        # Portfolio residuals: ω_i * (g_i - gbar) = 0
        fb_r_port = self.smooth_fisher_burmeister(w, xi - g, mu=mu_fb)

        # Euler Residual
        r0 = self.u_prime(c) - self.beta * gbar
        r0 = self.smooth_fisher_burmeister(r0, c_upper-c, mu=0.0)
        return np.concatenate([[r0], fb_r_port])
    
    # ---------------------------
    # Policy evaluation & errors
    # ---------------------------

    def get_euler_errors(
            self, 
            m_grid: np.ndarray, 
            z_grids: np.ndarray, 
            a_policy: RegularGridInterpolator, 
            w_policy: RegularGridInterpolator, 
            episodes: int=100):
        """
        Simulate Euler/KKT residuals along sample paths starting from each m on the grid,
        with the initial predictor z set to the component-wise median of each z-grid.
        Uses the provided consumption-share policy a(m,z) and portfolio policy ω(m,z).

        Returns
        -------
        errors : (len(m_grid)*episodes,)  vector of log10(max-abs residual) per step
        states : {'m': (len(m_grid), T+1), 'z': (len(m_grid), T+1, n_risky)}
        avg_return : float, average discounted utility over start nodes
        """

        T = int(episodes)
        n_risky = len(z_grids)

        # component-wise median z as start state
        z_med = np.array([zg[len(zg)//2] for zg in z_grids], dtype=float)

        n_nodes = len(m_grid)
        states_m = np.empty((n_nodes, T + 1), dtype=float)
        states_z = np.empty((n_nodes, T + 1, n_risky), dtype=float)
        euler_errors = np.empty((n_nodes, T), dtype=float)
        avg_return = 0.0

        def eval_w(m, z_vec):
            w = w_policy([[m, *z_vec]])[0]
            w = np.clip(w, 0.0, None)
            s = w.sum()
            return w / s if s > 0 else np.full(self.n_assets, 1.0 / self.n_assets)

        # unpack params
        kappa = np.atleast_1d(self.kappa_R)
        xi    = np.atleast_1d(self.xi_R)
        rho   = np.atleast_1d(self.rho_z)
        sig_z = np.atleast_1d(self.sigma_z)   # persistent predictor innovation
        sig_R = np.atleast_1d(self.sigma_R)   # transitory return shock

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

                # policies
                a_t = a_policy([[m_t, *z_t]])[0]
                c_t = self.consumption_from_share(a_t, m_t)
                w_t = eval_w(m_t, z_t)

                # returns: risky + safe
                eta = self.rng.standard_normal(n_risky)
                R_risky = np.exp(kappa + xi * z_t + sig_R * eta)
                R_safe  = self.R
                R_all   = np.concatenate(([R_safe], R_risky))
                Rp      = float(np.dot(w_t, R_all))

                # income shocks & permanent growth
                employed = (self.rng.random() > self.unemp_prob)
                theta = (np.exp(self.mu_theta + self.sigma_tran * self.rng.standard_normal())
                        if employed else self.theta_low)
                psi   =  np.exp(-0.5 * self.sigma_perm**2 + self.sigma_perm * self.rng.standard_normal())
                Gamma = self.G * psi

                # next states
                rs = max(m_t - c_t, 0.0)
                m_next = (Rp / Gamma) * rs + theta
                eps_z  = self.rng.standard_normal(n_risky)
                z_next = rho * z_t + sig_z * eps_z

                m_path[t + 1] = m_next
                z_path[t + 1, :] = z_next

                # utility
                c_path[t] = c_t
                U += disc * self.u(c_t)
                disc *= self.beta

                # Euler/KKT residual at (m_t, z_t) 
                c_upper = m_t  # consume-all cap
                if c_t >= c_upper - 1e-10:
                    euler_errors[i,t] = np.nan # Filter in constrained area
                else:
                    R_vec = self.euler_residuals(rs=rs, z=z_t, c=c_t, w=w_t,
                                                a_policy=a_policy, c_upper=c_upper, mu_fb=0.0)
                    # Log10 of maximum absolute residual along all state dimensions
                    euler_errors[i, t] = np.log10(max(1e-16, float(np.max(np.abs(R_vec)))))

            states_m[i, :] = m_path
            states_z[i, :, :] = z_path
            avg_return += U / n_nodes

        states = {'m': states_m, 'z': states_z}
        return euler_errors.ravel(), states, avg_return
    
    #-------------------------------------------------------------------
    # Policy evaluation: value function from policy
    #-------------------------------------------------------------------

    def V_from_a(
            self, 
            m_grid: np.ndarray, 
            z_grids: np.ndarray, 
            a_policy: RegularGridInterpolator, 
            w_policy: RegularGridInterpolator, 
            conv: float = 1e-5) -> RegularGridInterpolator:
        # Set up full grid
        grid_shape = (len(m_grid), ) + tuple(len(g) for g in z_grids)
        z_indices = [range(len(g)) for g in z_grids]
        z_tuples = list(itertools.product(*z_indices))
        # Double-buffer for value iteration
        V_old = np.zeros(grid_shape, dtype=float)
        V_new = np.empty_like(V_old)
        # Iterate
        while(True):
            # Interior multilinear interpolator on the updated grid values
            v_post_interior = self.make_multilinear(m_grid, z_grids, V_old)
            # Fit CRRA-consistent (or log) tail at the top rs slice and wrap:
            V_interp = VPostExtrapolator(
                v_post_grid=V_old,
                v_post_interior=v_post_interior,
                rs_grid=np.asarray(m_grid),
                z_grids=z_grids,
                gamma=self.gamma,
            )
            # Bellman update
            for z_idx_tuple in z_tuples:
                z_vec = np.array([z_grids[j][idx] for j, idx in enumerate(z_idx_tuple)], dtype=float)
                for im, m in enumerate(m_grid):              
                    a_curr   = a_policy(np.column_stack([m,  z_vec]))[0]
                    c = self.consumption_from_share(a_curr, m)
                    w = w_policy(np.column_stack([m,  z_vec]))[0]
                    V_new[(im, *z_idx_tuple)] = self.bellman(c, w, m, z_vec, V_interp)
            max_diff = np.max(np.abs(V_new - V_old))
            # Convergence check
            if max_diff < conv:
                v_post_interior = self.make_multilinear(m_grid, z_grids, V_new)
                V_interp = VPostExtrapolator(
                    v_post_grid=V_new,
                    v_post_interior=v_post_interior,
                    rs_grid=np.asarray(m_grid),
                    z_grids=z_grids,
                    gamma=self.gamma,
                )
                return v_post_interior
            # If not converged, swap buffers
            V_old, V_new = V_new, V_old


#-------------------------------------------------------------------
# Solvers: standardized on share policy a(m)
#-------------------------------------------------------------------

class PlanningSolvers:
    def __init__(
        self,
        m_points: int = 61,
        rs_points: int = 61,
        z_points: int = 21,
        tol: float = 1e-5,
        conv: float = 1e-5,
        min_share: float = 1e-6,
    ):
        self.m_points = m_points
        self.rs_points = rs_points
        self.z_points = z_points
        self.tol = float(tol)
        self.conv = float(conv)
        self.min_share = float(min_share)
    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------   
    
    def _init_policy(self, model: PlanningProblem, warm_start: bool = False):
        """Initialize grids, consumption share array and portfolio logits.

        a_new stores α(m,z)=c/m (consumption share, not assets!).  
        w_logits_new stores per-asset logits for ω(m,z).          
        """
        m_grid = model.make_m_grid(points=self.m_points, span=(0.2, 30.))
        z_grids = model.make_z_grids(points=self.z_points)
        m_grid_shape = (len(m_grid),) + tuple(len(g) for g in z_grids)
        rs_grid = model.make_m_grid(points=self.rs_points, span=(0.0, 30.))
        rs_grid_shape = (len(rs_grid),) + tuple(len(g) for g in z_grids)
        w_new = np.full(rs_grid_shape + (model.n_assets,), 1.0 / model.n_assets, dtype=float)
        w_policy = None

        if warm_start:
            try:
                from ecomodrl.benchmarks.bufferstock.problemclass import PlanningProblem as BufferStockProblem
                from ecomodrl.benchmarks.bufferstock.problemclass import PlanningSolvers as BufferStockSolvers
                buffer_stock = BufferStockProblem(
                    gamma = model.gamma,         
                    beta = model.beta,         
                    r = model.r,            
                    g = model.g,            
                    sigma_tran = model.sigma_tran,   
                    sigma_perm = model.sigma_perm,    
                    unemp_prob = model.unemp_prob,  
                    theta_low = model.theta_low,     
                )
                buffer_stock_solvers = BufferStockSolvers(grid_points=100, conv=1e-5)
                a_bs_policy = buffer_stock_solvers.EGM(buffer_stock)
                a_bs = np.asarray(a_bs_policy(m_grid), float)
                a_new = np.broadcast_to(a_bs[:, None], (len(m_grid),) + tuple(len(g) for g in z_grids)).copy()
            except Exception as e:
                logging.warning(f"Warm start for portfolio choice failed with exception {e}; proceeding with naive initialization.")
                warm_start = False
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
    ) -> Tuple[bool, float, float, float]:
        """
        Sup-norm check for both consumption share α, portfolio weights ω and post-decision value functions.
        """
        max_da = float(np.max(np.abs(a_old - a_new)))
        max_dw = float(np.max(np.abs(w_old - w_new)))
        return (max_da <= self.conv and max_dw <= self.conv), max_da, max_dw
    
    def weights_to_mz_policy(
            self, 
            model: PlanningProblem, 
            rs_grid: np.ndarray, 
            m_grid: np.ndarray, 
            z_grids: np.ndarray, 
            a_policy: RegularGridInterpolator, 
            ws: np.ndarray):
        """Map weights on post_decision grid to weights on pre-decision grid (m,z) for given consumption and asset allocation policy and return as multilinear interpolator"""
        w_policy = model.make_multilinear(rs_grid, z_grids, ws)
        m_grid_shape = (len(m_grid),) + tuple(len(g) for g in z_grids)
        w_mz = np.zeros(m_grid_shape + (model.n_assets,), dtype=float)
        z_indices = [range(len(g)) for g in z_grids]
        z_tuples = list(itertools.product(*z_indices))
        for im, m in enumerate(m_grid):
             for z_idx_tuple in z_tuples:
                z_vec = np.array([z_grids[j][idx] for j, idx in enumerate(z_idx_tuple)], dtype=float)
                a = a_policy([[m, *z_vec]])[0]
                rs = m * (1.0 - a)
                w_mz[(im, *z_idx_tuple, slice(None))] = w_policy([[rs, *z_vec]])[0]
        
        w_policy_mz = model.make_multilinear(m_grid, z_grids, w_mz)
        return w_policy_mz

    def logits_to_weights(self, logits):
        # Additive log-ratio: logits in R^{N-1} → ω ∈ Δ^N  (ALR inverse = softmax of [logits, 0])
        e = np.exp(np.clip(logits, -50, 50))
        denom = max(1.0 + np.sum(e), 1e-300)
        return np.concatenate([e / denom, [1.0 / denom]])
    
    def weights_to_logits(self, w):
        # ω ∈ Δ^N  → logits in R^{N-1}  (ALR forward = log(ω_1..N-1 / ω_N))
        w = np.asarray(w, dtype=float)
        assert w.ndim == 1 and w.shape[0] >= 2, "w must be 1D with at least 2 elements."
        w = np.clip(w, 1e-15, None) 
        w /= w.sum()
        logits = np.log(w[:-1]) - np.log(w[-1])
        return logits
    
    # ------------------------------------------------------------------
    # Nested Endogenous Gridpoint Method (NEGM)
    # ------------------------------------------------------------------
    def EGM(self, 
        model: PlanningProblem, 
        verbose: bool = False,
        max_iter: int = 100,
        warm_start: bool = True,
        mu_fb: float = 1e-4,
        ) -> Tuple[RegularGridInterpolator, RegularGridInterpolator]:
        """Compute optimal consumption share and portfolio choice policy by TI."""
        # --- Setup ----------------------------------------------------
        m_grid, rs_grid, z_grids, a_new, a_policy, w_new, _ = self._init_policy(model, warm_start=warm_start)
        z_indices = [range(len(g)) for g in z_grids]

        # scratch arrays for one z-slice
        c_buf = np.empty(len(rs_grid), dtype=float)
        m_buf = np.empty(len(rs_grid), dtype=float)

        # --- Iteration ------------------------------------------------
        for it in range(1, max_iter + 1):
            a_old = a_new.copy()
            w_old = w_new.copy()

            # simple μ homotopy for stability
            mu_now = max(1e-8, mu_fb * (0.5 ** (it // 2)))

            # Pointwise policy improvement: (c, ω) via local active-set solve
            for z_idx_tuple in itertools.product(*z_indices):
                z_vec = np.array([z_grids[j][idx] for j, idx in enumerate(z_idx_tuple)], dtype=float)

                for irs, rs in enumerate(rs_grid):
                    # (A) ω-update at (rs, z)
                    if rs <= self.min_share:           # negligible savings ⇒ ω irrelevant; use uniform
                        w_star = np.full(model.n_assets, 1.0 / model.n_assets, dtype=float) # TODO: replace model.n_assets
                        g = model.sdf_moments(rs=rs, z=z_vec, w=w_star, a_policy=a_policy)
                        gbar = float(w_star @ g)
                    else:
                        # Warm-start
                        w0 = w_new[(irs, *z_idx_tuple, slice(None))]
                        v0 = self.weights_to_logits(w0)  # logits in R^{N-1}
                        g0 = model.sdf_moments(rs=rs, z=z_vec, w=w0, a_policy=a_policy)     
                        xi0 = float(w0 @ g0)                 
                        x0 = np.concatenate([v0, [xi0]])

                        def residuals(x):
                            logits, xi = x[:-1], x[-1]
                            w   = self.logits_to_weights(logits)  # ω ∈ Δ^N (safe + risky)
                            g_loc = model.sdf_moments(rs, z_vec, w, a_policy)

                            return model.smooth_fisher_burmeister(w, xi - g_loc, mu=mu_now)

                        sol = root(residuals, x0, method='hybr', tol=self.tol)
                        #if sol.success and np.all(np.isfinite(sol.x)):
                        w_star = self.logits_to_weights(sol.x[:-1])
                        # reuse the last g via one more cheap call (small overhead)
                        g = model.sdf_moments(rs=rs, z=z_vec, w=w_star, a_policy=a_policy)
                        gbar = float(w_star @ g)
                    w_new[(irs, *z_idx_tuple, slice(None))] = w_star  

                    # (B) consumption by Euler inversion at this (rs,z)
                    c = max(model.u_prime_inv(model.beta * gbar), self.min_share)
                    m_buf[irs] = rs + c
                    c_buf[irs] = c
                # ---- EGM mapping to the m-grid on this z slice ----
                order = np.argsort(m_buf)
                m_ord, c_ord = m_buf[order], c_buf[order]
                a_new[(slice(None), *z_idx_tuple)] = np.interp(m_grid, np.concatenate([[0.0], m_ord]), np.concatenate([[1.0], c_ord/m_ord]))

            # Update interpolants
            a_policy = model.make_multilinear(m_grid, z_grids, a_new)

            # --- Convergence check ---------------------------------------
            converged, max_diff_a, max_diff_w = self._check_convergence(a_old, a_new, w_old, w_new)

            if verbose:
                print(
                    f"[EGM iter {it}] max|Δa|={max_diff_a:.3e}, "
                    f"max|Δω|={max_diff_w:.3e}, converged={converged}"
                )

            if converged:
                break
        
        w_policy = self.weights_to_mz_policy(model, rs_grid, m_grid, z_grids, a_policy, w_new)

        return a_policy, w_policy 


