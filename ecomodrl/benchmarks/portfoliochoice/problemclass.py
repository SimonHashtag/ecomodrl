import numpy as np
import time
import itertools
import logging
from typing import Union, Tuple, List
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root, minimize, minimize_scalar
from scipy.stats import qmc, norm

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
        sigma_perm_assets: Union[float, list, np.ndarray] = 0.1,    # Std. dev. of log permanent asset return shocks
        sigma_tran_assets: Union[float, list, np.ndarray] = 0.2,    # Std. dev. of log transitory asset return shocks
        rho_assets: Union[float, list, np.ndarray] = 0.8,           # Persistence of asset return predictor
        kappa_assets: Union[float, list, np.ndarray] = 0.0,         # Intercept of asset return predictor
        xi_assets: Union[float, list, np.ndarray] = 0.01,           # Loading on asset return predictor
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
        self.mu_z = np.log((1 - self.unemp_prob * self.theta_low) / (1 - self.unemp_prob)) - 0.5 * self.sigma_tran**2

        # ----- Asset process params -----
        self.sigma_perm_assets = np.atleast_1d(sigma_perm_assets).astype(float)
        self.sigma_tran_assets = np.atleast_1d(sigma_tran_assets).astype(float)
        self.rho_assets        = np.atleast_1d(rho_assets).astype(float)
        self.kappa_assets      = np.atleast_1d(kappa_assets).astype(float)
        self.xi_assets         = np.atleast_1d(xi_assets).astype(float)

        self.n_risky = len(self.sigma_perm_assets)
        assert all(
            len(p) == self.n_risky
            for p in [
                self.sigma_tran_assets,
                self.rho_assets,
                self.kappa_assets,
                self.xi_assets,
            ]
        ), "Asset parameters must have same length."
        self.n_assets = (
            self.n_risky + 1
        )

        self.seed: int = int(seed)
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

        self.build_rqmc(M, seed)

        # Precompute Γ' = G * Ψ' and Γ'^γ used in SDF
        self.Gamma_next = (self.G * self._psi_vals)     # Next-period permanent income growth factor
        self.SDF_denom = self.Gamma_next ** self.gamma  # Denominator of SDF
    
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
        theta_vals = np.exp(self.mu_z + self.sigma_tran * zeta_nodes)

        # ---- Store everything for CRN reuse ----
        self._M_rqmc = Xi.shape[0]
        self._eps_nodes  = eps_nodes         # (2M, n_assets)
        self._eta_nodes  = eta_nodes         # (2M, n_assets)
        self._psi_vals   = psi_vals          # (2M,)
        self._theta_vals = theta_vals        # (2M,)

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

        zprime = (self.rho_assets * z)[None, :] + self.sigma_perm_assets[None, :] * self._eps_nodes

        # Risky returns R_{j,t+1}
        locR   = (self.kappa_assets + self.xi_assets * z)[None, :]
        R_risky = np.exp(locR + self.sigma_tran_assets[None, :] * self._eta_nodes)

        # Stack risk-free as column 0
        R_safe = np.full((R_risky.shape[0], 1), self.R, dtype=np.float64)
        Rprime = np.concatenate([R_safe, R_risky], axis=1)

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
    # State transitions / resource accounting (single entrypoint: m_next)
    # ------------------------------------------------------------------

    def m_next(self, rs, R):
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
    # Post-decision Bellman equation
    # ------------------------------------------------------------------
    def pd_bellman(self, omega, a, z_vec, c_share_policy, v_post, min_share=1e-6):
        """Calculate the post-decision value function V^{+}(a, z) = max_{omega} E[u(c) + beta * V^{+}(a', z') ]
        given a consumption policy and asset weights.
        """
        # Shocks -> (z', R') and R^{p'}
        zs_next, Rs_next = self.zs_R_next(z_vec)                           # (2M, n_risky), (2M, n_assets)
        R_p_next = Rs_next @ omega                                       # (2M,)

        # m' branches
        m_emp, m_unemp = self.m_next(a, R_p_next)                     # (2M,), (2M,)

        # Build point arrays for interpolation
        pts_emp   = np.column_stack([m_emp,  zs_next])                   # (2M, 1+n_risky)
        pts_unemp = np.column_stack([m_unemp, zs_next])
        #pts_emp   = self._clamp_pts(c_policy, np.column_stack([m_emp,  zs_next]))
        #pts_unemp = self._clamp_pts(c_policy, np.column_stack([m_unemp, zs_next]))


        # Next-period consumption via current policy a'(m',z'), then c' = a' * m'
        c_share_next_emp   = np.clip(c_share_policy(pts_emp),   min_share, 1.0 - min_share)
        c_share_next_unemp = np.clip(c_share_policy(pts_unemp), min_share, 1.0 - min_share)
        c_next_emp   = self.consumption_from_share(c_share_next_emp,   m_emp)
        c_next_unemp = self.consumption_from_share(c_share_next_unemp, m_unemp)

        # Post-decision capital
        a_next_emp = m_emp - c_next_emp
        a_next_unemp = m_unemp - c_next_unemp

        # Utility
        u_emp   = self.u(c_next_emp)                  # (2M,)
        u_unemp = self.u(c_next_unemp)                 # (2M,)

        # Post-decision value
        v_post_emp = v_post(np.column_stack([a_next_emp, zs_next]))
        v_post_unemp = v_post(np.column_stack([a_next_unemp, zs_next]))

        # Weighted expectation over employment states
        pE = (1.0 - self.unemp_prob)
        obj = pE * (u_emp + self.beta * v_post_emp).mean() + \
            self.unemp_prob * (u_unemp + self.beta * v_post_unemp).mean()
        return float(obj)
    
    def bellman(self, c, m, z_vec, v_post):
        """Calculate the Bellman equation for the states (m,z) given a consumption level c and a post-decision-value function"""
        a = m-c
        return self.u(c) + self.beta * v_post([[a, *z_vec]])

    # ---------------------------
    # g_i expectations for portfolio FOCs
    # ---------------------------
    @staticmethod
    def _clamp_pts(policy, pts):
        grids = getattr(policy, "grid", None) or getattr(policy, "points", None)
        if grids is not None:
            for j, g in enumerate(grids):
                pts[:, j] = np.clip(pts[:, j], float(g[0]), float(g[-1]))
        return pts
    

    def g_vector(self, rs_t: float, z_t: np.ndarray, 
                 omega: np.ndarray, a_policy: RegularGridInterpolator) -> np.ndarray:
        """
        Compute g_i = E[ u'(c') / Γ'^γ * R_i' ]  (i=0..N), using current policies for c'(m', z').
        """
        # Shocks -> (z', R') and R^{p'}
        zs_next, Rs_next = self.zs_R_next(z_t)                           # (2M, n_risky), (2M, n_assets)
        R_p_next = Rs_next @ omega                                       # (2M,)

        # m' branches
        m_emp, m_unemp = self.m_next(rs_t, R_p_next)                     # (2M,), (2M,)

        # Build point arrays for interpolation
        #pts_emp   = np.column_stack([m_emp,  zs_next])                   # (2M, 1+n_risky)
        #pts_unemp = np.column_stack([m_unemp, zs_next])
        pts_emp   = self._clamp_pts(a_policy, np.column_stack([m_emp,  zs_next]))
        pts_unemp = self._clamp_pts(a_policy, np.column_stack([m_unemp, zs_next]))


        # Next-period consumption via current policy a'(m',z'), then c' = a' * m'
        a_next_emp   = np.clip(a_policy(pts_emp),   0.0, 1.0)
        a_next_unemp = np.clip(a_policy(pts_unemp), 0.0, 1.0)
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
    
    def smooth_fisher_burmeister(self, a, b, mu=1e-6):
        """Smoothed Fisher-Burmeister function for complementarity conditions."""
        return np.sqrt(a**2 + b**2 + 2.0*mu) - (a + b)
    
    def euler_residuals(self, rs, z, c, omega, a_policy, c_upper, xi: float | None = None, mu_fb: float = 1e-6) -> np.ndarray:
        """Residuals: [Euler, portfolio equalization on active face]."""

        g = self.g_vector(rs, z, omega, a_policy)       # (n,)
        gbar = float(np.dot(omega, g))

        if xi is None:
            xi = gbar

        fb_r_port = self.smooth_fisher_burmeister(omega, xi - g, mu=mu_fb)

        simplex_r = np.sum(omega) - 1.0
        # Euler
        r0 = self.u_prime(c) - self.beta * gbar
        r0 = self.smooth_fisher_burmeister(r0, c_upper-c, mu=0.0)
        return np.concatenate([[r0], [simplex_r], fb_r_port])
    
    # ---------------------------
    # Policy evaluation & errors
    # ---------------------------

    def get_euler_errors(self, m_grid, z_grids, a_policy, omega_policy, episodes=100):
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
            w = omega_policy([[m, *z_vec]])[0]
            w = np.clip(w, 0.0, None)
            s = w.sum()
            return w / s if s > 0 else np.full(self.n_assets, 1.0 / self.n_assets)

        # unpack params
        kappa = np.atleast_1d(self.kappa_assets)
        xi    = np.atleast_1d(self.xi_assets)
        rho   = np.atleast_1d(self.rho_assets)
        sig_z = np.atleast_1d(self.sigma_perm_assets)   # persistent predictor innovation
        sig_R = np.atleast_1d(self.sigma_tran_assets)   # transitory return shock

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
                theta = (np.exp(self.mu_z + self.sigma_tran * self.rng.standard_normal())
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
                R_vec = self.euler_residuals(rs=rs, z=z_t, c=c_t, omega=w_t,
                                            a_policy=a_policy, c_upper=c_upper, mu_fb=0.0)
                euler_errors[i, t] = np.log10(max(1e-16, float(np.max(np.abs(R_vec)))))

            states_m[i, :] = m_path
            states_z[i, :, :] = z_path
            avg_return += U / n_nodes

        states = {'m': states_m, 'z': states_z}
        return euler_errors.ravel(), states, avg_return


#-------------------------------------------------------------------
# Solvers: standardized on share policy a(m)
#-------------------------------------------------------------------

class PlanningSolvers:
    def __init__(
        self,
        m_points: int = 61,
        z_points: int = 21,
        tol: float = 1e-5,
        conv: float = 1e-5,
        min_share: float = 1e-6,
    ):
        self.m_points = m_points
        self.z_points = z_points
        self.tol = float(tol)
        self.conv = float(conv)
        self.min_share = float(min_share)
    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------   
    
    # VFI helpers
    
    def _init_policy_pd(self, model: PlanningProblem, warm_start: bool = False):
        """Initialize grids, consumption share array and portfolio logits.

        c_share_new stores α(m,z)=c/m (consumption share, not assets!).  
        w_new stores per-asset weights for ω(a,z), where a is the post-decision state.          
        """
        m_grid = model.make_m_grid(points=self.m_points, span=(0.2, 15.))
        rs_grid = model.make_m_grid(points=self.m_points//2, span=(0.0, 15.))   # post-decision grid is on a TODO: use looser grid for post-decision states?
        z_grids = model.make_z_grids(points=self.z_points)
        m_grid_shape = (len(m_grid),) + tuple(len(g) for g in z_grids)
        rs_grid_shape = (len(rs_grid),) + tuple(len(g) for g in z_grids)

        w_new = np.zeros(rs_grid_shape + (model.n_assets,), dtype=float)

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
                a_new = a_bs_policy(m_grid)
                a_new = np.tile(a_new[:, None], (1,) + tuple(len(g) for g in z_grids))
            except Exception as e:
                logging.warning(f"Warm start for portfolio choice failed with exception {e}; proceeding with naive initialization.")
                warm_start = False
        else: 
            a_new = np.full(m_grid_shape, 0.2, dtype=float) # TODO: Warm start by buffer-stock saving policy?
        a_policy = model.make_multilinear(m_grid, z_grids, a_new)

        return m_grid, rs_grid, z_grids, a_new, a_policy, w_new
    
    def _init_pd_value(
            self, 
            model: PlanningProblem, 
            rs_grid, 
            z_grids, 
            a_policy: None | RegularGridInterpolator = None, 
            omega_grid: None | np.ndarray = None, 
            warm_start: bool = False):
        """Initialize post-decision value function on (a,z) grid."""
        grid_shape = (len(rs_grid), ) + tuple(len(g) for g in z_grids)
        v_post_grid = np.zeros(grid_shape, dtype=float) # TODO: Warm start by buffer-stock saving value function?
        v_post = model.make_multilinear(rs_grid, z_grids, v_post_grid)
        if warm_start:
            assert a_policy is not None, "c_share_policy must be provided for warm start."
            assert omega_grid is not None, "omega_grid must be provided for warm start."
            omega_grid = omega_grid + 0.5 # TODO: Quick and dirty fix...
            v_post, v_post_grid = self.v_post_eval(model, rs_grid, z_grids, v_post, a_policy, omega_grid)
        return v_post, v_post_grid
    
    def v_post_eval(self, model, rs_grid, z_grids, v_post, a_policy, omega_grid, max_iter=50, conv=5e-4):
        """Iterate over (a,z) grid to evaluate post-decision value function until convergence."""
        grid_shape = (len(rs_grid), ) + tuple(len(g) for g in z_grids)
        z_indices = [range(len(g)) for g in z_grids]
        z_tuples = list(itertools.product(*z_indices))

        v_post_new = np.zeros(grid_shape, dtype=float)

        for _ in range(1, max_iter + 1):
            v_post_old = v_post_new.copy()

            for ia, a in enumerate(rs_grid):
                for z_idx_tuple in z_tuples:
                    z_vec = np.array([z_grids[j][idx] for j, idx in enumerate(z_idx_tuple)], dtype=float)
                    omega = omega_grid[(ia, *z_idx_tuple, slice(None))]
                    v_post_new[(ia, *z_idx_tuple)] = model.pd_bellman(omega, a, z_vec, a_policy, v_post)

            max_diff = np.max(np.abs(v_post_new - v_post_old))
            v_post = model.make_multilinear(rs_grid, z_grids, v_post_new)
            if max_diff < conv:
                break

        return v_post, v_post_new
    
    def _check_convergence_vfi(
        self,
        a_old,
        a_new,
        omega_old,
        omega_new,
        v_post_old,
        v_post_new
    ) -> Tuple[bool, float, float, float]:
        """
        Sup-norm check for both consumption share α, portfolio weights ω and post-decision value functions.
        """
        max_da = float(np.max(np.abs(a_old - a_new)))
        max_dw = float(np.max(np.abs(omega_old - omega_new)))
        max_dV = float(np.max(np.abs(v_post_old - v_post_new)))
        return max_dV < self.conv, max_da, max_dw, max_dV
    
    def weights_to_mz_policy(self, model, rs_grid, m_grid, z_grids, a_policy, omegas):
        """Map weights on post_decision grid to weights on pre-decision grid (m,z) for given consumption and asset allocation policy and return as multilinear interpolator"""
        omega_policy = model.make_multilinear(rs_grid, z_grids, omegas)
        m_grid_shape = (len(m_grid),) + tuple(len(g) for g in z_grids)
        omega_mz = np.zeros(m_grid_shape + (model.n_assets,), dtype=float)
        z_indices = [range(len(g)) for g in z_grids]
        z_tuples = list(itertools.product(*z_indices))
        for im, m in enumerate(m_grid):
             for z_idx_tuple in z_tuples:
                z_vec = np.array([z_grids[j][idx] for j, idx in enumerate(z_idx_tuple)], dtype=float)
                a = a_policy([[m, *z_vec]])[0]
                rs = m * (1.0 - a)
                omega_mz[(im, *z_idx_tuple, slice(None))] = omega_policy([[rs, *z_vec]])[0]
        
        omega_policy_mz = model.make_multilinear(m_grid, z_grids, omega_mz)
        return omega_policy_mz

    def logits_to_weights(self, logits):
        # Additive log-ratio: logits in R^{N-1} → ω ∈ Δ^N  (ALR inverse = softmax of [logits, 0])
        e = np.exp(np.clip(logits, -50, 50))
        denom = 1.0 + np.sum(e)
        return np.concatenate([e / denom, [1.0 / denom]])
    
    def weights_to_logits(self, omega):
        # ω ∈ Δ^N  → logits in R^{N-1}  (ALR forward = log(ω_1..N-1 / ω_N))
        omega = np.asarray(omega, dtype=float)
        assert omega.ndim == 1 and omega.shape[0] >= 2, "omega must be 1D with at least 2 elements."
        omega = np.clip(omega, 1e-15, None) 
        omega /= omega.sum()
        logits = np.log(omega[:-1]) - np.log(omega[-1])
        return logits

    # ------------------------------------------------------------------
    # Value Function Iteration (VFI)
    # ------------------------------------------------------------------

    def VFI(self,
            model: PlanningProblem,
            verbose: bool = False,
            max_iter: int = 1_000
            ) -> Tuple[RegularGridInterpolator, RegularGridInterpolator]:
            """Compute optimal consumption share and portfolio choice policy by TI."""
            # --- Setup ----------------------------------------------------
            m_grid, rs_grid, z_grids, a_new, a_policy, w_new = self._init_policy_pd(model, warm_start=True)
            v_post, v_post_new = self._init_pd_value(model, rs_grid, z_grids, a_policy, w_new, warm_start=True)
            z_indices = [range(len(g)) for g in z_grids]
            z_tuples = list(itertools.product(*z_indices))

            vfe_conv = 5e-4

            # --- Iteration ------------------------------------------------
            for it in range(1, max_iter + 1):
                a_old = a_new.copy()
                w_old = w_new.copy()
                v_post_old = v_post_new.copy()

                # --- (1) Weight improvement by value maximization on each (a,z)
                start = time.time()
                for irs, rs in enumerate(rs_grid):
                    for z_idx_tuple in z_tuples:
                        z_vec = np.array([z_grids[j][idx] for j, idx in enumerate(z_idx_tuple)], dtype=float)
                        logits = self.weights_to_logits(w_new[(irs, *z_idx_tuple, slice(None))])

                        def optimize_weights(v):
                            omega = self.logits_to_weights(v)
                            return -model.pd_bellman(omega, rs, z_vec, a_policy, v_post, min_share=self.min_share)

                        res = minimize(optimize_weights, logits, method="L-BFGS-B", 
                                       options=dict(maxiter=60, ftol=1e-4)) #xtol=1e-4, , direc=None
                        w_star = self.logits_to_weights(res.x) if res.success else w_new[(irs, *z_idx_tuple, slice(None))]
                        w_new[(irs, *z_idx_tuple, slice(None))] = w_star
                print(f"Weight update time: {time.time() - start:.2f} seconds")

                # --- Policy evaluation for post-decision V with new w
                v_post, v_post_new = self.v_post_eval(model, rs_grid, z_grids, v_post, a_policy, w_new, conv=vfe_conv)

                # --- (2) Outer-c update
                start = time.time()
                for im, m in enumerate(m_grid):
                    for z_idx_tuple in z_tuples:
                        z_vec = np.array([z_grids[j][idx] for j, idx in enumerate(z_idx_tuple)], dtype=float)
                        
                        # bounds; tiny margin to avoid degenerate corners numerically
                        c_lo = self.min_share * float(m)
                        c_hi = (1.0 - self.min_share) * float(m)
                        if c_hi <= c_lo + 1e-15:
                            a_new[(im, *z_idx_tuple)] = 1.0  # consume all if m≈0
                            continue

                        def optimize_cons(c: float):
                            c = float(c)
                            return -model.bellman(c, m, z_vec, v_post)

                        res = minimize_scalar(optimize_cons, bounds=(c_lo, c_hi), method='bounded',
                                            options=dict(xatol=self.tol, maxiter=max_iter))
                        c_star = float(res.x if res.success else (0.5*(c_lo+c_hi)))
                        a_new[(im, *z_idx_tuple)] = np.clip(c_star / m, 0.0, 1.0)
                a_policy = model.make_multilinear(m_grid, z_grids, a_new)
                print(f"Consumption update time: {time.time() - start:.2f} seconds")

                # --- (3) Policy evaluation for W with fixed (a, w)
                start = time.time()
                v_post, v_post_new = self.v_post_eval(model, rs_grid, z_grids, v_post, a_policy, w_new, conv=vfe_conv)
                print(f"Value function evaluation time: {time.time() - start:.2f} seconds")

                # --- Convergence check ---------------------------------------
                converged, max_diff_a_share, max_diff_w, max_diff_v_post = self._check_convergence_vfi(a_old, a_new, w_old, w_new, v_post_old, v_post_new)
                if max_diff_v_post < 1e-3:
                    vfe_conv = 1e-5  # tighten VFE convergence tolerance

                if verbose:
                    print(
                        f"[VFI iter {it}] max|Δa|={max_diff_a_share:.3e}, "
                        f"max|Δω|={max_diff_w:.3e}, max|ΔV|={max_diff_v_post:.3e}, converged={converged}"
                    )

                if converged:
                    break

            w_policy = self.weights_to_mz_policy(model, rs_grid, m_grid, z_grids, a_policy, w_new)

            return a_policy, w_policy


    # ------------------------------------------------------------------
    # Time iteration (TI)
    # ------------------------------------------------------------------

    # TI helpers

    def _init_policy(self, model: PlanningProblem, warm_start: bool = False):
        """Initialize grids, consumption share array and portfolio logits.

        a_new stores α(m,z)=c/m (consumption share, not assets!).  
        w_logits_new stores per-asset logits for ω(m,z).          
        """
        m_grid = model.make_m_grid(points=self.m_points)
        z_grids = model.make_z_grids(points=self.z_points)
        grid_shape = (len(m_grid),) + tuple(len(g) for g in z_grids)

        w_new = np.zeros(grid_shape + (model.n_assets,), dtype=float)
        w_policy = model.make_multilinear(m_grid, z_grids, w_new)

        if warm_start:
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
            a_new = a_bs_policy(m_grid)
            a_new = np.tile(a_new[:, None], (1,) + tuple(len(g) for g in z_grids))
        else: 
            a_new = np.full(grid_shape, 0.2, dtype=float)
        a_policy = model.make_multilinear(m_grid, z_grids, a_new)

        return m_grid, z_grids, a_new, a_policy, w_new, w_policy
    
    def _policy_eval_on_grid(self, policy: RegularGridInterpolator, grids: Tuple[np.ndarray, ...], out_shape):
        """Evaluate scalar policy on full tensor grid."""
        coords = np.meshgrid(*grids, indexing='ij')
        pts = np.column_stack([c.ravel() for c in coords])
        vals = policy(pts).reshape(out_shape)
        return vals
    
    def _project_simplex(self, v: np.ndarray) -> np.ndarray:
        """Project v to {w >= 0, sum w = 1} (Euclidean)."""
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            v = v[None, :]
        u = np.sort(v, axis=1)[:, ::-1]
        cssv = np.cumsum(u, axis=1) - 1
        ind = np.arange(1, v.shape[1] + 1)
        rho = np.sum(u > (cssv / ind), axis=1) - 1
        theta = (cssv[np.arange(v.shape[0]), rho] / (rho + 1))
        w = np.maximum(v - theta[:, None], 0.0)
        s = np.sum(w, axis=1, keepdims=True)
        w = w / np.maximum(s, 1e-16)
        return w[0] if w.shape[0] == 1 else w

    def _check_convergence(
        self,
        a_old,
        a_policy,
        w_old,
        w_policy,
        m_grid,
        z_grids,
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Sup-norm check for both consumption share α and portfolio logits ω.
        """
        grids = (m_grid, *z_grids)
        grid_shape = tuple(len(g) for g in grids)  # (len(m), len(z1), ..., len(zN))

        a_new_eval = self._policy_eval_on_grid(a_policy, grids, grid_shape)
        w_new_eval = self._policy_eval_on_grid(
            w_policy, grids, grid_shape + (w_old.shape[-1],)
        )

        max_diff_a = np.max(np.abs(a_old - a_new_eval))
        max_diff_w = float(np.max(np.abs(w_old.reshape(-1, w_old.shape[-1]) -
                            w_new_eval.reshape(-1, w_old.shape[-1]))))
        return (max_diff_a < self.conv and max_diff_w < self.conv), max_diff_a, max_diff_w

    def _solve_node(self,
                    model: PlanningProblem,
                    m: float,
                    z_vec: np.ndarray,
                    a0: float,
                    w0: np.ndarray,                                   
                    a_policy: RegularGridInterpolator,
                    mu_fb: float = 1e-4) -> Tuple[float, np.ndarray]:
        """
        Solve for (a, ω) at a node by root-finding on:
          [ Euler(c) = 0, sum(ω)-1 = 0, φ_μ(ω_i, gbar - g_i) = 0 ].
        Returns (a*, w*).
        """
        # ensure feasible warm start
        w0 = self._project_simplex(w0)
        rs0 = (1.0 - a0) * m
        g0 = model.g_vector(rs_t=rs0, z_t=z_vec, omega=w0, a_policy=a_policy)     
        xi0 = float(np.dot(w0, g0))                     
        x0 = np.concatenate([[a0], w0, [xi0]])

        # --- early exit when savings are (numerically) negligible -------
        #     portfolio is irrelevant if rs ≈ 0 ⇒ skip the ω,ξ solve                                                         
        if a0 >= (1.0 - self.min_share):    
            a0 = float(np.clip(a0, self.min_share, 1.0 - self.min_share))      
            rs0 = (1.0 - a0) * m    
            c0 = a0 * m                                
            # Decide consumption corner by the sign of F_a = u'(c) - β ḡ         
            g = model.g_vector(rs_t=rs0, z_t=z_vec, omega=w0, a_policy=a_policy)  
            gbar = float(np.dot(w0, g))                                          
            F_a = model.u_prime(c0) - model.beta * gbar                          
            a_star = (1.0 - self.min_share) if (F_a > 0) else self.min_share     
            return a_star, w0   

        def residuals(x):
            a_share = float(np.clip(x[0], self.min_share, 1.0 - self.min_share))

            # use a smooth interior map only for simulating g and c' (no sharp projection)
            # option S1: softplus + renorm (allows near-zero, differentiable)
            eps = 1e-12
            w_pos = np.maximum(x[1:-1], 0.) + np.log1p(np.exp(-np.abs(x[1:-1])))          # softplus
            omega = w_pos / w_pos.sum()          # sum-to-one
            xi = float(x[-1])   
            rs = (1.0 - a_share) * m             
            c_now = a_share * m
            c_upper = (1.0 - self.min_share) * m
            F = model.euler_residuals(rs, z_vec, c_now, omega, a_policy, c_upper, xi, mu_fb=mu_fb) 
            # Replace the simplex residual inside F with the raw one so it's not identically 0.
            F[1] = np.sum(x[1:-1]) - 1.0
            return F

        sol = root(residuals, x0, method='hybr', tol=self.tol)

        if not sol.success or not np.all(np.isfinite(sol.x)):
            a_star = float(np.clip(sol.x[0], self.min_share, 1.0 - self.min_share))
            #print((m, z_vec))
            #print(np.min(omega_test))
            print((m, z_vec, a_star))
            w_star = w0
        else:
            a_star = float(np.clip(sol.x[0], self.min_share, 1.0 - self.min_share))
            w_star = self._project_simplex(sol.x[1:-1])     
            #print((m, z_vec))      

        return a_star, w_star   

    def TI(self, 
           model: PlanningProblem, 
           verbose: bool = False,
           max_iter: int = 1_000
           ) -> Tuple[RegularGridInterpolator, RegularGridInterpolator]:
            """Compute optimal consumption share and portfolio choice policy by TI."""
            # --- Setup ----------------------------------------------------
            m_grid, z_grids, a_new, a_policy, w_new, w_policy = self._init_policy(model, warm_start=True)
            z_indices = [range(len(g)) for g in z_grids]

            # --- Iteration ------------------------------------------------
            for it in range(1, max_iter + 1):
                a_old = a_new.copy()
                w_old = w_new.copy()

                # Pointwise policy improvement: (c, ω) via local active-set solve
                for im, m in enumerate(m_grid):
                    for z_idx_tuple in itertools.product(*z_indices):
                        z_vec = np.array([z_grids[j][idx] for j, idx in enumerate(z_idx_tuple)], dtype=float)

                        # Warm starts from current policy interpolants
                        #pt = np.array([[m, *z_vec]], dtype=float)
                        #a0 = float(np.clip(a_policy(pt)[0], self.min_share, 1.0 - self.min_share))
                        #w0 = w_policy(pt)[0]  # logits vector (n_assets,)
                        a0 = a_new[(im, *z_idx_tuple)]
                        w0 = w_new[(im, *z_idx_tuple, slice(None))]

                        a_star, w_star = self._solve_node(model, m, z_vec, a0, w0, a_policy)

                        # write back
                        a_new[(im, *z_idx_tuple)] = a_star 
                        w_new[(im, *z_idx_tuple, slice(None))] = w_star

                # Update interpolants
                a_policy = model.make_multilinear(m_grid, z_grids, a_new)
                w_policy = model.make_multilinear(m_grid, z_grids, w_new)

                # --- Convergence check ---------------------------------------
                converged, max_diff_a, max_diff_w = self._check_convergence(a_old, a_policy, w_old, w_policy, m_grid, z_grids)

                if verbose:
                    print(
                        f"[TI iter {it}] max|Δa|={max_diff_a:.3e}, "
                        f"max|Δω|={max_diff_w:.3e}, converged={converged}"
                    )

                if converged:
                    break

            return a_policy, w_policy 
