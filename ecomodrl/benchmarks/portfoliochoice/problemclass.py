import numpy as np
import itertools
from typing import Union, Tuple, List
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root
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
        self.gamma, self.beta = float(gamma), float(beta)

        # Normalized income process
        self.R, self.G = 1.0 + r, 1.0 + g
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
            Number of Sobol' points *before* antithetics (final count is 2*M).
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
        U = eng.random(M)                                   # (M, d) in (0,1)
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
        Map cached RQMC normal nodes to next-period zs and returns Rprime.

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
    
    # ---------------------------
    # g_i expectations for portfolio FOCs
    # ---------------------------
    def g_vector(self, c_t: float, omega: np.ndarray, m_t: float, z_t: np.ndarray,
                 a_policy: RegularGridInterpolator) -> np.ndarray:
        """
        Compute g_i = E[ u'(c') / Γ'^γ * R_i' ]  (i=0..N), using current policies for c'(m', z').
        """
        # savings
        rs_t = m_t - c_t

        # Shocks -> (z', R') and R^{p'}
        zs_next, Rs_next = self.zs_R_next(z_t)                           # (2M, n_risky), (2M, n_assets)
        R_p_next = Rs_next @ omega                                       # (2M,)

        # m' branches
        m_emp, m_unemp = self.m_next(rs_t, R_p_next)                     # (2M,), (2M,)

        # Build point arrays for interpolation
        pts_emp   = np.column_stack([m_emp,  zs_next])                   # (2M, 1+n_risky)
        pts_unemp = np.column_stack([m_unemp, zs_next])

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
    
    def euler_residuals(self, c, omega, m, z, a_policy, idx_active):
        """Residuals: [Euler, portfolio equalization on active face]."""
        g = self.g_vector(c, omega, m, z, a_policy)       # (n,)
        gbar = float(np.dot(omega, g))
        # Euler
        r0 = self.u_prime(c) - self.beta * gbar
        # Portfolio equalization on the face (A-1 independent equations)
        if idx_active.size == 1:
            r_port = np.zeros(0, dtype=float)
        else:
            ga = g[idx_active]
            r_port = ga[:-1] - ga[-1]
        return np.concatenate([[r0], r_port])


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
        max_drop_iters: int = 5,
    ):
        self.m_points = m_points
        self.z_points = z_points
        self.tol = float(tol)
        self.conv = float(conv)
        self.min_share = float(min_share)
        self.max_drop_iters = int(max_drop_iters)
    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _init_policy(self, model: PlanningProblem):
        """Initialize grids, consumption share array and portfolio logits.

        a_new stores α(m,z)=c/m (consumption share, not assets!).  
        w_logits_new stores per-asset logits for ω(m,z).          
        """
        m_grid = model.make_m_grid(points=self.m_points)
        z_grids = model.make_z_grids(points=self.z_points)
        grid_shape = (len(m_grid),) + tuple(len(g) for g in z_grids)

        a_new = np.full(grid_shape, 0.2, dtype=float)
        a_policy = model.make_multilinear(m_grid, z_grids, a_new)

        w_logits_new = np.zeros(grid_shape + (model.n_assets,), dtype=float)
        w_policy = model.make_multilinear(m_grid, z_grids, w_logits_new)

        return m_grid, z_grids, a_new, a_policy, w_logits_new, w_policy
    
    def _policy_eval_on_grid(self, policy: RegularGridInterpolator, grids: Tuple[np.ndarray, ...], out_shape):
        """Evaluate scalar policy on full tensor grid."""
        coords = np.meshgrid(*grids, indexing='ij')
        pts = np.column_stack([c.ravel() for c in coords])
        vals = policy(pts).reshape(out_shape)
        return vals

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
        max_diff_w = np.max(np.abs(w_old - w_new_eval)) 
        return (max_diff_a < self.conv and max_diff_w < self.conv), max_diff_a, max_diff_w
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax for a 1D vector"""
        x = np.asarray(x, dtype=float)
        x = x - np.max(x)             
        ex = np.exp(x)
        s = np.sum(ex)
        return ex / s if s > 0 else np.full_like(ex, 1.0 / len(ex))
    
    @staticmethod
    def _sigmoid(x):  
        """stable mapping to (0,1)"""
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def _pack_phi_from_logits(logits, idx_active):
        """Convert full logits to active-face (A-1) anchored logits."""
        A = idx_active.size
        if A == 1:
            return np.zeros(0, dtype=float)
        l = np.asarray(logits, float)[idx_active]
        l[~np.isfinite(l)] = -20.0
        l = l - l[-1]              # anchor last active to 0
        return l[:-1]
    
    def _pack_phi_from_weights(self, w, idx_active):
        """Convert full weights (possibly with zeros) to (A-1) anchored logits."""
        A = idx_active.size
        if A == 1:
            return np.zeros(0, dtype=float)
        ww = np.clip(np.asarray(w, float)[idx_active], self.min_share * 0.1, 1.0)
        ww /= ww.sum()
        l = np.log(ww)
        l = l - l[-1]
        return l[:-1]
    
    def _weights_from_logits(self, logits):
        return self._softmax(np.asarray(logits, float))
    
    def _build_vars(self, chi, phi, idx_active, m, n):
        """
        Map (chi, phi) -> (c, omega_full) on current active set.

        chi -> consumption c in (0,m); we clip away from exact 0 or m to avoid
        u'(0)=∞ and zero savings instability.
        """
        # c in [tiny, m - tiny]
        cap = max(m, 1e-14)
        c_raw = self._sigmoid(chi) * cap
        c = np.clip(c_raw, 1e-12, cap - 1e-12)

        # ω on active set via softmax
        omega = np.zeros(n, dtype=float)
        if idx_active.size == 1:
            omega[idx_active[0]] = 1.0
        else:
            theta = np.concatenate([phi, [0.0]])   # last anchored
            omega[idx_active] = self._softmax(theta)
        return c, omega
    
    def _solve_node(self, model: PlanningProblem, m: float, z: np.ndarray,
                    a0: float, w_logits0: np.ndarray,
                    a_policy: RegularGridInterpolator):
        """
        Per-node solve with robust active-set:
        - Unknowns: chi (bounds c) + phi (A-1 logits on the active set).
        - KKT on active set: g_i equalized; Euler pins the common level via ḡ.
        - Drop tiny weights; re-activate any inactive asset that violates KKT.
        """
        n = model.n_assets
        a0 = float(np.clip(a0, self.min_share, 1.0 - self.min_share))
        chi0 = np.log(a0) - np.log(1.0 - a0)  # logistic param for a=c/m (consumption share)

        def residuals(x, idx_active):
            chi = x[0]; phi = x[1:]
            c, omega = self._build_vars(chi, phi, idx_active, m, n)
            return model.euler_residuals(c, omega, m, z, a_policy, idx_active)

        # --- initialization of active set ------------------------------
        w0_full = self._weights_from_logits(np.asarray(w_logits0, float))
        if np.all(w0_full < self.min_share):
            # Degenerate warm start: activate the largest-weight asset
            j0 = int(np.argmax(w0_full))
            active = np.array([j0], dtype=int)
        else:
            active = np.array(
                np.flatnonzero(w0_full >= self.min_share), dtype=int
            )
            if active.size == 0:
                active = np.array([int(np.argmax(w0_full))], dtype=int)

        phi0 = self._pack_phi_from_logits(w_logits0, active)
        kkt_tol = max(self.tol, 1e-10)

        # --- active-set loop: drop & (re)activate ----------------------
        for _ in range(self.max_drop_iters):
            # Solve on current face
            x0 = np.concatenate([[chi0], phi0])
            sol = root(
                lambda x: residuals(x, active), 
                x0, 
                method='hybr', 
                tol=self.tol,
            )
            x = sol.x if sol.success else x0  # robust fallback

            chi_hat = float(x[0]); phi_hat = x[1:]
            c_hat, omega_hat = self._build_vars(chi_hat, phi_hat, active, m, n)

            # ----- 1) Drop tiny active weights -------------------------
            tiny_mask = (omega_hat[active] < self.min_share)
            if np.any(tiny_mask):
                keep = active[~tiny_mask]
                if keep.size == 0:
                    # keep the largest among current actives to avoid empty face
                    keep = np.array([active[int(np.argmax(omega_hat[active]))]], dtype=int)
                # Repack φ on the smaller face; keep χ warm start
                phi0 = self._pack_phi_from_weights(omega_hat, keep)
                active = keep
                chi0 = chi_hat
                continue  # re-solve on reduced face

            # ----- 2) KKT re-activation check --------------------------
            # If some inactive asset i has g_i > gbar + tol, add the worst violator.
            g_all = model.g_vector(c_hat, omega_hat, m, z, a_policy)
            gbar_hat = float(np.dot(omega_hat, g_all))
            inactive = np.setdiff1d(np.arange(n), active, assume_unique=True)
            if inactive.size > 0:
                violations = g_all[inactive] - gbar_hat
                j_viol = np.argmax(violations)
                if violations[j_viol] > kkt_tol:
                    # re-activate this violator
                    i_add = inactive[j_viol]
                    new_active = np.concatenate([active, [i_add]])
                    phi0 = self._pack_phi_from_weights(omega_hat, new_active)
                    active = new_active
                    chi0 = chi_hat
                    continue  # re-solve on expanded face

            # No drops and no KKT violations -> converged on this node
            a_star = float(np.clip(c_hat / m, self.min_share, 1.0 - self.min_share))
            # store logits: active assets use log(ω), inactives set to large negative
            w_logits = np.full(n, -20.0, dtype=float)
            w_act = np.clip(omega_hat[active], self.min_share * 0.1, 1.0)
            w_act /= w_act.sum()
            w_logits[active] = np.log(w_act)
            return a_star, w_logits

         # --- final fallback (very rare): return warm start safely -----------
        a_fb = float(
            np.clip(a0, self.min_share, 1.0 - self.min_share)
        )
        w_fb = np.full(n, -20.0, dtype=float)
        w_fb[int(np.argmax(w0_full))] = 0.0  # log(1)=0 for the chosen asset
        return a_fb, w_fb 


    # ------------------------------------------------------------------
    # Time iteration (TI)
    # ------------------------------------------------------------------

    def TI(self, 
           model: PlanningProblem, 
           verbose: bool = False,
           max_iter: int = 1_000
           ) -> Tuple[RegularGridInterpolator, RegularGridInterpolator]:
            """Compute optimal consumption share and portfolio choice policy by TI."""
            # --- Setup ----------------------------------------------------
            m_grid, z_grids, a_new, a_policy, w_logits_new, w_policy = self._init_policy(model)
            z_indices = [range(len(g)) for g in z_grids]

            # --- Iteration ------------------------------------------------
            for it in range(1, max_iter + 1):
                a_old = a_new.copy()
                w_old = w_logits_new.copy()

                # Pointwise policy improvement: (c, ω) via local active-set solve
                for im, m in enumerate(m_grid):
                    for z_idx_tuple in itertools.product(*z_indices):
                        z_vec = np.array([z_grids[j][idx] for j, idx in enumerate(z_idx_tuple)], dtype=float)

                        # Warm starts from current policy interpolants
                        pt = np.array([[m, *z_vec]], dtype=float)
                        a0 = float(np.clip(a_policy(pt)[0], self.min_share, 1.0 - self.min_share))
                        w0 = w_policy(pt)[0]  # logits vector (n_assets,)

                        a_star, w_logits_star = self._solve_node(model, m, z_vec, a0, w0, a_policy)

                        # write back
                        a_new[(im, *z_idx_tuple)] = a_star
                        w_logits_new[(im, *z_idx_tuple, slice(None))] = w_logits_star

                # Update interpolants
                a_policy = model.make_multilinear(m_grid, z_grids, a_new)
                w_policy = model.make_multilinear(m_grid, z_grids, w_logits_new)

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



 # ---------------------------
    # Euler helpers
    # ---------------------------

    # def demu_next(self, c_next_employed: np.ndarray, c_next_unemployed: np.ndarray, R: np.ndarray) -> float:
    #     # β * E[ R / ((G*Ψ_{t+1})^γ) * u'(c_{t+1}) ]
    #     factor = (R / self.SDF_denom)
    #     employed = np.mean(factor * self.u_prime(c_next_employed))
    #     unemployed = np.mean(factor * self.u_prime(c_next_unemployed))
    #     return self.beta * ((1.0 - self.unemp_prob) * employed + self.unemp_prob * unemployed)

    # def euler_residual(self, a: np.ndarray, m_t: float, z_t: Union[float, np.ndarray], 
    #                    policy: RegularGridInterpolator) -> float:
    #     c_t  = self.consumption_from_share(a[0], m_t)
    #     rs_t = m_t - c_t

    #     zs_next, Rs_next = self.zs_R_next(z_t)                                      # (2M, n), (2M, n)
    #     R_eff = Rs_next @ a[1:]                                                     # (2M,)

    #     m_next_empl, m_next_unempl = self.m_next(rs_t, R_eff)                       # (2M,), (2M,)
    #     pts_empl    = np.column_stack([m_next_empl, zs_next])                       # (2M, 1+n)
    #     pts_unempl    = np.column_stack([m_next_unempl, zs_next])                   # (2M, 1+n)
    #     a_next_empl = policy(pts_empl)                                              # (2M,)
    #     a_next_unempl = policy(pts_unempl)                                          # (2M,)

    #     c_next_empl = self.consumption_from_share(a_next_empl, m_next_empl)         # (2M,)
    #     c_next_unempl = self.consumption_from_share(a_next_unempl, m_next_unempl)   # (2M,)
    #     rhs    = self.demu_next(c_next_empl, c_next_unempl, R_eff)                  # scalar
    #     return self.u_prime(c_t) - rhs
    
    # def c_from_euler(self, c_next: np.ndarray) -> float:
    #     """Recover c_t from: u'(c_t) = (β R / G) E[ u'(c_{t+1}) / Ψ_{t+1} ]."""
    #     mu = self.demu_next(c_next)
    #     return self.u_prime_inv(mu)

    # ---------------------------
    # Policy evaluation & errors
    # ---------------------------
    # def get_euler_errors(self, grid, policy, episodes=100):
    #     """
    #     Simulate Euler residuals along sample paths starting from each m on the grid,
    #     using the provided consumption-share policy a(·). Returns (errors, states, avg_return).
    #     """
    #     T = int(episodes)
    #     grid = np.asarray(grid, dtype=float)
    #     grid_len = grid.size

    #     states = np.empty((grid_len, T + 1), dtype=float)
    #     euler_errors = np.empty((grid_len, T), dtype=float)
    #     avg_return = 0.0

    #     for i, m0 in enumerate(grid):
    #         m_path = np.empty(T + 1, dtype=float)
    #         c_path = np.empty(T, dtype=float)
    #         m_path[0] = m0

    #         U = 0.0
    #         power = 1.0

    #         for t in range(T):
    #             a_t_share = float(policy(m_path[t]))  # consumption share
    #             c_t = self.consumption_from_share(a_t_share, m_path[t])

    #             # draw shocks
    #             if np.random.uniform(0.0, 1.0) < self.unemp_prob:
    #                 theta_draw = self.theta_low
    #             else:
    #                 theta_draw = np.exp(np.random.normal(self.mu_z, self.sigma_tran))
    #             psi_draw = np.exp(np.random.normal(-0.5 * self.sigma_perm**2, self.sigma_perm))

    #             m_path[t + 1] = self.R / (self.G * psi_draw) * (m_path[t] - c_t) + theta_draw

    #             c_path[t] = c_t
    #             U += power * self.u(c_t)
    #             power *= self.beta

    #         # Euler error along the realized path
    #         err = np.empty(T, dtype=float)
    #         for t, (c_t, m_t) in enumerate(zip(c_path, m_path[:-1])):
    #             a_t_share = c_t / m_t  # implied share
    #             err[t] = np.log10(abs(self.euler_residual(a_t_share, m_t, policy)))
            
    #         euler_errors[i, :] = err
    #         states[i, :] = m_path
    #         avg_return += U / grid_len

    #     return euler_errors.ravel(), states.ravel(), avg_return