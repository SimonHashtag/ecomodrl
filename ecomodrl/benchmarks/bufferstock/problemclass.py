"""
Buffer-stock saving model with permanent-income normalization.

Notation (all variables normalized by permanent income P_t):
  - m_t : cash-on-hand (resources at start of period t)
  - rs_t := (1 - a_t(m_t)) * m_t = m_t - c_t : resources saved / carried to next period before shocks
  - a(m) ∈ [0,1] : consumption share policy; consumption is a fraction of m
  - c_t : consumption level = a_t(m_t) * m_t

Law of motion for next period's cash-on-hand (given transitory Θ and permanent Ψ shocks):
  - m_{t+1} = (R / (G Ψ_{t+1})) * (m_t - c_t) + Θ_{t+1}
  -         = (R / (G Ψ_{t+1})) * rs_t + Θ_{t+1}

Euler equation under CRRA and normalization:
  - u'(c_t) = (β R / G^γ) * E[ u'(c_{t+1}) * Ψ_{t+1}^{-γ} ]

EGM step (working on a grid of resources saved rs):
  1) m^{next}(rs; Θ,Ψ) = (R / (G Ψ)) * rs + Θ
  2) c^{next} = a_t(m^{next}) * m^{next}
  3) c(rs) = u'^{-1}( (β R / G) * E[ u'(c^{next}) / Ψ ] )
  4) m = rs + c(rs),  a(m) = c(rs) / m

"""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import interp1d
from scipy.optimize import brentq


# -------------------------------------------------------------------
# Buffer-stock saving planning problem
# -------------------------------------------------------------------
class PlanningProblem:
    def __init__(
        self,
        gamma: float = 2.0,         # Coefficient of relative risk aversion
        beta: float = 0.95,         # Discount factor
        r: float = 0.02,            # Interest rate
        g: float = 0.01,            # Growth rate of permanent income
        sigma_tran: float = 0.1,    # Std. dev. of transitory shock (log)
        sigma_perm: float = 0.1,    # Std. dev. of permanent shock (log)
        unemp_prob: float = 0.005,  # Probability of unemployment
        theta_low: float = 0.3,     # Earnings in unemployment
        gh_order: int = 7           # Gauss–Hermite quadrature order
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

        # Gauss–Hermite nodes/weights
        x, w = hermgauss(gh_order)
        self.gh_x = np.sqrt(2.0) * x
        self.gh_w = w / np.sqrt(np.pi)

        # ---- Shock grids & joint quadrature ----
        # Transitory Θ: unemployment mass at θ_low plus lognormal part.
        theta_nodes = np.concatenate(([self.theta_low], np.exp(self.sigma_tran * self.gh_x + self.mu_z)))
        theta_w = np.concatenate(([self.unemp_prob], (1.0 - self.unemp_prob) * self.gh_w))

        # Permanent Ψ: lognormal
        psi_nodes = np.exp(self.sigma_perm * self.gh_x - 0.5 * self.sigma_perm**2)
        psi_w = self.gh_w

        # All (Θ, Ψ) pairs with joint probabilities = outer product of marginals.
        T, P = np.meshgrid(theta_nodes, psi_nodes, indexing="ij")
        self.theta = T.ravel()
        self.psi = P.ravel()
        self.probs = (theta_w[:, None] * psi_w[None, :]).ravel()

    # ------------------------------------------------------------------
    # Grid setup, and spline construction
    # ------------------------------------------------------------------

    def make_grid(self, points, span=(1e-6, 15.0)):
        m_min, m_max = span
        return np.linspace(m_min, m_max, points)
    
    def make_spline(self, x: np.ndarray, y: np.ndarray, egm: bool = False) -> interp1d:
        """
        Create a PCHIP interpolator for the policy function a(m).
        Prepends the point (0, 1) to enforce the boundary condition that
        at zero cash-on-hand (m=0), the consumption share is 1.
        """
        if egm:
            # Prepend boundary condition: a(0) = 1
            x = np.insert(x, 0, 0.0)
            y = np.insert(y, 0, 1.0)

        return interp1d(x, y, kind='linear', fill_value="extrapolate")

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

    def m_next(self, rs):
        """Next-period cash-on-hand: m_{t+1} = (R / (G Ψ_{t+1})) * rs_t + Θ_{t+1} with rs_t = m_t - c_t"""
        return self.R / (self.G * self.psi) * rs + self.theta

    def consumption_from_share(self, a, m):
        """Map share a(m) to consumption level:c_t = a_t(m_t) * m_t."""
        return a * m
    
    def resources_saved(self, m, c):
        """Resources saved rs_t = m_t - c_t."""
        return m - c
    
    def cash_on_hand(self, m):
        """Identity function for cash-on-hand state."""
        return np.asarray(m)

    # ------------------------------------------------------------------
    # Bellman functionality
    # ------------------------------------------------------------------

    def bellman(self, a, m_t, value):
        """Bellman operator given consumption-share policy a(m) and value function v(m)."""
        c_t = self.consumption_from_share(a, m_t)
        rs_t = self.resources_saved(m_t, c_t)
        m_next = self.m_next(rs_t)
        cont_value = np.dot(self.probs, value(m_next))
        return self.u(c_t) + self.beta * cont_value

    # ---------------------------
    # Euler helpers
    # ---------------------------

    def demu_next(self, c_next: np.ndarray) -> float:
        # (β R / G^γ) * E[ u'(c_{t+1}) * Ψ_{t+1}^{-γ} ]
        weights = self.probs * (self.psi ** (-self.gamma))
        return (self.beta * self.R / (self.G ** self.gamma)) * np.dot(weights, self.u_prime(c_next))

    def euler_residual(self, a: float, m_t: float, policy: interp1d) -> float:
        """Euler residual."""
        c_t = self.consumption_from_share(a, m_t)
        rs_t = self.resources_saved(m_t, c_t)
        m_next = self.m_next(rs_t)
        a_next = policy(m_next)
        c_next = self.consumption_from_share(a_next, m_next)
        rhs = self.demu_next(c_next)
        return self.u_prime(c_t) - rhs
    
    def c_from_euler(self, c_next: np.ndarray) -> float:
        """Recover c_t from: u'(c_t) = (β R / G) E[ u'(c_{t+1}) / Ψ_{t+1} ]."""
        mu = self.demu_next(c_next)
        return self.u_prime_inv(mu)
    
    def _is_binding(self, c_t: float, m_t: float) -> bool:
        """Borrowing constraint c_t <= m_t. Treat as binding when savings m_t - c_t is effectively zero."""
        return (m_t - c_t) <= 1e-3 * m_t
    

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
                if self._is_binding(c_t, m_t):
                    # Constraint binds => inequality holds; do not penalize
                    err[t] = np.nan
                else:
                    a_t_share = c_t / m_t  # implied share
                    res = self.euler_residual(a_t_share, m_t, policy)
                    # log10 of absolute residual
                    err[t] = np.log10(abs(res))
            
            euler_errors[i, :] = err
            states[i, :] = m_path
            avg_return += U / grid_len

        return euler_errors.ravel(), states.ravel(), avg_return


    #-------------------------------------------------------------------
    # Policy evaluation: value function from policy
    #-------------------------------------------------------------------

    def V_from_a(self, grid, a_policy, conv) -> interp1d:
        """Compute value function on grid by iterating the Bellman equation under a fixed policy."""
        Vn = np.zeros(len(grid))
        V = self.make_spline(grid, Vn)
        a = a_policy(grid)
        while True:
            # Policy evaluation step: compute Bellman operator under fixed policy
            Vn = self.bellman(a, grid, V)
            # Check convergence and update spline
            err = np.max(np.abs(Vn - V(grid)))
            V = self.make_spline(grid, Vn)
            if err < conv:
                return V

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
    
    def TI(self, model: PlanningProblem) -> interp1d:
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

    # ---------------------------
    # Endogenous Grid Method (EGM)
    # ---------------------------
    def EGM(self, model: PlanningProblem) -> interp1d:
        """
        Compute optimal consumption-share policy a(m) by EGM.
        Works on a grid for 'rs' (resources saved before shocks) and maps back to current m.
        """
        # --- Setup -------------------------------------------------------
        rs_grid, a_new, a_policy = self._init_policy(model)
        rs_next = rs_grid.copy()

        while True:
            a_old = a_new.copy()

            # Containers on the endogenous grid (current-period objects)
            m_t = np.empty_like(rs_grid, dtype=float)  # implied current m for each rs
            a_t = np.empty_like(rs_grid, dtype=float)  # implied current a_t = c / m

            # EGM core:
            # 1) continuation at rs via m_next(rs; Θ,Ψ)
            # 2) current c from Euler given c_{t+1}=a_{t+1}(m_{t+1})*m_{t+1}
            # 3) map back to current m via m = c + rs
            # 4) current share a_t = c / m
            for i, rs in enumerate(rs_next):
                # m_{t+1}(Θ,Ψ) as a function of resources saved rs
                m_next = model.m_next(rs)               # vector over (Θ,Ψ)

                # Continuation policy next period
                a_next = a_policy(m_next)               # a_{t+1}(m_{t+1})
                c_next = a_next * m_next                # c_{t+1}(Θ,Ψ)

                # Back out c_t from Euler equation
                c_curr = model.c_from_euler(c_next)     # scalar c_t(rs)

                # Budget identity to current state
                m_curr = c_curr + rs                    # m_t = c_t + rs
                m_t[i] = m_curr
                a_t[i] = c_curr / m_curr

            # Interpolate a_t over current m (the endogenous grid)
            order = np.argsort(m_t)
            m_sorted, a_sorted = m_t[order], a_t[order]
            a_policy = model.make_spline(m_sorted, a_sorted, egm=True)

            # --- Convergence check ---------------------------------------
            converged, a_new = self._check_convergence(a_old, a_policy, rs_grid)
            if converged:
                return a_policy