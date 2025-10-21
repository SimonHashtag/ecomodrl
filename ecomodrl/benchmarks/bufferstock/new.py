"""
Buffer-stock saving model with permanent-income normalization.

Notation (all variables normalized by permanent income P_t):
  - m_t : pre-decision cash-on-hand (resources at start of period t)
  - a(m) ∈[0,1] : consumption share policy; consumption is a fraction of m
  - c_t : consumption level  = a(m_t) * m_t
  - x_t : end-of-period assets (post-decision savings) = m_t - c_t

Law of motion for next period's cash-on-hand (given transitory θ and permanent ψ shocks):
  m_{t+1} = (R / (G ψ_{t+1})) * x_t + θ_{t+1}.

Euler equation under CRRA and normalization:
  u'(c_t) = (β R / G) * E[ u'(c_{t+1}) / ψ_{t+1} ].

EGM step (working on a grid of post-decision assets x):
  1) m^{next}(x; θ,ψ) = (R / (G ψ)) * x + θ
  2) c^{next} = a(m^{next}) * m^{next}
  3) c(x) = u'^{-1}( (β R / G) * E[ u'(c^{next}) / ψ ] )
  4) m = x + c(x),  a(m) = c(x) / m

This file enforces the following naming scheme:
  c = total consumption level, a = consumption share policy,
  m = pre-decision cash-on-hand, x = post-decision assets.
"""

from typing import Union
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import PchipInterpolator
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

        # Mean adjustment so E[theta]=1
        self.mu_z = np.log((1 - self.unemp_prob * self.theta_low) / (1 - self.unemp_prob)) - 0.5 * self.sigma_tran**2

        # Gauss–Hermite nodes/weights
        x, w = hermgauss(gh_order)
        self.gh_x = np.sqrt(2.0) * x
        self.gh_w = w / np.sqrt(np.pi)

        # ---- Shock grids & joint quadrature ----
        # Transitory θ: unemployment mass at θ_low plus lognormal part.
        theta_nodes = np.concatenate(([self.theta_low], np.exp(self.sigma_tran * self.gh_x + self.mu_z)))
        theta_w = np.concatenate(([self.unemp_prob], (1.0 - self.unemp_prob) * self.gh_w))

        # Permanent ψ: lognormal
        psi_nodes = np.exp(self.sigma_perm * self.gh_x - 0.5 * self.sigma_perm**2)
        psi_w = self.gh_w

        # All (θ, ψ) pairs with joint probabilities = outer product of marginals.
        T, P = np.meshgrid(theta_nodes, psi_nodes, indexing="ij")
        self.theta = T.ravel()
        self.psi = P.ravel()
        self.probs = (theta_w[:, None] * psi_w[None, :]).ravel()

    # ------------------------------------------------------------------
    # Grid setup, and spline construction
    # ------------------------------------------------------------------

    def make_grid(self, points, span=(0.01, 15.0)):
        m_min, m_max = span
        return np.linspace(m_min, m_max, points)
    
    def make_spline(self, x: np.ndarray, y: np.ndarray) -> PchipInterpolator:
        """
        Create a PCHIP interpolator for the policy function a(m).

        Prepends the point (0, 1) to enforce the boundary condition that
        at zero cash-on-hand (m=0), the consumption share is 1 (a=1).
        """
        # Prepend boundary condition: a(0) = 1
        x_ext = np.insert(x, 0, 0.0)
        y_ext = np.insert(y, 0, 1.0)

        return PchipInterpolator(x_ext, y_ext, extrapolate=True)

    # ------------------------------------------------------------------
    # Model primitives (preferences & technology)
    # ------------------------------------------------------------------

    def u(self, c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CRRA utility function: u(c)."""
        return c ** (1 - self.gamma) / (1 - self.gamma) if self.gamma != 1 else np.log(c)

    def u_prime(self, c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Marginal utility of consumption: u'(c)."""
        return c ** (-self.gamma)

    def u_prime_inv(self, mu: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Inverse marginal utility: given μ = u'(c), return c = u'^{-1}(μ)."""
        return mu ** (-1 / self.gamma)

    # ------------------------------------------------------------------
    # State transitions / resource accounting (single entrypoint: cash_on_hand)
    # ------------------------------------------------------------------

    def cash_on_hand(self, x_prev: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Next-period cash-on-hand m_{t+1} from last period's end-of-period assets x_t:
            m_{t+1} = (R / (G * ψ_{t+1})) * x_t + θ_{t+1}.
        Vectorized over (θ, ψ) Gauss–Hermite nodes.
        """
        return self.R / (self.G * self.psi) * x_prev + self.theta

    def consumption_from_share(self, a: Union[float, np.ndarray], m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Map share a(m) to consumption level: c = a * m."""
        return a * m

    def next_x(self, m: Union[float, np.ndarray], c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """End-of-period assets (post-decision): x = m - c."""
        return m - c
    
    # ---------------------------
    # Euler helpers
    # ---------------------------

    def euler_residual(self, a: float, m: float, policy: PchipInterpolator) -> float:
        """
        Residual of Euler equation evaluated at (a, m) with policy a(·).
        Zero at optimum. Uses:
          c_t = a * m,  x_t = m - c_t,
          m' = (R / (G ψ)) * x_t + θ,  c' = a(m') * m'.
        """
        # Current choices
        c_t = self.consumption_from_share(a, m)
        x_t = self.next_x(m, c_t)

        # Next period (vector over GH nodes)
        m_next = self.cash_on_hand(x_t)
        a_next = policy(m_next)
        c_next = self.consumption_from_share(a_next, m_next)

        rhs = self.beta * self.R / self.G * np.dot(self.probs, self.u_prime(c_next / self.psi))
        return self.u_prime(c_t) - rhs
    
    def c_from_euler(self, c_next: np.ndarray) -> float:
        """
        Compute current optimal consumption c_t from Euler equation given c_{t+1}(θ,ψ):
            c_t = u'^{-1}( (β R / G) * E[ u'(c_{t+1}) / ψ_{t+1} ] ).
        """
        mu = self.beta * self.R / self.G * np.dot(self.probs, self.u_prime(c_next / self.psi))
        return self.u_prime_inv(mu)

    # ---------------------------
    # Policy evaluation & errors
    # ---------------------------
    def get_euler_errors(self, grid, policy, episodes=100):
        """
        Simulate Euler residuals along sample paths starting from each m in `grid`,
        using the provided consumption-share policy a(·). Returns (errors, states, avg_return).
        """
        u = self.u
        beta = self.beta

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
                a_t = float(policy(m_path[t]))  # consumption share
                c_t = self.consumption_from_share(a_t, m_path[t])
                x_t = self.next_x(m_path[t], c_t)

                # draw shocks
                if np.random.uniform(0.0, 1.0) < self.unemp_prob:
                    theta_draw = self.theta_low
                else:
                    theta_draw = np.exp(np.random.normal(self.mu_z, self.sigma_tran))
                psi_draw = np.exp(np.random.normal(-0.5 * self.sigma_perm**2, self.sigma_perm))

                m_path[t + 1] = self.R / (self.G * psi_draw) * x_t + theta_draw

                c_path[t] = c_t
                U += power * u(c_t)
                power *= beta

            # Euler error along the realized path
            err = np.empty(T, dtype=float)
            for t, (c_t, m_t) in enumerate(zip(c_path, m_path[:-1])):
                a_t = c_t / m_t  # implied share
                err[t] = np.log10(abs(self.euler_residual(a_t, m_t, policy)))
            
            euler_errors[i, :] = err
            states[i, :] = m_path
            avg_return += U / grid_len

        return euler_errors.ravel(), states.ravel(), avg_return

# ------------------------------------------------------------------- 
# Solvers (TI and EGM only; model lives in PlanningProblem)
# -------------------------------------------------------------------
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
        a_new = np.full_like(grid, 0.1)  # initial guess for shares a(m)
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

                def euler_root(a_candidate: float) -> float:
                    # Residual at (a_candidate, m) under current policy a_policy
                    return model.euler_residual(a_candidate, m, a_policy)

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
    def EGM(self, model: PlanningProblem) -> PchipInterpolator:
        """
        Compute optimal consumption-share policy a(m) by EGM.
        Works on a grid for post-decision assets x and maps back to current m.
        """
        # --- Setup -------------------------------------------------------
        x_grid, a_new, a_policy = self._init_policy(model)  # reuse allocation; interpret as x-grid
        x_next = x_grid.copy()

        while True:
            a_old = a_new.copy()

            # Containers for current-period objects on the endogenous grid
            m_t = np.empty_like(x_grid, dtype=float)  # implied current m for each x
            a_t = np.empty_like(x_grid, dtype=float)  # implied current a = c / m

            # 1) continuation at x
            # 2) current c from Euler at x
            # 3) map back to current m via budget m = c + x
            # 4) current share a = c / m
            for i, x in enumerate(x_next):
                m_next = model.cash_on_hand(x)        # vector over (θ, ψ)
                a_next = a_policy(m_next)             # a_{t+1}(m_{t+1})
                c_next = a_next * m_next              # c_{t+1}(θ, ψ)
                c_curr = model.c_from_euler(c_next)   # scalar c_t(x)
                m_curr = c_curr + x                   # budget: m_t = c_t + x
                m_t[i] = m_curr
                a_t[i] = c_curr / m_curr

            # 5) Interpolate consumption-share policy over current m
            order = np.argsort(m_t)
            m_sorted, a_sorted = m_t[order], a_t[order]
            a_policy = model.make_spline(m_sorted, a_sorted)

            # --- Convergence check ---------------------------------------
            converged, a_new = self._check_convergence(a_old, a_policy, m_t)
            if converged:
                return a_policy


# -------------------------------------------------------------------
# Demo / comparison
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Model and solvers
    model = PlanningProblem()
    solvers = PlanningSolvers(grid_points=100, conv=1e-3)

    # Time iterations
    start = time.time()
    a_ti = solvers.TI(model)
    print(f"Time iterations took {time.time() - start} seconds.")

    # EGM
    start = time.time()
    a_egm = solvers.EGM(model)
    print(f"EGM took {time.time() - start} seconds.")

    m_grid = np.linspace(1.5, 15.0, 1000)

    # Euler equation errors & return comparison
    euler_errors_ti, m_states_ti, U_ti = model.get_euler_errors(m_grid, a_ti, episodes=10)
    euler_errors_egm, m_states_egm, U_egm = model.get_euler_errors(m_grid, a_egm, episodes=10)

    print(
        "Mean absolute Euler equation error: \n"
        f"  TI: {np.mean(euler_errors_ti)}       EGM: {np.mean(euler_errors_egm)}"
    )
    print(
        "The optimal average discounted episode return is: \n"
        f"  TI: {U_ti}      EGM: {U_egm}"
    )

    m_plot = np.concatenate(([0.0], m_grid))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(m_plot, a_ti(m_plot), linestyle='--', color='orange', label='Time iterations')
    ax[0].plot(m_plot, a_egm(m_plot), linestyle='--', color='green', label='EGM')
    ax[0].set_title('Policy function a(m)')
    ax[0].set_xlabel('Cash-on-hand m')
    ax[0].set_ylabel('Consumption share a')
    ax[1].hist(euler_errors_ti, density=True, bins=50, color="orange", alpha=0.5, label='Time iterations')
    ax[1].hist(euler_errors_egm, density=True, bins=50, color="green", alpha=0.5, label='EGM')
    ax[1].set_title('Log10 absolute Euler equation errors')
    ax[1].set_xlabel('Log10 |Euler residual|')
    ax[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.legend()
    plt.show()
