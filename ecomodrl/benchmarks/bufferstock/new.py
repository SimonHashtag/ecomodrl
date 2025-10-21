from typing import Union
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


# -------------------------------------------------------------------
# Buffer-stock saving planning problem (model only)
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
    # State transitions / resource accounting (single entrypoint: cash_on_hand)
    # ------------------------------------------------------------------

    def cash_on_hand(self, x_prev: float) -> Union[float, np.ndarray]:
        """Normalized cash-on-hand m_t from last period's end-of-period assets x_{t-1}"""
        return self.R / (self.G * self.psi) * x_prev + self.theta

    def consumption_from_share(self, a: float, m: float) -> float:
        return a * m

    def next_x(self, m: float, c: float) -> float:
        return m - c
    
    # ---------------------------
    # Euler helpers
    # ---------------------------

    def euler_residual(self, a: float, m: float, policy: PchipInterpolator):
        # Current choices
        c_t = self.consumption_from_share(a, m)
        x_t = self.next_x(m, c_t)

        # Next period (vector over GH nodes)
        m_next = self.cash_on_hand(x_t)
        a_next = policy(m_next)
        c_next = self.consumption_from_share(a_next, m_next)

        rhs = self.beta * self.R / self.G * np.dot(self.probs, self.u_prime(c_next / self.psi))
        return self.u_prime(c_t) - rhs
    
    def c_from_euler(self, c_next):
        """Compute current optimal consumption c_t from Euler equation given c_{t+1}(θ,ψ)."""
        return self.u_prime_inv(self.beta * self.R / self.G * np.dot(self.probs, self.u_prime(c_next / self.psi)))

    # ---------------------------
    # Policy evaluation & errors
    # ---------------------------
    def get_euler_errors(self, grid, policy, episodes=100):
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
        lower_m: float = 0.1,
        upper_m: float = 15.0,
        tol: float = 1e-5,
        conv: float = 1e-5,
        min_share: float = 1e-6,
        verbose: bool = True,
    ):
        self.grid_points = grid_points
        self.lower_m = lower_m
        self.upper_m = upper_m
        self.tol = tol
        self.conv = conv
        self.min_share = min_share
        self.verbose = verbose

    # ---------------------------
    # Time Iterations (TI)
    # ---------------------------
    def TI(self, model: PlanningProblem) -> Union[np.ndarray, PchipInterpolator]:
        """
        Compute optimal consumption-share policy a(m) by time iterations.
        Returns (m_grid, a_policy), where a_policy maps m -> a(m).
        """

        m_grid = np.linspace(self.lower_m, self.upper_m, self.grid_points)

        a_policy = PchipInterpolator(
            np.concatenate(([0.0], m_grid)),
            np.concatenate(([1.0], np.ones(self.grid_points) * 0.1)),
            extrapolate=True,
        )
        a_vals = a_policy(m_grid)
        euler_residual = model.euler_residual

        while True:
            for i, m in enumerate(m_grid):
                lo = self.min_share
                hi = 1.0  # consumption share upper bound

                f_lo = euler_residual(lo, m, a_policy)
                f_hi = euler_residual(hi, m, a_policy)

                if f_lo * f_hi > 0:
                    # No interior root. Choose corner consistent with sign.
                    a_vals[i] = hi if f_lo > 0 else lo
                else:
                    a_vals[i] = brentq(euler_residual, lo, hi, xtol=self.tol, args=(m, a_policy))

            diff = np.max(np.abs(a_policy(m_grid) - a_vals))
            if diff < self.conv:
                break

            if self.verbose:
                print(f"Convergence: {diff}.  Convergence criterion: {self.conv}.")

            a_policy = PchipInterpolator(
                np.concatenate(([0.0], m_grid)),
                np.concatenate(([1.0], a_vals)),
                extrapolate=True,
            )

        return m_grid, PchipInterpolator(
            np.concatenate(([0.0], m_grid)),
            np.concatenate(([1.0], a_vals)),
            extrapolate=True,
        )

    # ---------------------------
    # Endogenous Grid Method (EGM)
    # ---------------------------
    def EGM(self, model: PlanningProblem) -> Union[np.ndarray, PchipInterpolator]:
        """
        Compute optimal consumption-share policy a(m) by EGM.
        Returns (x_grid, a_policy), where a_policy maps m -> a(m).
        """
        c_from_euler = model.c_from_euler

        # Grid over end-of-period assets x (post-decision)
        x_grid = np.linspace(self.lower_m, self.upper_m, self.grid_points)

        # Initial guess for a(m)
        a_policy = PchipInterpolator(
            np.concatenate(([0.0], x_grid)),  # seed; immediately replaced with m-grid below
            np.concatenate(([1.0], np.ones(self.grid_points) * 0.1)),
            extrapolate=True,
        )
        a_policy_old = PchipInterpolator(
            np.concatenate(([0.0], x_grid)),
            np.concatenate(([1.0], np.ones(self.grid_points) * 0.2)),
            extrapolate=True,
        )

        m_vals = x_grid.copy()    # will hold implied current m
        c_vals = np.empty(self.grid_points)  # will hold implied current c
        m_sorted = m_vals.copy()  # fallback before first iteration

        while np.max(np.abs(a_policy_old(m_vals) - a_policy(m_vals))) > self.conv:
            a_policy_old = a_policy
            for i, x in enumerate(x_grid):
                m_next = model.cash_on_hand(x)      # vector over (θ, ψ)
                c_next = a_policy(m_next) * m_next  # c_{t+1}(θ,ψ)
                c_curr = c_from_euler(c_next)       # c_t from Euler
                m_curr = x + c_curr                 # budget: m_t = x_t + c_t
                c_vals[i] = c_curr
                m_vals[i] = m_curr

            order = np.argsort(m_vals)
            m_sorted, c_sorted = m_vals[order], c_vals[order]
            a_policy = PchipInterpolator(
                np.concatenate(([0.0], m_sorted)),
                np.concatenate(([1.0], c_sorted / m_sorted)),  # a(m) = c/m
                extrapolate=True,
            )

            if self.verbose:
                print(
                    f"Convergence: {np.max(np.abs(a_policy_old(m_vals) - a_policy(m_vals)))}.  "
                    f"Convergence criterion: {self.conv}."
                )

        return x_grid, PchipInterpolator(
            np.concatenate(([0.0], m_sorted)),
            np.concatenate(([1.0], (c_sorted / m_sorted))),
            extrapolate=True,
        )


# -------------------------------------------------------------------
# Demo / comparison
# -------------------------------------------------------------------
if __name__ == "__main__":
    
    # Model and solvers
    model = PlanningProblem()
    solvers = PlanningSolvers(grid_points=100, lower_m=1e-6, upper_m=15.0, conv=1e-10, verbose=False)

    # Time iterations
    start = time.time()
    _, a_ti = solvers.TI(model)
    print(f"Time iterations took {time.time() - start} seconds.")

    # EGM
    start = time.time()
    _, a_egm = solvers.EGM(model)
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
