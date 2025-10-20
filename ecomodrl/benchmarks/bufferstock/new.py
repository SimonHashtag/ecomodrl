from typing import Union
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


#-------------------------------------------------------------------
# Buffer stock saving planning problem
#-------------------------------------------------------------------
class PlanningProblem():
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
        n_knots: int = 8            # Gauss–Hermite quadrature order
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

        # ---- Gauss–Hermite nodes/weights ----
        gh_x, gh_w = hermgauss(n_knots)
        gh_x *= np.sqrt(2.0)
        gh_w /= np.sqrt(np.pi)

        # ---- Shock grids & joint quadrature ----
        # Transitory θ: unemployment mass at θ_low plus lognormal part.
        theta_nodes = np.concatenate(([self.theta_low], np.exp(self.sigma_tran * gh_x + self.mu_z)))
        theta_w = np.concatenate(([self.unemp_prob], (1.0 - self.unemp_prob) * gh_w))

        # Permanent ψ: lognormal
        psi_nodes = np.exp(self.sigma_perm * gh_x - 0.5 * self.sigma_perm**2)
        psi_w = gh_w

        # All (θ, ψ) pairs with joint probabilities = outer product of marginals.
        T, P = np.meshgrid(theta_nodes, psi_nodes, indexing="ij")
        self.theta = T.ravel()
        self.psi = P.ravel()
        self.probs = (theta_w[:, None] * psi_w[None, :]).ravel()

    # ------------------------------------------------------------------
    # Grid & spline helpers
    # ------------------------------------------------------------------

    def make_grid(self, points: int = 1000, span=(0.2, 5.0)):
        m_min, m_max = span
        return np.linspace(m_min, m_max, points)

    def make_spline(self, grid, y):
        """PCHIP spline (monotone, shape-preserving)."""
        return PchipInterpolator(grid, y, extrapolate=True)

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def u(self, c):
        return c ** (1 - self.gamma) / (1 - self.gamma) if self.gamma != 1 else np.log(c)

    def u_prime(self, c):
        return c ** (-self.gamma)

    def u_prime_inv(self, mu):
        return mu ** (-1 / self.gamma)

    # ------------------------------------------------------------------
    # State transitions / resource accounting (single entrypoint: cash_on_hand)
    # ------------------------------------------------------------------

    def cash_on_hand(self, x_prev: float) -> Union[float, np.ndarray]:
        """
        Normalized cash-on-hand m_t from last period's end-of-period assets x_{t-1}
        and shocks (theta_t, psi_t):
            m_t = (R / (G * psi_t)) * x_{t-1} + theta_t
        Returns values across the GH shock nodes (vector) for expectation calculations.
        """
        return self.R / (self.G * self.psi) * x_prev + self.theta

    def consumption_from_share(self, a: float, m: float) -> float:
        return a * m

    def next_x(self, m: float, c: float) -> float:
        return m - c

    # ------------------------------------------------------------------
    # Expectations used by EGM / Euler equation
    # ------------------------------------------------------------------

    def c_from_euler(self, c_next):
        Emup = np.dot(self.probs, self.u_prime(c_next / self.psi))
        return self.u_prime_inv(self.beta * self.R / self.G * Emup)

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

    # ------------------------------------------------------------------
    # Simulation & Euler error diagnostics
    # ------------------------------------------------------------------

    def get_euler_errors(self, grid, policy, n_eps=100):
        u = self.u
        beta = self.beta

        T = int(n_eps)
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

        return euler_errors[~np.isnan(euler_errors)].ravel(), states.ravel(), avg_return



#-------------------------------------------------------------------
# Solvers (EGM and Time Iterations moved here)
#-------------------------------------------------------------------

class PlanningSolvers:
    def __init__(
        self,
        grid_points: int = 1000,    # number of state grid points
        tol: float = 1e-5,          # optimization tolerance
        conv: float = 1e-5,         # convergence tolerance (consumption share policy)
        min_share: float = 1e-6     # feasibility guard for shares
    ):
        self.grid_points = grid_points
        self.tol = tol
        self.conv = conv
        self.min_share = min_share

    #-------------------------------------------------------------------
    # Time iterations
    #-------------------------------------------------------------------

    def TI(self, 
            model: PlanningProblem,
            grid_points:int = 1000, 
            lower_limit:float=0.1, 
            upper_limit:float=15., 
            tol:float=1e-5, 
            conv:float=0.1) -> Union[np.ndarray, PchipInterpolator]:
        """ 
        Computes optimal consumption policy by time iterations
        
        grid_points: number of grid points
        lower_limit: lower limit of grid
        upper_limit: upper limit of grid
        tol: tolerance level for minimization
        conv: convergence criterion
        
        return: grid and optimal consumption policy as CubicSpline object
        """
        
        grid = np.linspace(lower_limit, upper_limit, grid_points)
        aopt = PchipInterpolator(np.concatenate(([0.0], grid)), np.concatenate(([1.0], np.ones(grid_points)*0.1)), extrapolate=True)
        a_vals = aopt(grid)
        euler_residual = model.euler_residual
        
        while True:
            for i, m in enumerate(grid):
                lo = 1e-6
                hi = max(1e-6, 1.0)  # ensure lo < hi

                flo = euler_residual(lo, m, aopt)
                fhi = euler_residual(hi, m, aopt)

                if flo * fhi > 0:
                    # No interior root. Choose corner consistent with the sign:
                    # Residual>0 everywhere -> increase c -> consume at upper corner.
                    # Residual<0 everywhere -> decrease c -> consume at lower corner.
                    a_vals[i] = hi if flo > 0 else lo
                else:
                    a_vals[i] = brentq(euler_residual, lo, hi, xtol=tol, args=(m, aopt))
            if max(abs(aopt(grid)-a_vals)) < conv:
                break

            aopt = PchipInterpolator(np.concatenate(([0.], grid)), np.concatenate(([1.0], a_vals)), extrapolate=True)
        return grid, PchipInterpolator(np.concatenate(([0.], grid)), np.concatenate(([1.0], a_vals)), extrapolate=True)

    #-------------------------------------------------------------------
    # EGM
    #-------------------------------------------------------------------

    def EGM(self, 
            model: PlanningProblem,
            grid_points:int = 1000, 
            lower_limit:float=0.1, 
            upper_limit:float=15., 
            conv:float=0.1) -> Union[np.ndarray, PchipInterpolator]:
        """
        Computes optimal consumption policy by endogenous grid method (EGM)
        
        grid_points: number of grid points
        lower_limit: lower limit of grid
        upper_limit: upper limit of grid
        conv: convergence criterion
        
        return: grid and optimal consumption policy as CubicSpline object
        """

        x_grid = np.linspace(lower_limit, upper_limit, grid_points)
        a_policy = PchipInterpolator(np.concatenate(([0.0], x_grid)), np.concatenate(([1.0], np.ones(grid_points)*0.1)), extrapolate=True)
        a_policy_old = PchipInterpolator(np.concatenate(([0.0], x_grid)), np.concatenate(([1.0], np.ones(grid_points)*0.2)), extrapolate=True)
        m_vals = x_grid.copy()
        c_vals = np.empty(grid_points)
        while max(abs(a_policy_old(m_vals)-a_policy(m_vals))) > conv:
            a_policy_old = a_policy
            for i, x in enumerate(x_grid):
                m_next = model.cash_on_hand(x)              # m_{t+1} = m(x_t, shocks)
                a_next = a_policy(m_next)                   # a_{t+1} = a(m_{t+1})
                c_next = a_next * m_next                    # c_{t+1}
                c_vals[i] = model.c_from_euler(c_next)      # c_t from Euler
                m_vals[i] = x + c_vals[i]                   # m_t = x_t + c_t
            order = np.argsort(m_vals)
            m_sorted, c_sorted = m_vals[order], c_vals[order]
            a_policy = PchipInterpolator(np.concatenate(([0.0], m_sorted)), np.concatenate(([1.0], c_sorted/m_sorted)), extrapolate=True)

        return x_grid, PchipInterpolator(np.concatenate(([0.0], m_sorted)), np.concatenate(([1.0], c_sorted/m_sorted)), extrapolate=True)
    


if __name__ == "__main__":
    pp = PlanningProblem()
    solvers = PlanningSolvers()

    # Time iterations 
    start = time.time()
    _, policy_ti = solvers.TI(
        pp,
        grid_points=50, lower_limit=1.0, upper_limit=15., 
        tol=1e-5, conv=1e-6)
    print(f'Time iterations took {time.time()-start} seconds.')
    
    # EGM
    start = time.time()
    _, policy_egm = solvers.EGM(
        pp,
        grid_points=100, lower_limit=1e-6, upper_limit=15., 
        conv=1e-10)
    print(f'EGM took {time.time()-start} seconds.')

    grid = np.linspace(1.5, 15., 1000)
    # Euler equation errors & return comparison
    euler_errors_ti, states_ti, U_ti = pp.get_euler_errors(grid, policy_ti, n_eps=10)
    euler_errors_egm, states_egm, U_egm = pp.get_euler_errors(grid, policy_egm, n_eps=10)
    print(f'Mean absolute Euler equation error: \r\n  TI: {np.mean(euler_errors_ti)}       EGM: {np.mean(euler_errors_egm)}')
    print(f'The optimal average discounted episode return is: \r\n  TI: {U_ti}      EGM: {U_egm}')
    
    grid = np.concatenate(([0], grid))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(grid, policy_ti(grid), linestyle='--', color='orange', label='Time iterations')
    ax[0].plot(grid, policy_egm(grid), linestyle='--', color='green', label='EGM')
    ax[0].set_title('Policy function')
    ax[0].set_xlabel('Normalized cash-on-hand')
    ax[0].set_ylabel('Normalized Consumption')
    ax[1].hist(euler_errors_ti, density=True, bins=50, color="orange", alpha=0.5, label='Time iterations')
    ax[1].hist(euler_errors_egm, density=True, bins=50, color="green", alpha=0.5, label='EGM')
    ax[1].set_title('Log10 absolute Euler equation errors')
    ax[1].set_xlabel('Log10 absolute Euler equation error')
    ax[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.legend()
    plt.show()