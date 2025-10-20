import numpy as np
import matplotlib.pyplot as plt
from time import time
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


#-------------------------------------------------------------------
# Buffer–Stock planning problem (preferences, accounting, Euler helpers)
#-------------------------------------------------------------------

class PlanningProblem():
    def __init__(
        self, 
        gamma: float = 2,        # Coefficient of relative risk aversion
        beta: float = 0.95,      # Discount factor
        r: float = 0.02,         # Interest rate
        g: float = 0.01,         # Growth rate of permanent income
        sigma_tran: float = 0.10,# Std. dev. of log transitory shock
        sigma_perm: float = 0.10,# Std. dev. of log permanent shock
        prob: float = 0.005,     # Probability of unemployment
        theta_low: float = 0.3,  # Earnings in unemployment
        gh_order: int = 7        # Gauss–Hermite quadrature order
    ):
        # Preferences and growth/return
        self.gamma, self.beta = gamma, beta
        self.R = 1.0 + r
        self.G = 1.0 + g

        # Shock process parameters
        self.sigma_tran, self.sigma_perm = sigma_tran, sigma_perm
        self.prob, self.theta_low = prob, theta_low

        # Gauss–Hermite nodes/weights (standard normal)
        x, w = hermgauss(gh_order)
        self.gh_x = np.sqrt(2.0) * x
        self.gh_w = w / np.sqrt(np.pi)

        # Transitory shock mean shift so that E[theta]=1 after mixing unemployment mass
        self.mu_z = np.log((1.0 - self.prob * self.theta_low) / (1.0 - self.prob)) - 0.5 * self.sigma_tran**2

        # Continuous GH nodes for transitory and permanent components (both mean 1)
        theta_cont = np.exp(self.sigma_tran * self.gh_x + self.mu_z)                         # (gh,)
        psi_cont   = np.exp(self.sigma_perm * self.gh_x - 0.5 * self.sigma_perm**2)          # (gh,)

        # Build joint (theta, psi) nodes with:
        #  - a mass block for unemployment at theta_low paired with all psi GH nodes
        #  - a continuous block for GH(theta) × GH(psi)
        gh = len(self.gh_w)

        # Mass block (length gh)
        theta_mass = np.full(gh, self.theta_low)
        psi_mass   = psi_cont.copy()
        w_mass     = self.prob * self.gh_w                                                   # P(theta=theta_low) * w_psi(j)

        # Continuous block (length gh*gh)
        theta_cont_blk = np.repeat(theta_cont, gh)
        psi_cont_blk   = np.tile(psi_cont, gh)
        w_theta_cont   = (1.0 - self.prob) * self.gh_w
        w_cont         = np.repeat(w_theta_cont, gh) * np.tile(self.gh_w, gh)

        # Concatenate to final joint arrays (length gh*(gh+1))
        self.theta_nodes = np.concatenate([theta_mass, theta_cont_blk])
        self.psi_nodes   = np.concatenate([psi_mass,  psi_cont_blk])
        self.joint_w     = np.concatenate([w_mass,    w_cont])

    # ------------------------------------------------------------------
    # Grid setup and interpolation constructors (1D)
    # ------------------------------------------------------------------

    @property
    def m_bind(self) -> float:
        """
        Borrowing-constraint threshold where a(m)=1 is just optimal:
          m_bind = [ beta * (R/G) * E[(theta/psi)^(-gamma)] ]^(-1/gamma)
        Uses prebuilt GH nodes (theta_nodes, psi_nodes, joint_w).
        """
        term = np.sum(self.joint_w * (self.theta_nodes / self.psi_nodes) ** (-self.gamma))
        return (self.beta * (self.R / self.G) * term) ** (-1.0 / self.gamma)

    def make_m_grid(self, points: int = 1000, span=(0.2, 5.0)):
        """Grid for normalized resources m scaled around the threshold m_bind."""
        m_min, m_max = (s * self.m_bind for s in span)
        return np.linspace(m_min, m_max, points)

    def make_spline(self, grid, y, extrapolate=True):
        """PCHIP spline (1D; used for a(m) and V(m))."""
        return PchipInterpolator(grid, y, extrapolate=extrapolate)

    # ------------------------------------------------------------------
    # Model primitives (preferences)
    # ------------------------------------------------------------------

    def u(self, c):
        """CRRA utility function"""
        return c ** (1 - self.gamma) / (1 - self.gamma) if self.gamma != 1 else np.log(c)

    def u_prime(self, c):
        """Marginal utility of consumption"""
        return c ** (-self.gamma)

    def u_prime_inv(self, x):
        """Inverse marginal utility"""
        return x ** (-1 / self.gamma)

    # ------------------------------------------------------------------
    # Resource accounting and state transition
    # ------------------------------------------------------------------

    def consumption_from_share(self, a, m):
        """Total consumption from consumption share a_t in resources m_t."""
        return a * m

    def m_next_from(self, a_post, theta, psi):
        """Next-period normalized resources m' given post-decision assets a_post and shocks."""
        return self.R / (self.G * psi) * a_post + theta

    # ------------------------------------------------------------------
    # Bellman functionality
    # ------------------------------------------------------------------

    def bellman(self, a, m, value):
        """Bellman equation value at m for consumption share a_t; expectation over (theta, psi)."""
        c = self.consumption_from_share(a, m)
        a_post = m - c
        m_next = self.m_next_from(a_post, self.theta_nodes, self.psi_nodes)  # (J,)
        V_next = value(m_next)                                               # (J,)
        EV = np.sum(self.joint_w * V_next)
        return self.u(c) + self.beta * EV

    # ------------------------------------------------------------------
    # Euler functionality
    # ------------------------------------------------------------------

    def euler_residual(self, c_t, c_next_nodes):
        """
        Euler residual (zero at optimum):
          beta * (R/G) * E[ (c_t / (c_{t+1}/psi))^gamma ] - 1
        where the expectation is over the prebuilt joint GH nodes.
        """
        ratio = c_t / (c_next_nodes / self.psi_nodes)                        # (J,)
        term = ratio ** self.gamma
        return self.beta * (self.R / self.G) * np.sum(self.joint_w * term) - 1.0

    def c_from_euler(self, c_next_nodes):
        """
        Invert Euler equation to get current consumption from continuation consumption nodes:
          c_t = [ beta * (R/G) * E[ (c_{t+1}/psi)^(-gamma) ] ]^(-1/gamma)
        """
        Emu = self.beta * (self.R / self.G) * np.sum(self.joint_w * self.u_prime(c_next_nodes / self.psi_nodes))
        return self.u_prime_inv(Emu)

    # ------------------------------------------------------------------
    # Euler errors and value evaluation under fixed policy
    # ------------------------------------------------------------------

    def get_euler_errors(self, m_grid, policy, episodes=100, seed=123):
        """
        Compute Euler equation errors over simulated paths starting from every m in m_grid.
        Returns flattened (errors, states, avg_return).
        """
        rng = np.random.default_rng(seed)
        T = episodes

        G = self.G
        R = self.R

        grid_len = len(m_grid)
        m_states = np.empty((grid_len, T+1), dtype=float)
        euler_errors = np.empty((grid_len, T), dtype=float)
        avg_return = 0.0

        for idx, m0 in enumerate(m_grid):
            m_path = np.empty(T+1)
            c_path = np.empty(T)
            m_path[0] = m0

            U = 0.0
            power = 1.0

            for t in range(T):
                a_t = policy([m_path[t]])[0]
                c_t = self.consumption_from_share(a_t, m_path[t])
                a_post = m_path[t] - c_t

                # Simulated shocks for the path
                if rng.random() < self.prob:
                    theta = self.theta_low
                else:
                    theta = np.exp(rng.normal(self.mu_z, self.sigma_tran))
                psi = np.exp(rng.normal(-0.5 * self.sigma_perm**2, self.sigma_perm))

                # Advance state
                m_path[t+1] = self.m_next_from(a_post, theta, psi)
                c_path[t] = c_t

                # Euler error at time t using GH expectation from current state
                m_next_nodes = self.m_next_from(a_post, self.theta_nodes, self.psi_nodes)  # (J,)
                a_next_nodes = policy(m_next_nodes)                                        # (J,)
                c_next_nodes = self.consumption_from_share(a_next_nodes, m_next_nodes)     # (J,)
                euler_errors[idx, t] = self.euler_residual(c_t, c_next_nodes)

                # Discounted utility
                U += power * self.u(c_t)
                power *= self.beta

            m_states[idx, :] = m_path
            avg_return += U / grid_len

        return euler_errors.ravel(), m_states.ravel(), avg_return

    def V_from_a(self, m_grid, a_policy, conv) -> PchipInterpolator:
        """Iterate Bellman under fixed a(m) policy using 1D PCHIP interpolation."""
        Nm = len(m_grid)
        Vn = np.zeros(Nm)
        V = self.make_spline(m_grid, Vn)
        # Convergence check: evaluate previous V on the grid
        while True:
            for i, m in enumerate(m_grid):
                a_i = a_policy([m])[0]
                Vn[i] = self.bellman(a_i, m, V)
            V_prev_on_grid = V(m_grid)
            err = np.max(np.abs(Vn - V_prev_on_grid))
            V = self.make_spline(m_grid, Vn)
            if err < conv:
                break
        return V

#-------------------------------------------------------------------
# Solvers: standardized on share policy a(m)
#-------------------------------------------------------------------

class PlanningSolvers:
    def __init__(
        self,
        m_points: int = 400,       # number of resource grid points
        tol: float = 1e-5,         # optimization tolerance
        conv: float = 1e-5,        # convergence tolerance (policy)
        min_share: float = 1e-6    # feasibility guard for shares
    ):
        self.m_points = m_points
        self.tol = tol
        self.conv = conv
        self.min_share = min_share

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _init_policy(self, model: PlanningProblem):
        """Initialize m grid, consumption share array and spline policy."""
        m_grid = model.make_m_grid(points=self.m_points)
        a_new = np.full_like(m_grid, 0.2, dtype=float)
        a_policy = model.make_spline(m_grid, a_new)
        return m_grid, a_new, a_policy

    def _check_convergence(self, a_old, a_policy, m_grid):
        """Check convergence of consumption share policy."""
        a_new = a_policy(m_grid)
        return np.max(np.abs(a_old - a_new)) < self.conv, a_new

    # ------------------------------------------------------------------
    # Time iteration (TI)
    # ------------------------------------------------------------------

    def TI(self, model: PlanningProblem) -> PchipInterpolator:
        """Compute optimal consumption share policy by time iteration."""
        # --- Setup ----------------------------------------------------
        m_grid, a_new, a_policy = self._init_policy(model)

        # --- Iteration ------------------------------------------------
        it = 0
        while True:
            it += 1
            a_old = a_new.copy()

            # Pointwise policy improvement: root-finding on the Euler residual
            for i, m in enumerate(m_grid):

                def euler_root(a_candidate):
                    c_t = model.consumption_from_share(a_candidate, m)
                    a_post = m - c_t
                    m_next_nodes = model.m_next_from(a_post, model.theta_nodes, model.psi_nodes)  # (J,)
                    a_next_nodes = a_policy(m_next_nodes)                                         # (J,)
                    c_next_nodes = model.consumption_from_share(a_next_nodes, m_next_nodes)       # (J,)
                    return model.euler_residual(c_t, c_next_nodes)

                a_lo, a_hi = self.min_share, 1.0 - self.min_share
                f_lo, f_hi = euler_root(a_lo), euler_root(a_hi)

                if f_lo * f_hi < 0.0:
                    # Sign change: use brentq root solver
                    a_new[i] = brentq(euler_root, a_lo, a_hi, xtol=self.tol)
                else:
                    # No sign change: choose endpoint with smaller residual magnitude
                    a_new[i] = a_lo if abs(f_lo) < abs(f_hi) else a_hi

            # Update policy
            a_policy = model.make_spline(m_grid, a_new)

            # --- Convergence check -----------------------------------
            converged, a_new = self._check_convergence(a_old, a_policy, m_grid)
            if converged:
                return a_policy

    # ------------------------------------------------------------------
    # Endogenous Grid Method (EGM)
    # ------------------------------------------------------------------

    def EGM(self, model: PlanningProblem) -> PchipInterpolator:
        """Compute optimal consumption share policy by endogenous grid method."""
        # --- Setup ----------------------------------------------------
        m_grid, a_new, a_policy = self._init_policy(model)

        # Use post-decision asset grid including zero (captures corner a=1 region)
        a_post_grid = np.concatenate(([0.0], m_grid))

        it = 0
        while True:
            it += 1
            a_old = a_new.copy()

            # Storage for endogenous (m_t, a_t)
            m_t_vec = np.empty_like(a_post_grid)
            a_t_vec = np.empty_like(a_post_grid)

            for i, a_post in enumerate(a_post_grid):
                # 1) continuation at m' nodes from a_post
                m_next_nodes = model.m_next_from(a_post, model.theta_nodes, model.psi_nodes)  # (J,)
                a_next_nodes = a_policy(m_next_nodes)                                         # (J,)
                c_next_nodes = model.consumption_from_share(a_next_nodes, m_next_nodes)       # (J,)

                # 2) current c from Euler inversion (scalar for this a_post)
                c_t = model.c_from_euler(c_next_nodes)

                # 3) map back to current m via budget: m_t = c_t + a_post
                m_t = c_t + a_post

                # 4) current share a_t = c_t / m_t
                a_t = c_t / m_t

                m_t_vec[i] = m_t
                a_t_vec[i] = a_t

            # 5) Interpolate a_t(m_t) onto the exogenous m_grid, with a(0)=1 anchor
            sort_idx = np.argsort(m_t_vec)
            m_knots = np.concatenate(([0.0], m_t_vec[sort_idx]))
            a_knots = np.concatenate(([1.0], a_t_vec[sort_idx]))
            a_policy = model.make_spline(m_knots, a_knots)

            # --- Convergence check -----------------------------------
            converged, a_new = self._check_convergence(a_old, a_policy, m_grid)
            if converged:
                return a_policy


if __name__ == "__main__":

    # Model and solvers
    model   = PlanningProblem()
    solvers = PlanningSolvers(m_points=1000,  # number of grid points for state space (resources m)
                              tol=1e-5,       # optimization tolerance (root finding in time iteration)
                              conv=1e-5)      # convergence tolerance (consumption share policy)

    # Solve with timing
    def timed_solve(method_name, solver, model):
        t0 = time()
        policy = solver(model)
        t1 = time()
        print(f'{method_name} converged in {t1 - t0:.5f} seconds.')
        return policy

    # Policies as spline objects
    policy_ti  = timed_solve('TI',  solvers.TI,  model)
    policy_egm = timed_solve('EGM', solvers.EGM, model)

    # ------------------------------------------------------------------
    # Plot 1: Optimal consumption share policies a(m)
    # ------------------------------------------------------------------

    grid = model.make_m_grid(points=solvers.m_points)
    a_ti, a_egm = (p(grid) for p in (policy_ti, policy_egm))

    plt.figure()
    plt.plot(grid, a_ti,  linewidth=5, label="TI policy a(m)")
    plt.plot(grid, a_egm, linewidth=2, label="EGM policy a(m)")
    plt.xlabel("Resources m")
    plt.ylabel("Consumption share a(m)")
    plt.title("Buffer–Stock: Optimal Consumption Share Policies (TI, EGM)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    euler_errors_ti,  states_ti,  U_ti   = model.get_euler_errors(grid, policy_ti,  episodes=100)
    euler_errors_egm, states_egm, U_egm  = model.get_euler_errors(grid, policy_egm, episodes=100)

    ti_err, egm_err = np.max(np.abs(euler_errors_ti)), np.max(np.abs(euler_errors_egm))
    print("Max absolute Euler equation error")
    print(f"TI:  {ti_err:.9f}")
    print(f"EGM: {egm_err:.9f}")

    print("\nAverage discounted episode return")
    print(f"TI:  {U_ti:.9f}")
    print(f"EGM: {U_egm:.9f}")

    # ------------------------------------------------------------------
    # Plot 2: Consumption policies and Euler error histograms
    # ------------------------------------------------------------------

    # Consumption policies on the same grid for reference
    c_ti, c_egm = (model.consumption_from_share(a, grid) for a in (a_ti, a_egm))

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: consumption policies
    ax[0].plot(grid, c_ti,  linewidth=5, label='Time iteration')
    ax[0].plot(grid, c_egm, linewidth=2, label='EGM')
    ax[0].set_title('Policy function (consumption)')
    ax[0].set_xlabel('Resources m')
    ax[0].set_ylabel('Consumption')

    # Right: histograms of log10 absolute Euler errors, restricted to the CI
    def log_errors_ci(errors, width=0.95):
        logs = np.log10(np.abs(errors))
        alpha = (1 - width) / 2
        lo, hi = np.quantile(logs, [alpha, 1-alpha])
        return logs[(logs >= lo) & (logs <= hi)]

    # Compute log10 Euler errors with CI
    logs_ti, logs_egm = map(log_errors_ci, (euler_errors_ti, euler_errors_egm))

    ax[1].hist(logs_ti,  density=True, bins=50, alpha=0.5, label='Time iteration')
    ax[1].hist(logs_egm, density=True, bins=50, alpha=0.5, label='EGM')
    ax[1].set_title('Log10 |Euler error|')
    ax[1].set_xlabel('Log10 absolute Euler equation error')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.legend()
    plt.show()