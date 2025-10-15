import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

#-------------------------------------------------------------------
# Cass-Koopmans planning problem (preferences, technology, accounting, Euler helpers)
#-------------------------------------------------------------------

class PlanningProblem():
    def __init__(
        self, 
        gamma: float = 2,     # Coefficient of relative risk aversion
        beta: float = 0.95,   # Discount factor
        delta: float = 0.02,  # Depreciation rate on capital
        alpha: float = 0.33,  # Return to capital per capita
        A: int = 1            # Technology
    ):
        self.gamma, self.beta = gamma, beta                 # Preferences
        self.delta, self.alpha, self.A = delta, alpha, A    # Production


    # ------------------------------------------------------------------
    # Steady state, grid setup, and spline construction
    # ------------------------------------------------------------------

    @property
    def k_ss(self) -> float:
        numerator = self.alpha * self.A
        denominator = 1.0 / self.beta - (1.0 - self.delta)
        return (numerator / denominator) ** (1.0 / (1.0 - self.alpha))

    def make_grid(self, points: int = 1000, span=(0.2, 5.0)):
        k_min, k_max = (s * self.k_ss for s in span)
        return np.linspace(k_min, k_max, points)

    def make_spline(self, grid, y, extrapolate=True):
        """PCHIP spline (monotone, shape-preserving)."""
        return PchipInterpolator(grid, y, extrapolate=extrapolate)


    # ------------------------------------------------------------------
    # Model primitives (preferences & technology)
    # ------------------------------------------------------------------

    def u(self, c):
        """CRRA utility function"""
        return c ** (1 - self.gamma) / (1 - self.gamma) if self.gamma != 1 else np.log(c)

    def u_prime(self, c):
        """Derivative of utility"""
        return c ** (-self.gamma)

    def u_prime_inv(self, c):
        """Inverse of derivative of utility"""
        return c ** (-1 / self.gamma)

    def f(self, k):
        """Production function"""
        return self.A * k ** self.alpha

    def f_prime(self, k):
        """Derivative of production function"""
        return self.alpha * self.A * k ** (self.alpha - 1)


    # ------------------------------------------------------------------
    # State transitions / resource accounting
    # ------------------------------------------------------------------

    def cash_on_hand(self, k):
        """Compute current cash-on-hand (available resources) given capital stock k_t"""
        return self.f(k) + (1 - self.delta) * k

    def consumption_from_share(self, a, k):
        """Reusable helper: c(k) from share a(k) using m(k)=f(k)+(1-δ)k."""
        return a * self.cash_on_hand(k)

    def next_k(self, k, c):
        """Compute next-period capital stock k_{t+1} given current capital k_t and feasible consumption c_t"""
        return self.f(k) + (1 - self.delta) * k - c


    # ------------------------------------------------------------------
    # Bellman functionality
    # ------------------------------------------------------------------

    def bellman(self, a, k, value):
        """Bellman equation value at (k, a_t)"""
        c = self.consumption_from_share(a, k)
        k_next = self.next_k(k, c)
        return self.u(c) + self.beta * value(k_next)


    # ------------------------------------------------------------------
    # Euler functionality
    # ------------------------------------------------------------------

    def euler_residual(self, c_t, c_next, k_next):
        """Euler residual."""
        return self.beta * (c_t / c_next) ** self.gamma * (self.f_prime(k_next) + (1 - self.delta)) - 1.0

    def c_from_euler(self, c_next, k_next):
        """Compute current optimal consumption c_t from Euler equation given c_{t+1} and k_{t+1}."""
        return self.u_prime_inv(self.beta * self.u_prime(c_next) * (self.f_prime(k_next) + (1 - self.delta)))

    def get_euler_errors(self, grid, policy, episodes=100):
        """
        Given a grid of capital values and a consumption-share policy a(k), computes Euler equation errors
        over episodes periods for each initial capital value on the grid. Returns flattened arrays.
        """
        grid_len = len(grid)
        states = np.empty((grid_len, episodes+1), dtype=float)
        euler_errors = np.empty((grid_len, episodes-1), dtype=float)
        avg_return = 0.0

        for i, k in enumerate(grid):
            state_path = np.empty(episodes+1, dtype=float)
            state_path[0] = k
            cons_path = np.empty(episodes, dtype=float)
            U = 0.0
            power = 1.0

            for t in range(episodes):
                a_t = policy(state_path[t])
                cons_path[t] = self.cash_on_hand(state_path[t]) * a_t
                state_path[t+1] = self.next_k(state_path[t], cons_path[t])
                U += power * self.u(cons_path[t])
                power *= self.beta

            states[i, :] = state_path
            c, c_next = cons_path[:-1], cons_path[1:]
            k_next = state_path[1:-1]
            euler_errors[i, :] = self.euler_residual(c, c_next, k_next)
            avg_return += U / grid_len

        return euler_errors.ravel(), states.ravel(), avg_return


    #-------------------------------------------------------------------
    # Policy evaluation: value function from policy
    #-------------------------------------------------------------------

    def V_from_a(self, grid, a_policy, conv) -> PchipInterpolator:
        """Computes value function on grid by iterating the Bellman equation under a fixed policy."""
        Vn = np.zeros(len(grid))
        V = self.make_spline(grid, Vn)
        while True:
            for i in range(len(grid)):
                a_i = a_policy(grid[i])
                Vn[i] = self.bellman(a_i, grid[i], V)
            if np.max(np.abs(Vn - V(grid))) < conv:
                break
            V = self.make_spline(grid, Vn)
        return self.make_spline(grid, Vn)


#-------------------------------------------------------------------
# Solvers: standardized on share policy a(k)
#-------------------------------------------------------------------

class PlanningSolvers:
    def __init__(
        self,
        grid_points: int = 1000,    # number of capital grid points
        tol: float = 1e-5,          # optimization tolerance
        conv: float = 1e-5,         # convergence tolerance (consumption share policy)
        min_share: float = 1e-6     # feasibility guard for shares
    ):
        self.grid_points = grid_points
        self.tol = tol
        self.conv = conv
        self.min_share = min_share


    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _init_policy(self, model: PlanningProblem):
        """Initialize grid, consumption share array and spline policy."""
        grid = model.make_grid(points=self.grid_points)
        a_new = np.full_like(grid, 0.2)
        a_policy = model.make_spline(grid, a_new)
        return grid, a_new, a_policy

    def _check_convergence(self, a_old, a_policy, grid):
        """Check convergence of consumption share policy."""
        a_new = a_policy(grid)
        return np.max(np.abs(a_old - a_new)) < self.conv, a_new


    # ------------------------------------------------------------------
    # Time iteration (TI)
    # ------------------------------------------------------------------

    def TI(self, model: PlanningProblem) -> PchipInterpolator:
        """Compute optimal consumption share policy by time iteration."""
        # --- Setup -------------------------------------------------------
        grid, a_new, a_policy = self._init_policy(model)

        # --- Iteration ---------------------------------------------------
        it = 0
        while True:
            it += 1
            a_old = a_new.copy()

            # Pointwise policy improvement: root-finding on the Euler residual
            for i, k in enumerate(grid):

                def euler_root(a_candidate):
                    c_t = model.consumption_from_share(a_candidate, k)
                    k_next = model.next_k(k, c_t)
                    a_next = a_policy(k_next)
                    c_next = model.consumption_from_share(a_next, k_next)
                    return model.euler_residual(c_t, c_next, k_next)

                a_new[i] = brentq(euler_root,
                                  self.min_share, 1.0 - self.min_share,
                                  xtol=self.tol)

            a_policy = model.make_spline(grid, a_new)

            # --- Convergence check ---------------------------------------
            converged, a_new = self._check_convergence(a_old, a_policy, grid)
            if converged:
                return a_policy


    # ------------------------------------------------------------------
    # Endogenous Grid Method (EGM)
    # ------------------------------------------------------------------

    def EGM(self, model: PlanningProblem) -> PchipInterpolator:
        """Compute optimal consumption share policy by endogenous grid method."""
        # --- Setup -------------------------------------------------------
        grid, a_new, a_policy = self._init_policy(model)
        m_grid = model.cash_on_hand(grid)
        k_of_m = model.make_spline(m_grid, grid)

        k_next = grid.copy()
        it = 0
        while True:
            it += 1
            a_old = a_new.copy()

            # 1) continuation at k'
            a_next = a_policy(k_next)
            c_next = model.consumption_from_share(a_next, k_next)

            # 2) current c from Euler at k'
            c_t = model.c_from_euler(c_next, k_next)  # c_t(k')

            # 3) map back to current k via inverse m(k)
            m_t = c_t + k_next  # budget: m_t = c_t + k_{t+1}
            k_t = k_of_m(m_t)   # invert to current k_t

            # 4) current share a_t = c_t / m_t
            a_t = c_t / m_t

            # 5) Interpolate consumption share policy
            sort_idx = np.argsort(k_t)
            a_policy = model.make_spline(k_t[sort_idx], a_t[sort_idx])

            # --- Convergence check ---------------------------------------
            converged, a_new = self._check_convergence(a_old, a_policy, grid)
            if converged:
                return a_policy