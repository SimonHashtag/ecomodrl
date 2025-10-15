import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar


#-------------------------------------------------------------------
# Shared numerics (grid / spline / optimizer)
#-------------------------------------------------------------------

def make_grid(grid_points: int, lower_limit: float, upper_limit: float):
    """Unified grid setup."""
    return np.linspace(lower_limit, upper_limit, grid_points)

def make_spline(grid, y, extrapolate=True):
    """Wrap spline construction (PCHIP: monotone, shape-preserving)."""
    return PchipInterpolator(grid, y, extrapolate=extrapolate)

def optimize_1d(obj, lower, upper, tol=1e-5):
    """Lightweight bounded 1-D optimizer. Returns (x_star, obj(x_star))."""
    res = minimize_scalar(obj, bounds=(lower, upper), method='bounded', options={'xatol': tol})
    return res.x, res.fun


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
                # a_t = policy(k_t), c_t = a_t * m(k_t)
                a_t = policy(state_path[t])
                cons_path[t] = self.cash_on_hand(state_path[t]) * a_t
                state_path[t+1] = self.next_k(state_path[t], cons_path[t])
                U += power * self.u(cons_path[t])
                power *= self.beta

            states[i, :] = state_path
            c, c_next = cons_path[:-1], cons_path[1:]
            k_next = state_path[1:-1] # state path is longer because k_0 exists
            euler_errors[i, :] = self.euler_residual(c, c_next, k_next)
            avg_return += U / grid_len

        return euler_errors.ravel(), states.ravel(), avg_return


    #-------------------------------------------------------------------
    # Policy evaluation: value function from share policy
    #-------------------------------------------------------------------

    def V_from_a(self, grid, a_policy, conv):
        """
        Computes value function on grid by iterating the Bellman equation under a fixed share policy.

        grid: grid on which value function is computed
        a_policy: consumption-share policy a(k), callable (e.g. PchipInterpolator)
        conv: convergence criterion (max-norm)

        return: PchipInterpolator V(k)
        """
        Vn = np.zeros(len(grid))
        V = make_spline(grid, Vn)
        while True:
            for i in range(len(grid)):
                a_i = a_policy(grid[i])
                Vn[i] = self.bellman(a_i, grid[i], V)
            if np.max(np.abs(Vn - V(grid))) < conv:
                break
            V = make_spline(grid, Vn)
        return make_spline(grid, Vn)


#-------------------------------------------------------------------
# Solvers: standardized on share policy a(k)
#-------------------------------------------------------------------

class PlanningSolvers:
    def __init__(
        self,
        grid_points: int = 1000,    # number of capital grid points
        lower_limit: float = 0.1,   # k_min
        upper_limit: float = 15.0,  # k_max
        tol: float = 1e-5,          # optimization tolerance
        conv: float = 1e-5,         # convergence tolerance (consumption share policy)
        min_share: float = 1e-6     # feasibility guard for shares
    ):
        self.grid_points = grid_points
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.tol = tol
        self.conv = conv
        self.min_share = min_share

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _init_policy(self):
        """Initialize grid, consumption share array and spline policy."""
        grid = make_grid(self.grid_points, self.lower_limit, self.upper_limit)
        a_new = np.full_like(grid, 0.2) # start with 20% consumption share policy
        a_policy = make_spline(grid, a_new)
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
        grid, a_new, a_policy = self._init_policy()
        sq_err = np.empty(self.grid_points)

        # --- Iteration ---------------------------------------------------
        it = 0
        while True:
            it += 1
            a_old = a_new.copy()

            # Pointwise policy improvement: minimize squared Euler error in share space
            for i, k in enumerate(grid):

                def objective(a_candidate):
                    c_t = model.consumption_from_share(a_candidate, k)
                    k_next = model.next_k(k, c_t)
                    a_next = a_policy(k_next)
                    c_next = model.consumption_from_share(a_next, k_next)
                    r = model.euler_residual(c_t, c_next, k_next)
                    return r ** 2

                a_new[i], sq_err[i] = optimize_1d(objective, 
                                                  lower=self.min_share, upper=1.0-self.min_share, 
                                                  tol=self.tol)

            a_policy = make_spline(grid, a_new)

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
        grid, a_new, a_policy = self._init_policy()
        m_grid = model.cash_on_hand(grid)
        k_of_m = make_spline(m_grid, grid)

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
            a_policy = make_spline(k_t[sort_idx], a_t[sort_idx])

            # --- Convergence check ---------------------------------------
            converged, a_new = self._check_convergence(a_old, a_policy, grid)
            if converged:
                return a_policy