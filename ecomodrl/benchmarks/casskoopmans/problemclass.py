import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


#-------------------------------------------------------------------
# Shared numerics (grid / spline / optimizer)
#-------------------------------------------------------------------

def make_grid(grid_points: int, lower_limit: float, upper_limit: float):
    """Unified grid setup."""
    return np.linspace(lower_limit, upper_limit, grid_points)

def make_spline(grid, y):
    """Wrap spline construction."""
    return CubicSpline(grid, y, bc_type='natural', extrapolate=True)

def optimize_1d(obj, lower, upper, tol=1e-5):
    """
    Lightweight bounded 1-D optimizer for educational clarity.
    Returns (x_star, obj(x_star)).
    """
    res = minimize_scalar(obj, bounds=(lower, upper), method='bounded', options={'xatol': tol})
    return res.x, res.fun


#-------------------------------------------------------------------
# Cass-Koopmans planning problem (Model only: preferences, technology, accounting, Euler helpers)
#-------------------------------------------------------------------

class PlanningProblem():
    def __init__(
        self, 
        sigma: float = 2,     # Coefficient of relative risk aversion
        beta: float = 0.95,   # Discount factor
        delta: float = 0.02,  # Depreciation rate on capital
        alpha: float = 0.33,  # Return to capital per capita
        A: int = 1            # Technology
    ):
        self.sigma, self.beta = sigma, beta                 # Preferences
        self.delta, self.alpha, self.A = delta, alpha, A    # Production


    # ------------------------------------------------------------------
    # Model primitives (preferences & technology)
    # ------------------------------------------------------------------

    def u(self, c):
        """CRRA utility function"""
        return c ** (1 - self.sigma) / (1 - self.sigma) if self.sigma != 1 else np.log(c)

    def u_prime(self, c):
        """Derivative of utility"""
        return c ** (-self.sigma)

    def u_prime_inv(self, c):
        """Inverse of derivative of utility"""
        return c ** (-1 / self.sigma)

    def f(self, k):
        """Production function"""
        return self.A * k ** self.alpha

    def f_prime(self, k):
        """Derivative of production function"""
        return self.alpha * self.A * k ** (self.alpha - 1)

    def f_prime_inv(self, mpk): # SHOOTING ALGORITHM (REMOVE IF NOT NEEDED)
        """Inverse of f'(k)=alpha*A*k^(alpha-1). Returns k given mpk>0."""
        return (mpk / (self.alpha * self.A)) ** (1.0 / (self.alpha - 1.0))


    # ------------------------------------------------------------------
    # State transitions / resource accounting
    # ------------------------------------------------------------------

    def cash_on_hand(self, k):
        """Compute current cash-on-hand (available resources) given capital stock k_t"""
        return self.f(k) + (1 - self.delta) * k

    def next_k(self, k, c):
        """Compute next-period capital stock k_{t+1} given current capital k_t and feasible consumption c_t"""
        return self.f(k) + (1 - self.delta) * k - c

    def next_k_c(self, k, c): # SHOOTING ALGORITHM (REMOVE IF NOT NEEDED)
        """Compute (k_{t+1}, c_{t+1}) given current capital k_t and feasible consumption c_t using the Euler equation"""
        k_next = self.next_k(k, c)
        c_next = self.u_prime_inv(self.u_prime(c) / (self.beta * (self.f_prime(k_next) + (1 - self.delta))))
        return k_next, c_next

    def consumption_from_share(self, a, k):
        """Reusable helper: c(k) from share a(k) using m(k)=f(k)+(1-δ)k."""
        return a * self.cash_on_hand(k)


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
        return self.beta * (c_t / c_next) ** self.sigma * (self.f_prime(k_next) + (1 - self.delta)) - 1.0

    def calculate_euler_errors(self, c_path, k_path):
        # SHOOTING ALGORITHM (REMOVE IF NOT NEEDED AND MERGE INTO get_euler_errors() BECAUSE NOT USED OTHERWISE)
        """Euler equation errors along given consumption and capital paths."""
        c, c_next = c_path[:-1], c_path[1:]
        k_next = k_path[1:-1] # state path is longer because k_0 exists
        return self.euler_residual(c, c_next, k_next)

    def get_euler_errors(self, grid, policy, n_eps=100):
        """
        Given a grid of capital values and a consumption-share policy a(k), computes Euler equation errors
        over n_eps periods for each initial capital value on the grid. Returns flattened arrays.
        """
        grid_len = len(grid)
        states = np.empty((grid_len, n_eps+1), dtype=float)
        euler_errors = np.empty((grid_len, n_eps-1), dtype=float)
        avg_return = 0.0

        for i, k in enumerate(grid):
            state_path = np.empty(n_eps+1, dtype=float)
            state_path[0] = k
            cons_path = np.empty(n_eps, dtype=float)
            U = 0.0
            power = 1.0

            for t in range(n_eps):
                # a_t = policy(k_t), c_t = a_t * m(k_t)
                a_t = policy(state_path[t])
                cons_path[t] = self.cash_on_hand(state_path[t]) * a_t
                state_path[t+1] = self.next_k(state_path[t], cons_path[t])
                U += power * self.u(cons_path[t])
                power *= self.beta

            states[i, :] = state_path
            euler_errors[i, :] = self.calculate_euler_errors(cons_path, state_path)
            avg_return += U / grid_len

        return euler_errors.ravel(), states.ravel(), avg_return
    
    def euler_egm(self, c_next, k_next):
        """Compute current optimal consumption c_t from Euler equation given c_{t+1} and k_{t+1}."""
        return self.u_prime_inv(self.beta * self.u_prime(c_next) * (self.f_prime(k_next) + (1 - self.delta)))


    #-------------------------------------------------------------------
    # Policy evaluation: value function from share policy
    #-------------------------------------------------------------------

    def V_from_a(self, grid, a_policy, conv):
        """
        Computes value function on grid by iterating the Bellman equation under a fixed share policy.

        grid: grid on which value function is computed
        a_policy: consumption-share policy a(k), callable (e.g. CubicSpline)
        conv: convergence criterion (max-norm)

        return: CubicSpline V(k)
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
    # Value function iteration (VFI)
    # ------------------------------------------------------------------

    def VFI(self, model: PlanningProblem) -> CubicSpline:

        """Compute optimal consumption share policy by value function iteration."""
        
        # --- Setup -------------------------------------------------------
        grid = make_grid(self.grid_points, self.lower_limit, self.upper_limit)  # unified grid
        V = make_spline(grid, np.zeros(self.grid_points))                       # V_0(k) = 0
        V_grid = V(grid)
        a_new = 0.2 * np.ones_like(grid)                                        # initial policy a(k) ≡ 0.2

        # --- Iteration ---------------------------------------------------
        it = 0
        while True:
            it += 1
            a_old = a_new.copy()

            # Pointwise policy improvement (maximize Bellman via minimizing its negative)
            for i, k in enumerate(grid):
                lower, upper = self.min_share, 1.0 - self.min_share

                # Negative of Bellman for minimization
                objective = lambda a, k=k, V=V: -model.bellman(a, k, V)

                a_star, neg_val = optimize_1d(objective, lower, upper, tol=self.tol)
                a_new[i] = a_star
                V_grid[i] = -neg_val

            # Interpolate consumption share policy
            a_policy = make_spline(grid, a_new)

            # --- Convergence check ---------------------------------------
            a_new = a_policy(grid)
            if np.max(np.abs(a_old - a_new)) < self.conv:
                return a_policy

            # Value update
            V = make_spline(grid, V_grid)


    # ------------------------------------------------------------------
    # Time iteration (TI)
    # ------------------------------------------------------------------

    def TI(self, model: PlanningProblem) -> CubicSpline:
        
        """Compute optimal consumption share policy by time iteration."""
        
        # --- Setup -------------------------------------------------------
        grid = make_grid(self.grid_points, self.lower_limit, self.upper_limit)
        a_policy = make_spline(grid, 0.2 * np.ones_like(grid))  # initial policy a(k) ≡ 0.2
        a_new = a_policy(grid)
        sq_err = np.empty(self.grid_points)

        # --- Iteration ---------------------------------------------------
        it = 0
        while True:
            it += 1
            a_old = a_new.copy()

            # Pointwise policy improvement: minimize squared Euler error in share space
            for i, k in enumerate(grid):
                lower, upper = self.min_share, 1.0 - self.min_share

                def objective(a_candidate):
                    # Current-period consumption from share
                    c_t = model.consumption_from_share(a_candidate, k)
                    # Next state and next-period consumption from next-period share
                    k_next = model.next_k(k, c_t)
                    a_next = a_policy(k_next)
                    c_next = model.consumption_from_share(a_next, k_next)
                    r = model.euler_residual(c_t, c_next, k_next)
                    return r ** 2

                a_new[i], sq_err[i] = optimize_1d(objective, lower, upper, tol=self.tol)

            # Interpolate consumption share policy
            a_policy = make_spline(grid, a_new)

            # --- Convergence check ---------------------------------------
            a_new = a_policy(grid)
            if np.max(np.abs(a_old - a_new)) < self.conv:
                return a_policy



    # ------------------------------------------------------------------
    # Endogenous Grid Method (EGM)
    # ------------------------------------------------------------------

    def EGM(self, model: PlanningProblem) -> CubicSpline:

        """Compute optimal consumption share policy by endogenous grid method."""

        # --- Setup -------------------------------------------------------
        grid = make_grid(self.grid_points, self.lower_limit, self.upper_limit)
        m_grid = model.cash_on_hand(grid)
        k_of_m = make_spline(m_grid, grid)

        # policy as spline; arrays as *_new / *_old
        a_policy = make_spline(grid, 0.2 * np.ones_like(grid))
        a_new = a_policy(grid)

        k_next = grid.copy()
        it = 0
        while True:
            it += 1
            a_old = a_new.copy()

            # 1) continuation at k'
            a_next = a_policy(k_next)
            m_next = model.cash_on_hand(k_next)
            c_next = a_next * m_next

            # 2) current c from Euler at k'
            c_t = model.euler_egm(c_next, k_next) # c_t(k')

            # 3) map back to current k via inverse m(k)
            m_t = c_t + k_next # budget: m_t = c_t + k_{t+1}
            k_t = k_of_m(m_t)  # invert to current k_t

            # 4) current share a_t = c_t / m_t
            a_t = c_t / m_t

            # 5) # Interpolate consumption share policy
            a_policy = make_spline(k_t, a_t)

            # --- Convergence check ---------------------------------------
            a_new = a_policy(grid)
            if np.max(np.abs(a_old - a_new)) < self.conv:
                return a_policy