import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

#-------------------------------------------------------------------
# Cass-Koopmans planning problem
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
        sigma = self.sigma
        return c ** (1 - sigma) / (1 - sigma) if sigma != 1 else np.log(c)

    def u_prime(self, c):
        """Derivative of utility"""
        sigma = self.sigma
        return c ** (-sigma)

    def u_prime_inv(self, c):
        """Inverse of derivative of utility"""
        sigma = self.sigma
        return c ** (-1 / sigma)

    def f(self, k):
        """Production function"""
        alpha, A = self.alpha, self.A
        return A * k ** alpha

    def f_prime(self, k):
        """Derivative of production function"""
        alpha, A = self.alpha, self.A
        return alpha * A * k ** (alpha - 1)

    def f_prime_inv(self, mpk):
        """Inverse of f'(k)=alpha*A*k^(alpha-1). Returns k given mpk>0."""
        alpha, A = self.alpha, self.A
        return (mpk / (alpha * A)) ** (1.0 / (alpha - 1.0))


    # ------------------------------------------------------------------
    # State transitions / resource accounting
    # ------------------------------------------------------------------

    def cash_on_hand(self, k):
        """Compute current cash-on-hand (available resources) given capital stock k_t"""
        f, delta = self.f, self.delta
        return f(k) + (1 - delta) * k

    def next_k(self, k, c):
        """Compute next-period capital stock k_{t+1} given current capital k_t and feasible consumption c_t"""
        f, delta = self.f, self.delta
        return f(k) + (1 - delta) * k - c

    def next_k_c(self, k, c):
        """Compute (k_{t+1}, c_{t+1}) given current capital k_t and feasible consumption c_t using the Euler equation"""
        beta, delta = self.beta, self.delta
        u_prime, u_prime_inv = self.u_prime, self.u_prime_inv
        f_prime = self.f_prime
        k_next = self.next_k(k, c)
        c_next = u_prime_inv(u_prime(c) / (beta * (f_prime(k_next) + (1 - delta))))
        return k_next, c_next
    
    def share_to_total(self, a, m):
        """Map consumption share a(k) to total consumption c(k) using resources m(k)."""
        return a * m
    
    def total_to_share(self, c, m):
        """Map total consumption c(k) to consumption share a(k) using resources m(k)."""
        return c / m

    # ------------------------------------------------------------------
    # Bellman functionality
    # ------------------------------------------------------------------

    def bellman(self, a, k, value):
        """Bellman equation: evaluate value function for consumption share a_t given capital k_t"""
        cash_on_hand, next_k, share_to_total = self.cash_on_hand, self.next_k, self.share_to_total
        u, beta = self.u, self.beta
        c = share_to_total(a, cash_on_hand(k))
        k_next = next_k(k, c)
        return u(c) + beta * value(k_next)

    # ------------------------------------------------------------------
    # Euler functionality
    # ------------------------------------------------------------------

    def _euler_residual(self, c_t, c_next, k_next):
        """Core Euler residual. Works with scalars or numpy arrays."""
        beta, sigma, delta = self.beta, self.sigma, self.delta
        f_prime = self.f_prime
        return beta * (c_t / c_next) ** sigma * (f_prime(k_next) + (1 - delta)) - 1.0

    def euler_error(self, c, k, policy):
        """Euler equation error for given consumption c_t and capital k_t, using a policy for c_{t+1}."""
        k_next = self.next_k(k, c)
        c_next = policy(k_next)
        return self._euler_residual(c, c_next, k_next)

    def sq_euler_error(self, c, k, policy):
        """Squared Euler error for given consumption c_t and capital k_t."""
        return self.euler_error(c, k, policy) ** 2

    def calculate_euler_errors(self, c_path, k_path):
        """Euler equation errors along given consumption and capital paths."""
        c, c_next = c_path[:-1], c_path[1:]
        k_next = k_path[1:]
        return self._euler_residual(c, c_next, k_next)

    def euler_egm(self, c_next, k_next):
        """Compute current optimal consumption c_t from Euler equation given c_{t+1} and k_{t+1}."""
        beta, delta = self.beta, self.delta
        u_prime, u_prime_inv = self.u_prime, self.u_prime_inv
        f_prime = self.f_prime
        return u_prime_inv(beta * u_prime(c_next) * (f_prime(k_next) + (1 - delta)))


    # ------------------------------------------------------------------
    # Value function iteration
    # ------------------------------------------------------------------

    def VFI(
        self,
        grid_points: int = 1000,
        lower_limit: float = 0.1,
        upper_limit: float = 15.0,
        tol: float = 1e-5,
        conv: float = 1e-5,
        verbose: bool = False
    ) -> CubicSpline:
        
        'Compute optimal consumption policy by value function iteration (returns optimal consumption-share policy)'
        
        # --- Setup -------------------------------------------------------
        # Small positive floor to keep the share in (0,1) and avoid infeasible points
        MIN_SHARE = 1e-6
        # Capital grid k_t over which we solve the fixed point
        grid = np.linspace(lower_limit, upper_limit, grid_points)
        # Initial value function V_0(k) = 0
        V = CubicSpline(grid, np.zeros(grid_points), bc_type='natural', extrapolate=True)
        # Working array for the pointwise-updated shares a_t(k); seed with constant 20%
        a_new = 0.2 * np.ones_like(grid)
        # Empty buffer for updated value function
        V_new = np.empty(grid_points)

        # --- Iteration ---------------------------------------------------
        it = 0
        while True:
            it += 1
            a_old = a_new.copy() # Keep previous iterate to measure policy movement (sup norm)
            V_old = V(grid) # Previous value function evaluation

            # Pointwise policy improvement: minimize Bellman residual at each grid node
            for i, k in enumerate(grid):
                # Feasible bracket for consumption share a ∈ [eps, 1-eps]
                lower, upper = MIN_SHARE, 1.0 - MIN_SHARE

                res = minimize(
                    -self.bellman, # Objective: negative Bellman value at (k, a)
                    x0=np.clip(a_new[i], lower, upper), # Robust starting value
                    bounds=((lower, upper),), # Enforce feasibility directly in the optimizer
                    args=(k, V), # k is the state; V provides the continuation value
                    method='Powell',
                    tol=tol,
                )

                # Updated pointwise optimum a*(k)
                a_new[i] = float(res.x[0])
                # Updated value function value at grid point
                V_new[i] = -float(res.fun)

            # --- Convergence check ---------------------------------------
            delta_policy = np.max(np.abs(a_old - a_new))  # ∥a^{new} - a^{old}∥
            if verbose:
                delta_value = np.max(np.abs(V_old - V_new))  # max-norm difference of value functions
                print(
                    f'Iter {it:4d} | Δa = {delta_policy:.3e} | '
                    f'ΔV = {delta_value:.3e} | target = {conv:.3e}'
                )

            if delta_policy < conv:
                # Converged: return the share policy spline a(k)
                return CubicSpline(grid, a_new, bc_type='natural', extrapolate=True)

            # Policy iteration step: update continuation value function for next sweep
            V = CubicSpline(grid, V_new, bc_type='natural', extrapolate=True)



    # ------------------------------------------------------------------
    # Time iteration
    # ------------------------------------------------------------------

    def TI(
        self,
        grid_points: int = 1000,
        lower_limit: float = 0.1,
        upper_limit: float = 15.0,
        tol: float = 1e-5,
        conv: float = 1e-5,
        verbose: bool = False
    ) -> CubicSpline:
        
        'Compute optimal consumption policy by time iteration (returns optimal consumption-share policy)'
        
        # --- Setup -------------------------------------------------------
        # Small positive floor to avoid numerical issues
        MIN_CONS = 1e-6
        # Capital grid k_t over which we solve the fixed point
        grid = np.linspace(lower_limit, upper_limit, grid_points)
        # Resources m_t(k) = f(k) + (1-delta)k available before choosing c_t
        m = self.cash_on_hand(grid)
        # Initial policy guess: consume a constant 20% of available resources
        c_policy = CubicSpline(grid, 0.2 * m, bc_type='natural', extrapolate=True) 
        # Working array for the pointwise-updated consumption levels c_t(k)
        c_new = c_policy(grid)
        # Empty array to record squared Euler errors for diagnostics
        sq_err = np.empty(grid_points)

        # --- Iteration ---------------------------------------------------
        it = 0
        while True:
            it += 1
            c_old = c_new.copy() # Keep previous iterate to measure policy movement (sup norm)

            # Pointwise policy improvement: minimize squared Euler error at each grid node
            for i, k in enumerate(grid):
                upper = max(MIN_CONS, m[i] - MIN_CONS)  # Feasible bracket: c ∈ [ε, m(k)-ε] ensures k_{t+1} ≥ 0 and c>0

                res = minimize(
                    self.sq_euler_error, # Objective: squared Euler residual at (k, c), given current policy as continuation
                    x0=np.clip(c_new[i], MIN_CONS, upper), # Robust starting value: clamp previous iterate to the feasible bracket
                    bounds=((MIN_CONS, upper),), # Enforce feasibility directly in the optimizer
                    args=(k, c_policy), # k is the state; c_policy provides c_{t+1} via policy(k_{t+1})
                    method='Powell',
                    tol=tol,
                )

                # Updated pointwise optimum c*(k)
                c_new[i] = float(res.x[0])
                # Store residual for logging (max |Euler| = sqrt(max sq_err))              
                sq_err[i] = float(res.fun)

            # --- Convergence check -----------
            a_old = self.total_to_share(c_old, m) # previous share policy
            a_new = self.total_to_share(c_new, m) # updated share policy
            delta_policy = np.max(np.abs(a_old - a_new)) # ∥a^{new} - a^{old}∥

            if verbose:
                abs_euler = np.sqrt(np.max(sq_err)) # worst-case absolute Euler error
                print(
                    f'Iter {it:4d} | deltaa = {delta_policy:.3e} | '
                    f'max |Euler| = {abs_euler:.3e} | target = {conv:.3e}'
                )

            if delta_policy < conv:
                # Converged: return the share policy spline a(k)
                return CubicSpline(grid, a_new, bc_type='natural', extrapolate=True)

            # Policy iteration step: use the updated total consumption policy for continuation
            c_policy = CubicSpline(grid, c_new, bc_type='natural', extrapolate=True)



        

    #-------------------------------------------------------------------
    # EGM
    #-------------------------------------------------------------------

    def EGM(self, grid_points:int = 1000, lower_limit:float=0.1, upper_limit:float=15., conv:float=0.1, verbose:bool=True):
        """
        Computes optimal consumption policy by endogenous grid method (EGM)
        
        grid_points: number of grid points
        lower_limit: lower limit of grid
        upper_limit: upper limit of grid
        conv: convergence criterion
        verbose: if True, prints convergence information
        
        return: grid and optimal consumption policy as CubicSpline object
        """
        cash_on_hand = self.cash_on_hand
        euler_egm = self.euler_egm
        
        k_next_grid = np.linspace(lower_limit, upper_limit, grid_points)
        sopt = CubicSpline(np.concatenate(([0.0], k_next_grid)), np.concatenate(([0.0], k_next_grid*0.2)), bc_type="natural", extrapolate=True)
        ms = np.empty(grid_points)
        while True:
            for i, k_next in enumerate(k_next_grid):
                m_next = cash_on_hand(k_next)
                c_next = m_next - sopt(m_next)
                c = euler_egm(c_next, k_next)
                ms[i] = c + k_next
            if verbose:
                print(
                    "Convergence: {}.".format(max(abs(sopt(ms)-k_next_grid))),
                    "Convergence criterion: {}.".format(conv),
                )
            if max(abs(sopt(ms)-k_next_grid)) < conv:
                c_of_m = CubicSpline(np.concatenate(([0.0], ms)), np.concatenate(([0.0], ms-k_next_grid)), bc_type="natural", extrapolate=True)
                m_grid = cash_on_hand(k_next_grid)
                copt = CubicSpline(k_next_grid, c_of_m(m_grid)/m_grid, bc_type="natural", extrapolate=True) 
                return k_next_grid, copt 
            sopt = CubicSpline(np.concatenate(([0.0], ms)), np.concatenate(([0.0], k_next_grid)), bc_type="natural", extrapolate=True)
        

    #-------------------------------------------------------------------
    # Simulation
    #-------------------------------------------------------------------
    def simulation(self, copt, periods=100, start_value=5.0):
        """
        Simulate consumption path over number of periods
        
        copt: consumption policy as CubicSpline object
        periods: number of periods to simulate
        start_value: starting value of capital
        """
        next_k, cash_on_hand = self.next_k, self.cash_on_hand
        u = self.u
        beta = self.beta
        
        state = np.empty(periods+1)
        state[0] = start_value
        cons = np.empty(periods)
        U = 0
        power = 1.0
        for t in range(periods):
            action = copt(state[t])
            cons[t] = cash_on_hand(state[t])*action
            state[t+1] = next_k(state[t], cons[t])
            U += power * u(cons[t])
            power *= beta
        return cons, state, U
    
    def get_vfrom_copt(self, grid, copt, conv):
        """
        Computes value function on grid by iterating on Bellman equation
        
        grid: grid on which value function is computed
        copt: optimal consumption policy as CubicSpline object
        conv: convergence criterion
        
        return: value function as CubicSpline object
        """
        bellman = self.bellman
        Vn = np.zeros(len(grid))
        V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
        while True:
            for i in range(len(grid)):
                Vn[i] = bellman(copt(grid[i]), grid[i], V)
            if max(abs(Vn - V(grid))) < conv:
                break
            V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
        return CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)

    #-------------------------------------------------------------------
    # Euler equation errors
    #-------------------------------------------------------------------
    def get_euler_errors(self, grid, policy, n_eps=100):
        '''
        Given a grid of capital values and a consumption policy, computes Euler equation errors
        over n_eps periods for each initial capital value in the grid.
        '''
        calculate_euler_errors = self.calculate_euler_errors
        grid_len = len(grid)
        states = np.empty((grid_len, n_eps+1), dtype=float)
        euler_errors = np.empty((grid_len, n_eps-1), dtype=float)
        avg_return = 0
        for i, k in enumerate(grid):
            cons_path, state_path, U = self.simulation(policy, n_eps, start_value=k)
            states[i,:] = state_path
            euler_errors[i,:] = calculate_euler_errors(cons_path, state_path)
            avg_return += U/grid_len
        return euler_errors.ravel(), states.ravel(), avg_return