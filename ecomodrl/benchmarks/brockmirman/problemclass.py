import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.optimize import brentq

#-------------------------------------------------------------------
# Brock–Mirman planning problem (preferences, technology, accounting, Euler helpers)
#-------------------------------------------------------------------

class PlanningProblem():
    def __init__(
        self, 
        gamma: float = 2,      # Coefficient of relative risk aversion
        beta: float = 0.95,    # Discount factor
        delta: float = 0.02,   # Depreciation rate on capital
        alpha: float = 0.33,   # Return to capital per capita
        A: int = 1,            # Technology level (trend/mean of z is 1)
        rho: float = 0.9,      # Persistence of log productivity
        sigma: float = 0.02,   # Std. dev. of log productivity shock
        gh_order: int = 5      # Gauss–Hermite quadrature order
    ):
        self.gamma, self.beta = gamma, beta                 # Preferences
        self.delta, self.alpha, self.A = delta, alpha, A    # Production
        self.rho, self.sigma = rho, sigma                   # Shock process

        # Gauss–Hermite nodes/weights
        x, w = hermgauss(gh_order)
        self.gh_x = np.sqrt(2.0) * x
        self.gh_w = w / np.sqrt(np.pi)

    # ------------------------------------------------------------------
    # Steady state, grid setup, and interpolation constructors
    # ------------------------------------------------------------------

    @property
    def k_ss(self) -> float:
        numerator = self.alpha * self.A
        denominator = 1.0 / self.beta - (1.0 - self.delta)
        return (numerator / denominator) ** (1.0 / (1.0 - self.alpha))

    def make_k_grid(self, points: int = 1000, span=(0.2, 5.0)):
        k_min, k_max = (s * self.k_ss for s in span)
        return np.linspace(k_min, k_max, points)

    def make_z_grid(self, points: int = 21, m: float = 3.0):
        """Lognormal z grid centered at 1 using ±m stationary std devs of log z."""
        sd = self.sigma / np.sqrt(1 - self.rho**2)
        logz = np.linspace(-m*sd, m*sd, points)
        return np.exp(logz)

    def make_spline(self, grid, y, extrapolate=True):
        """PCHIP spline (1D; used for m<->k inversions inside EGM)."""
        return PchipInterpolator(grid, y, extrapolate=extrapolate)

    def make_bilinear(self, k_grid, z_grid, values):
        """Bilinear interpolation on (k,z)."""
        return RegularGridInterpolator(
            (k_grid, z_grid), values, method='linear',
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

    def u_prime_inv(self, x):
        """Inverse marginal utility"""
        return x ** (-1 / self.gamma)

    def f(self, k, z):
        """Production function"""
        return self.A * z * k ** self.alpha

    def f_prime(self, k, z):
        """Marginal productivity of capital"""
        return self.alpha * self.A * z * k ** (self.alpha - 1)

    # ------------------------------------------------------------------
    # State transitions / resource accounting
    # ------------------------------------------------------------------

    def z_next_from(self, z):
        """Return z' at GH nodes for scalar z (gh,) or array z (n, gh)."""
        z = np.asarray(z)
        base = self.rho * np.log(z)[..., None]
        return np.exp(base + self.sigma * self.gh_x)

    def cash_on_hand(self, k, z):
        """Compute current cash-on-hand (available resources) given capital stock k_t and shock z_t"""
        return self.f(k, z) + (1 - self.delta) * k

    def consumption_from_share(self, a, k, z):
        """Derive total consumption from consumption share."""
        return a * self.cash_on_hand(k, z)

    def next_k(self, k, z, c):
        """Compute next-period capital stock k_{t+1} given k_t, z_t, and feasible consumption c_t"""
        return self.f(k, z) + (1 - self.delta) * k - c

    # ------------------------------------------------------------------
    # Bellman functionality
    # ------------------------------------------------------------------

    def bellman(self, a, k, z, value):
        """Bellman equation value at (k,z) with expectation over z' (GH) for consumption share a_t."""
        c = self.consumption_from_share(a, k, z)
        k_next = self.next_k(k, z, c)
        z_next = self.z_next_from(z)
        pts = np.column_stack([np.repeat(k_next, len(self.gh_w)), z_next.ravel()])
        V_next = value(pts).reshape(z_next.shape)
        EV = np.sum(self.gh_w * V_next, axis=-1)
        return self.u(c) + self.beta * EV

    # ------------------------------------------------------------------
    # Euler functionality
    # ------------------------------------------------------------------

    def euler_residual(self, c_t, c_next, k_next, z_next):
        """Euler residual."""
        c_t = np.asarray(c_t)
        term = (np.expand_dims(c_t, -1) / c_next) ** self.gamma
        R = self.f_prime(k_next[..., None], z_next) + (1 - self.delta)
        return self.beta * np.sum(self.gh_w * term * R, axis=-1) - 1.0

    def c_from_euler(self, c_next, k_next, z_next):
        """
        Solve u'(c_t) = beta * E[ u'(c_{t+1}) * (f'(k_{t+1}, z') + 1-δ) ].
        Returns c_t (vectorized in k_next).
        """
        EmuR = self.beta * np.sum(self.gh_w * self.u_prime(c_next) * (self.f_prime(k_next, z_next) + (1 - self.delta)), axis=-1)
        return self.u_prime_inv(EmuR)

    def get_euler_errors(self, k_grid, z_grid, policy, episodes=100, seed=123):
        """Compute Euler equation errors over simulated paths from every (k,z) grid node. Returns flattened arrays."""
        rng = np.random.default_rng(seed)
        T = episodes # rename for brevity
        
        Nk, Nz = len(k_grid), len(z_grid)
        grid_len = Nk * Nz
        
        k_states = np.empty((grid_len, T+1), dtype=float)
        euler_errors = np.empty((grid_len, T-1), dtype=float)
        avg_return = 0.0
        
        # Prebuild mesh of starting points
        K0, Z0 = np.meshgrid(k_grid, z_grid, indexing='ij')
        start_pairs = np.column_stack([K0.ravel(), Z0.ravel()])

        for idx, (k0, z0) in enumerate(start_pairs):
            k_path = np.empty(T+1)
            z_path = np.empty(T+1)
            c_path = np.empty(T)
            k_path[0] = k0
            z_path[0] = z0
            U = 0.0
            power = 1.0

            for t in range(T):
                a_t = policy([[k_path[t], z_path[t]]])[0]
                c_t = self.consumption_from_share(a_t, k_path[t], z_path[t])
                k_next = self.next_k(k_path[t], z_path[t], c_t)
                # advance shock with a draw (for the path), residual still uses GH expectation at z_t
                eps = rng.standard_normal()
                z_next_realized = np.exp(self.rho * np.log(z_path[t]) + self.sigma * eps)
                k_path[t+1] = k_next
                z_path[t+1] = z_next_realized
                c_path[t] = c_t
                U += power * self.u(c_t)
                power *= self.beta

            # Extract objects (T-1,)
            c_t = c_path[:T-1]
            z_t = z_path[:T-1]
            k_next = k_path[1:T]

            # GH nodes for next-period shocks: z_{t+1}^{(g)} | z_t
            z_next_nodes = self.z_next_from(z_t) # (T-1, gh)

            # Tile k_{t+1} for each GH node
            gh = len(self.gh_w)
            k_next_tile = np.repeat(k_next[:, None], gh, axis=1) # (T-1, gh)

            # Evaluate next-period policy at all (k_{t+1}, z_{t+1}^{(g)})
            pts = np.column_stack([k_next_tile.ravel(), z_next_nodes.ravel()])
            a_next_nodes = policy(pts).reshape(T-1, gh) # (T-1, gh)

            # Convert shares to consumption: c_{t+1}^{(g)}
            cons_next_nodes = self.consumption_from_share(a_next_nodes, k_next_tile, z_next_nodes)

            # Euler residual using expected marginal utility at t+1 (GH integration inside euler_residual)
            euler_errors[idx, :] = self.euler_residual(c_t, cons_next_nodes, k_next, z_next_nodes)

            # Store simulated capital path for this starting state
            k_states[idx, :] = k_path

            # Accumulate average discounted utility over all starting states
            avg_return += U / grid_len

        return euler_errors.ravel(), k_states.ravel(), avg_return


    #-------------------------------------------------------------------
    # Policy evaluation: value function from policy
    #-------------------------------------------------------------------

    def V_from_a(self, k_grid, z_grid, a_policy, conv) -> RegularGridInterpolator:
        """Iterate Bellman under fixed a(k,z) policy using bilinear interpolation."""
        Nk, Nz = len(k_grid), len(z_grid)
        Vn = np.zeros((Nk, Nz))
        V = self.make_bilinear(k_grid, z_grid, Vn)
        # Precompute grid evaluation points for the convergence check
        K, Z = np.meshgrid(k_grid, z_grid, indexing='ij')
        pts = np.column_stack([K.ravel(), Z.ravel()])
        while True:
            # Policy evaluation step: apply Bellman operator under fixed policy a(k,z)
            for i, k in enumerate(k_grid):
                for j, z in enumerate(z_grid):
                    a_ij = a_policy([[k, z]])[0]
                    Vn[i, j] = self.bellman(a_ij, k, z, V)
            # Convergence check on the grid against the previous V
            V_prev_on_grid = V(pts).reshape(Nk, Nz)
            err = np.max(np.abs(Vn - V_prev_on_grid))
            # Update V to match Vn; if converged, break with the final interpolant
            V = self.make_bilinear(k_grid, z_grid, Vn)
            if err < conv:
                break
        return V


#-------------------------------------------------------------------
# Solvers: standardized on share policy a(k,z)
#-------------------------------------------------------------------

class PlanningSolvers:
    def __init__(
        self,
        k_points: int = 400,        # number of capital grid points
        z_points: int = 21,         # number of shock grid points
        tol: float = 1e-5,          # optimization tolerance
        conv: float = 1e-5,         # convergence tolerance (policy)
        min_share: float = 1e-6     # feasibility guard for shares
    ):
        self.k_points = k_points
        self.z_points = z_points
        self.tol = tol
        self.conv = conv
        self.min_share = min_share

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _init_policy(self, model: PlanningProblem):
        """Initialize (k,z) grids, consumption share array and bilinear policy."""
        k_grid = model.make_k_grid(points=self.k_points)
        z_grid = model.make_z_grid(points=self.z_points)
        a_new = np.full((len(k_grid), len(z_grid)), 0.2, dtype=float)
        a_policy = model.make_bilinear(k_grid, z_grid, a_new)
        return k_grid, z_grid, a_new, a_policy

    def _check_convergence(self, a_old, a_policy, k_grid, z_grid):
        """Check convergence of consumption share policy."""
        K, Z = np.meshgrid(k_grid, z_grid, indexing='ij')
        pts = np.column_stack([K.ravel(), Z.ravel()])
        a_new = a_policy(pts).reshape(a_old.shape)
        return np.max(np.abs(a_old - a_new)) < self.conv, a_new


    # ------------------------------------------------------------------
    # Time iteration (TI)
    # ------------------------------------------------------------------

    def TI(self, model: PlanningProblem) -> RegularGridInterpolator:
        """Compute optimal consumption share policy by TI."""
        # --- Setup ----------------------------------------------------
        k_grid, z_grid, a_new, a_policy = self._init_policy(model)

        # --- Iteration ------------------------------------------------
        it = 0
        while True:
            it += 1
            a_old = a_new.copy()

            # Pointwise policy improvement: root-finding on the Euler residual
            for i, k in enumerate(k_grid):
                for j, z in enumerate(z_grid):

                    def euler_root(a_candidate):
                        c_t = model.consumption_from_share(a_candidate, k, z)
                        k_next = model.next_k(k, z, c_t)
                        z_next = model.z_next_from(z).ravel()  # (gh,)
                        pts = np.column_stack([np.full_like(z_next, k_next, dtype=float), z_next])
                        a_next = a_policy(pts)                 # (gh,)
                        c_next = model.consumption_from_share(a_next, k_next, z_next)
                        return model.euler_residual(c_t, c_next, k_next, z_next)

                    a_lo, a_hi = self.min_share, 1.0 - self.min_share
                    f_lo, f_hi = euler_root(a_lo), euler_root(a_hi)

                    if f_lo * f_hi < 0.0:
                        # Sign change: use brentq root solver
                        a_new[i, j] = brentq(euler_root, a_lo, a_hi, xtol=self.tol)
                    else:
                        # No sign change: shift in direction with smaller residual magnitude
                        a_new[i, j] = a_lo if abs(f_lo) < abs(f_hi) else a_hi

            # Update policy
            a_policy = model.make_bilinear(k_grid, z_grid, a_new)

            # --- Convergence check ---------------------------------------
            converged, a_new = self._check_convergence(a_old, a_policy, k_grid, z_grid)
            if converged:
                return a_policy


    # ------------------------------------------------------------------
    # Endogenous Grid Method (EGM)
    # ------------------------------------------------------------------

    def EGM(self, model: PlanningProblem) -> RegularGridInterpolator:
        """Compute optimal consumption share policy by EGM."""
        # --- Setup ----------------------------------------------------
        k_grid, z_grid, a_new, a_policy = self._init_policy(model)

        # For each z, build 1D inverse m->k on the exogenous k grid (used every iteration)
        k_of_m_list = []
        for z in z_grid:
            m_grid = model.cash_on_hand(k_grid, z) # monotone in k
            k_of_m_list.append(model.make_spline(m_grid, k_grid))

        k_next = k_grid.copy()

        # --- Iteration ------------------------------------------------
        it = 0
        while True:
            it += 1
            a_old = a_new.copy()

            for j, z in enumerate(z_grid):
                # 1) continuation at (k', z') for GH nodes
                z_next = model.z_next_from(z) # (gh,)
                # Interpolate a(k', z') for all GH nodes at once
                pts = np.column_stack([np.repeat(k_next, len(z_next)), np.tile(z_next, len(k_next))])
                a_next = a_policy(pts).reshape(len(k_next), len(z_next)) # (Nk, gh)
                c_next = model.consumption_from_share(a_next, k_next[:, None], z_next[None, :])

                # 2) current c from Euler at (k', current z): vector in k'
                c_t = model.c_from_euler(c_next, k_next[:, None], z_next[None, :])

                # 3) map back to current k via inverse m(k,z)
                m_t = c_t + k_next # budget: m_t = c_t + k_{t+1}
                k_t = k_of_m_list[j](m_t) # 1D invert to current k_t, at fixed z_t

                # 4) current share a_t = c_t / m_t
                a_t = c_t / m_t

                # 5) Interpolate a_t(k_t) onto the exogenous k grid for this z
                sort_idx = np.argsort(k_t)
                a_interp = model.make_spline(k_t[sort_idx], a_t[sort_idx])
                a_new[:, j] = a_interp(k_grid)

            # Update policy
            a_policy = model.make_bilinear(k_grid, z_grid, a_new)

            # --- Convergence check ---------------------------------------
            converged, a_new = self._check_convergence(a_old, a_policy, k_grid, z_grid)
            if converged:
                return a_policy