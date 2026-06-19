# -----------------------------------------------------------------------------
# EGM solver for canonical models in a common (m, ξ) state space.
#
#   1. CassKoopmans  -- deterministic growth, ξ = ∅.
#   2. BrockMirman   -- stochastic growth with AR(1) productivity, ξ = z.
#   3. BufferStock   -- normalized buffer-stock with borrowing constraint.
#
# Policy analysis (Euler errors, Bellman value, rollout value) lives in
# analysis.py as free functions so the same code applies to EGM and RL.
# -----------------------------------------------------------------------------


from abc import abstractmethod
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.stats import norm

from ecomodrl.analysis import accuracy_metrics
from ecomodrl.configs.parameters import CassKoopmansParams, BrockMirmanParams, BufferStockParams

# =============================================================================
# Gauss-Hermite helper (shared by stochastic models)
# =============================================================================

def _gauss_hermite_nodes(n_nodes):
    """Raw Gauss-Hermite nodes / weights, scaled for E[f(eps)] under N(0, 1)."""
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    return x * np.sqrt(2), w / np.sqrt(np.pi)


# =============================================================================
# Abstract base class
# =============================================================================

class EconomicModel:
    """Physics layer: CRRA primitives, shock integration, env-facing dynamics,
    and the small set of primitives consumed by the EGM solver
    (`models.EGMModel`) and the analysis drivers (`analysis.py`)."""

    # Defaults: subclasses can override as needed.
    STATE_DIM = 1            # dimension of the raw economic state (e.g. m for CK, (m, z) for BM)
    HAS_EXO = False
    MIN_ACTION_SHARE = 1e-3  # Safe boundaries for the action (avoid -inf rewards from u(0))
    MAX_ACTION_SHARE = 1.0 - MIN_ACTION_SHARE  # Corresponding upper bound
    CONSTRAINED = False      # occasionally-binding constraint TODO: Ugly
    

    def __init__(self, params = None):
        """Initialize the model with the supplied parameters."""
        self.params = params
        self.weights = None           # quadrature weights; set by stochastic subclasses
        
    @property
    def BETA_EFF(self):     return self.params.beta     # Bufferstock model needs a tilted discount for RL path; default is the original β for EGM path. 

    # ------------------------------------------------------------------
    # Grid utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _log_grid(lo, hi, n):
        """*n* log-spaced points on [lo, hi] (lo > 0)."""
        return np.exp(np.linspace(np.log(lo), np.log(hi), n))

    # ------------------------------------------------------------------
    # CRRA utility
    # ------------------------------------------------------------------

    def utility(self, c):
        """u(c) = c^{1-GAMMA}/(1-GAMMA)  (log if GAMMA == 1)."""
        if self.params.gamma == 1.0:
            return np.log(c)
        return c ** (1 - self.params.gamma) / (1 - self.params.gamma)

    def marginal_utility(self, c):
        """u'(c) = c^{-GAMMA}"""
        return c ** (-self.params.gamma)

    def inv_marginal_utility(self, mu):
        """(u')^{-1}(mu) = mu^{-1/GAMMA}"""
        return mu ** (-1 / self.params.gamma)
    
    @abstractmethod
    def _euler_rhs(self, c_next, **trans):
        """Return β·E[u'(c')·R'] (or its model-specific analogue) for the supplied next-period transition nodes.

        `trans` is **-unpacked from either `_transition_grid` (EGM solver) or
        `next_state_distribution` (Euler-error analysis); the same signature
        serves both paths.
        """
        ...

    # ------------------------------------------------------------------
    # Shock integration
    # ------------------------------------------------------------------

    def _integrate(self, x):
        """Integrate *x* over shock nodes (last axis) with quadrature weights.

        If ``self.weights`` exists (stochastic models), contracts the last
        axis via dot product; otherwise returns *x* unchanged (deterministic).
        """
        if self.weights is not None:
            return x @ self.weights
        return x
    
    def value_kernel(self):
        """Multiplicative factor of V' inside the normalized Bellman; default 1."""
        return 1.0

    # ------------------------------------------------------------------
    # Policy evaluation helpers
    # ------------------------------------------------------------------

    def consumption(self, m, xi=None):
        """Interpolate the solved policy at arbitrary (m, ξ) points."""
        if self.STATE_DIM == 1:
            return np.interp(m, self.m_grid, self.policy)
        interp = RegularGridInterpolator(
            (self.xi_grid, self.m_grid), self.policy,
            bounds_error=False, fill_value=None,
        )
        return interp((xi, m))

    def share(self, m, xi=None):
        """Consumption share s = c/m at arbitrary (m, ξ) points."""
        return self.consumption(m, xi) / m
    
    def __call__(self, state):
        """Evaluate the solved consumption policy at flat states (..., STATE_DIM)."""
        if self.STATE_DIM == 2:
            return self.consumption(state[..., 0], state[..., 1])
        return self.consumption(state[..., 0])

    # ------------------------------------------------------------------
    # Initial policy (consume everything)
    # ------------------------------------------------------------------

    def _initial_policy(self):
        """Initial guess c(m, ξ) = m (broadcast across ξ if applicable)."""
        if self.STATE_DIM == 2:
            return np.broadcast_to(
                self.m_grid, (len(self.xi_grid), len(self.m_grid))
            ).copy()
        return self.m_grid.copy()
    
    # ------------------------------------------------------------------
    # Subclass hooks: env-facing dynamics
    # ------------------------------------------------------------------

    @abstractmethod
    def sample_shocks(self, rng, shape):
        """Draw a batch of raw random innovations for use with `step`.

        Distinct from the quadrature nodes used by `next_state_distribution`:
        these are Monte Carlo draws consumed by the RL environment. Separating
        shock generation from `step` lets callers pre-draw a full trajectory
        of length T in one vectorised call and feed shocks one step at a time.
        Returns None for deterministic models (CassKoopmans).
        """
        ...

    @abstractmethod
    def transition(self, state, c, shocks):
        """Apply one period of model dynamics to a batched (state, c) pair.

        Consumes the raw innovations from `sample_shocks` (or None for
        deterministic models) and returns the next state with shape
        (..., STATE_DIM), broadcasting over arbitrary leading dims.
        """
        ...

    @abstractmethod
    def eval_start_states(self, M):
        """Return (M, STATE_DIM) starting states for batched rollout evaluation."""
        ...
    
    # ------------------------------------------------------------------
    # Subclass hooks: analysis (off-grid transition + shock sampling)
    # ------------------------------------------------------------------

    @abstractmethod
    def next_state_distribution(self, state, a):
        """Compute the next-state distribution at arbitrary (state, a) points.

        The off-grid counterpart to `_transition_grid`: given any batch of
        (state, a) pairs, fans out over all quadrature shock nodes and
        returns (next_states, trans_kwargs). `trans_kwargs` is the same dict
        that `_euler_rhs` expects, so Euler errors can be evaluated at
        arbitrary simulated states rather than only at the solver's fixed grid.
        Integration over the node axis is handled by `_integrate` via
        `self.weights` (None for deterministic, quadrature weights otherwise).
        """
        ...


# =============================================================================
# 1. Cass-Koopmans (deterministic, single state k)
# =============================================================================

class CassKoopmans(EconomicModel):
    """Deterministic Cass-Koopmans growth model. ξ = ∅."""
    # Grid spans 0.2 to 5 times the deterministic steady-state capital
    K_MIN_FACTOR, K_MAX_FACTOR = 0.2, 5.0
    STATE_DIM = 1
    HAS_EXO = False

    def __init__(
            self, 
            params=None,
            grid_size=5000,):
        super().__init__(params)
        self.GRID_SIZE = grid_size
        self.m_ref = self.cash_on_hand(self.k_steady, 1.0)
        # Post-decision asset grid (a = next-period capital).
        self.a_grid = self._build_a_grid(self.GRID_SIZE)
        # Cash-on-hand state grid spans m of those asset levels.
        self.m_grid = self._log_grid(
            self.cash_on_hand(self.a_grid[0]),
            self.cash_on_hand(self.a_grid[-1]), self.GRID_SIZE)

        # Pre-computed transition (deterministic, no shock dim).
        self.m_next = self.cash_on_hand(self.a_grid)
        self.xi_next = None
        self.R_next = self.gross_interest_rate(self.a_grid)

    @property
    def k_steady(self):
        """Deterministic-z steady-state capital: f'(k*) = 1/β - 1 + δ."""
        return ((1 / self.params.beta - 1 + self.params.delta) / (self.params.A * self.params.alpha)) ** (1 / (self.params.alpha - 1))

    def _production(self, k, z=1.0):
        return z * self.params.A * k ** self.params.alpha

    def cash_on_hand(self, k, z=1.0):
        """m = z·f(k) + (1-δ)k  (CK: z = 1)."""
        return self._production(k, z) + (1 - self.params.delta) * k

    def gross_interest_rate(self, a, z=1.0):
        """R'(a, z') = z'·f'(a) + (1-δ)  (CK: z = 1)."""
        return z * self.params.A * self.params.alpha * a ** (self.params.alpha - 1) + (1 - self.params.delta)

    def _build_a_grid(self, n):
        """Log-spaced post-decision asset grid centered on k_steady."""
        return self._log_grid(self.K_MIN_FACTOR * self.k_steady, self.K_MAX_FACTOR * self.k_steady, n)

    def next_state_distribution(self, state, a):
        m_next = self.cash_on_hand(a)
        return m_next[..., None], {'R_next': self.gross_interest_rate(a)}
    
    def _euler_rhs(self, c_next, R_next):
        """β · E[u'(c') · R']; integration handled by ``_integrate``."""
        return self.params.beta * self._integrate(self.marginal_utility(c_next) * R_next)
    
    def sample_shocks(self, rng, shape):
        """Deterministic: no shocks."""
        return None
    
    def transition(self, state, c, shocks):
        """Batched single-step transition. `shocks` is ignored (deterministic)."""
        a = state[..., 0] - c
        return self.cash_on_hand(a)[..., None]
    
    def eval_start_states(self, M):
        """M log-spaced m values across the grid."""
        return self._log_grid(self.m_grid[0], self.m_grid[-1], M)[:, None]

# =============================================================================
# 2. Brock-Mirman (stochastic, states (k, z))
# =============================================================================

class BrockMirman(CassKoopmans):
    """Stochastic Brock-Mirman growth model with AR(1) productivity. ξ = z."""
    # Number of z-anchors used to build the (m x z) starting-state mesh.
    # quantiles (0.05 / 0.15 / ... / 0.95) of the AR(1) stationary
    # distribution of log z, log z ~ N(0, SIGMA^2 / (1 - RHO^2)).
    EVAL_Z_QUANTILES = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)

    N_GH_NODES = 7
    STATE_DIM = 2
    HAS_EXO = True

    def __init__(
            self,
            params=None,
            m_grid_size=500,
            a_grid_size=500,
            z_grid_size=100,
        ):
        super().__init__(params, m_grid_size)
        self.M_GRID_SIZE = m_grid_size
        self.A_GRID_SIZE = a_grid_size
        self.Z_GRID_SIZE = z_grid_size
        # Gauss-Hermite shock grid (innovations to log z)
        x, w = _gauss_hermite_nodes(self.N_GH_NODES)
        self.shock_nodes = self.params.sigma * x
        self.weights = w

        # Productivity grid (log-spaced symmetrically around z = 1).
        sigma_z = self.params.sigma / np.sqrt(1 - self.params.rho ** 2) if self.params.rho != 0 else self.params.sigma
        z_lo = np.exp(-3 * sigma_z)
        z_hi = np.exp(3 * sigma_z)
        self.xi_grid = self._log_grid(z_lo, z_hi, self.Z_GRID_SIZE)

        # Pre-computed transition: shape (n_z, n_a, n_shock).
        z_today = self.xi_grid[:, None, None]
        a_today = self.a_grid[None, :, None]
        z_next = self._z_next(z_today, self.shock_nodes[None, None, :])
        z_next, a_b = np.broadcast_arrays(z_next, a_today)

        # Overwrite CassKoopmans' deterministic transition with the stochastic one: m' = z'·f(a) + (1-δ)·a.
        # Cash-on-hand grid spans m across all (k, z) on the asset grid.
        m_lo = self.cash_on_hand(self.a_grid[0], self.xi_grid[0])
        m_hi = self.cash_on_hand(self.a_grid[-1], self.xi_grid[-1])
        self.m_grid = self._log_grid(m_lo, m_hi, self.M_GRID_SIZE)

        self.xi_next = z_next
        self.m_next = self.cash_on_hand(a_b, z_next)
        self.R_next = self.gross_interest_rate(a_b, z_next)

    # ------------------------------------------------------------------
    # AR(1) productivity transition (shared by all z-next sites)
    # ------------------------------------------------------------------

    def _z_next(self, z, eps):
        """z' = exp(RHO·log(z) + eps); shape-preserving on any broadcast."""
        return np.exp(self.params.rho * np.log(z) + eps)

    def next_state_distribution(self, state, a):
        z = state[..., 1]
        z_b = np.asarray(z)[..., None]
        a_b = np.asarray(a)[..., None]
        z_next = self._z_next(z_b, self.shock_nodes[None, :])
        z_next, a_bc = np.broadcast_arrays(z_next, a_b)
        m_next = self.cash_on_hand(a_bc, z_next)
        next_state = np.stack([m_next, z_next], axis=-1)
        return next_state, {'R_next': self.gross_interest_rate(a_bc, z_next)}
    
    def sample_shocks(self, rng, shape):
        """Gaussian innovations to log z."""
        return rng.normal(0.0, self.params.sigma, size=shape)

    def transition(self, state, c, shocks):
        """Batched single-step transition with pre-drawn log-z innovations."""
        a = state[..., 0] - c
        z_next = self._z_next(state[..., 1], shocks)
        m_next = self.cash_on_hand(a, z_next)
        return np.stack([m_next, z_next], axis=-1)

    def _stationary_z_anchors(self):
        sigma_z = self.params.sigma / np.sqrt(1.0 - self.params.rho ** 2) if self.params.rho != 0 else self.params.sigma
        return np.exp(norm.ppf(np.asarray(self.EVAL_Z_QUANTILES)) * sigma_z)

    def eval_start_states(self, M):
        """M_m x M_z meshgrid of (m, z) start states with M_m * M_z = M.
        M_z = len(EVAL_Z_QUANTILES); M_m = M // M_z. Returns (M_eff, 2)."""
        M_z = len(self.EVAL_Z_QUANTILES)
        M_m = int(M) // M_z
        z_anchors = self._stationary_z_anchors()
        m_anchors = self._log_grid(self.m_grid[0], self.m_grid[-1], M_m)
        Mm, Mz = np.meshgrid(m_anchors, z_anchors, indexing='ij')
        return np.stack([Mm.ravel(), Mz.ravel()], axis=-1)


# =============================================================================
# 3. Buffer stock (normalized, single state m, occasionally binding constraint)
# =============================================================================

class BufferStock(EconomicModel):
    """Normalized buffer-stock model with an occasionally binding constraint.

    A subtle but important point: the endogenous m-grid does NOT extend down to
    m = 0. The smallest endogenous m is a_grid[0] + c_endo[0] > 0. For lower m
    the borrowing constraint binds and the optimal policy is c = m. We enforce
    this by setting `HAS_CONSTRAINT = True`, so the base-class EGM step
    prepends the kink (m = 0, c = 0) before interpolating; np.interp then
    linearly connects (0, 0) to the first endogenous node and recovers c = m
    on the constrained region.

    Grid anchoring (analogue of the deterministic steady state):
    Under the Growth Impatience Condition there exists a target cash-on-hand
    m* satisfying E[m_{t+1} | m_t = m*] = m*. Since c(.) enters this fixed
    point, m* has no closed form; we solve in two passes (rough -> refined).
    
    RL training and MC welfare use the Esscher-tilted ψ sampler
        (`sample_shocks`, `transition`) together with `BETA == β_eff`,
        so the per-step reward u(c) and a constant discount recover the
        correct normalized Bellman:
            v(m) = u(c) + β · E_true[(Gψ')^{1-γ} v(m')]
                 = u(c) + β_eff · E_tilted[v(m')]
        with β_eff = β · G^{1-γ} · exp(½γ(γ-1)σ_ψ²),
             ψ_tilted ~ LN(-½σ² + (1-γ)σ², σ²).
    """
    N_GH_NODES = 7
    M_MAX_PASS1 = 50.0
    M_TARGET_FACTOR = 5.0

    STATE_DIM = 1
    HAS_EXO = False
    MAX_ACTION_SHARE = 1.0      # No upper bound on the consumption share (can consume all cash-on-hand)
    CONSTRAINED = True

    def __init__(self,
                 params=None,
                m_grid_size=5000,
                 ):
        super().__init__(params)
        self.R = 1.0 + self.params.r
        self.G = 1.0 + self.params.g
        self.M_GRID_SIZE = m_grid_size

        self._build_shocks()
        
        # Esscher exponent θ = 1-γ on X = ln ψ
        theta = 1.0 - self.params.gamma
        self._mean_perm_tilt = self._mean_perm + theta * self.params.sigma_perm**2
        M_psi = np.exp(0.5 * theta * (theta - 1.0) * self.params.sigma_perm**2)
        self._beta_eff = self.params.beta * self.G**theta * M_psi
        
        # First pass starts from a generous, model-agnostic upper bound
        self.m_grid = self._build_m_grid(self.M_MAX_PASS1, self.M_GRID_SIZE)
        # Placeholder m_ref so attribute exists before EGM solve() refines it.
        # solve() overwrites this with m_target after each pass.
        self.m_ref = 1.0
    
    @property
    def BETA_EFF(self):     return self._beta_eff     # MC path uses tilted discount

    # ------------------------------------------------------------------
    # Model primitives
    # ------------------------------------------------------------------

    def _transition_arrays(self, a):
        """a -> (m', perm') over (a, shock) nodes."""
        a = np.asarray(a)[..., None]
        m_next = (self.R / (self.G * self.perm_shocks)) * a + self.tran_shocks
        perm_next = np.broadcast_to(self.perm_shocks, m_next.shape)
        return m_next, perm_next

    def _compute_target(self, policy, m_grid):
        """Target m* satisfying E[m'|m*] = m* via Brent root-finding."""
        def excess(m):
            c = np.interp(m, m_grid, policy)
            return ((self.R / self.G) * self.inv_perm_e * (m - c) + self.tran_e - m)
        return brentq(excess, m_grid[0], m_grid[-1])

    # ------------------------------------------------------------------
    # State construction (shocks)
    # ------------------------------------------------------------------

    def _build_shocks(self):
        x, w = _gauss_hermite_nodes(self.N_GH_NODES)

        # Log-normal mean-one realizations
        self._mean_perm = -0.5 * self.params.sigma_perm ** 2
        self._mean_tran = (np.log((1 - self.params.unemp_prob * self.params.theta_low) / (1 - self.params.unemp_prob)) - 0.5 * self.params.sigma_tran ** 2)
        perm_nodes = np.exp(self._mean_perm + self.params.sigma_perm * x)
        tran_nodes = np.exp(self._mean_tran + self.params.sigma_tran * x)

        # Unemployment branch: degenerate transitory at replacement rate
        perm_unemp = perm_nodes
        tran_unemp = np.full_like(perm_nodes, self.params.theta_low)
        weights_unemp = self.params.unemp_prob * w

        # Employment branch: tensor product of permanent and transitory shocks
        perm_mesh, tran_mesh = np.meshgrid(perm_nodes, tran_nodes, indexing='ij')
        perm_emp = perm_mesh.flatten()
        tran_emp = tran_mesh.flatten()
        weights_emp = (1 - self.params.unemp_prob) * np.outer(w, w).flatten()

        # Stack both branches into flat arrays
        self.perm_shocks = np.concatenate([perm_unemp, perm_emp])
        self.tran_shocks = np.concatenate([tran_unemp, tran_emp])
        self.weights = np.concatenate([weights_unemp, weights_emp])

        # Pre-computed moments for the target equation
        self.inv_perm_e = float(np.dot(1.0 / self.perm_shocks, self.weights))
        self.tran_e = float(np.dot(self.tran_shocks, self.weights))


    @staticmethod
    def _build_m_grid(m_max, size):
        """Log-spaced m-grid on [0, m_max]; denser near the borrowing constraint."""
        return EconomicModel._log_grid(1.0, m_max + 1.0, size) - 1.0


    def next_state_distribution(self, state, a):
        m_next, perm_next = self._transition_arrays(a)
        return m_next[..., None], {'perm_next': perm_next}
    
    def _euler_rhs(self, c_next, perm_next):
        # Normalized BS Euler: u'(c) = R·β·G^(-γ)·E[u'(c'·ψ')]
        return (self.R * self.params.beta * self.G ** (-self.params.gamma) *
                self._integrate(self.marginal_utility(c_next * perm_next)))
    
    def sample_shocks(self, rng, shape):
        """Raw innovations: (eps_perm, eps_trans) ~ N(0,1); unemployed ~ Bernoulli."""
        return np.stack([
            rng.normal(size=shape),
            rng.normal(size=shape),
            (rng.random(size=shape) < self.params.unemp_prob).astype(float),
        ], axis=-1)
        
    def value_kernel(self):
        return (self.G * self.perm_shocks) ** (1.0 - self.params.gamma)

    def transition(self, state, c, shocks):
        a   = state[..., 0] - c
        psi = np.exp(self._mean_perm_tilt + self.params.sigma_perm * shocks[..., 0])
        theta_emp = np.exp(self._mean_tran + self.params.sigma_tran * shocks[..., 1])
        theta = np.where(shocks[..., 2].astype(bool), self.params.theta_low, theta_emp)
        m_next = (self.R / (self.G * psi)) * a + theta
        return m_next[..., None]
    
    def eval_start_states(self, M):
        """M log-spaced m starts across the (post-pass-2) m_grid range.
        m_grid[0] == 0 here, so restrict to strictly positive nodes."""
        m_grid_pos = self.m_grid[self.m_grid > 0]
        return self._log_grid(m_grid_pos[0], m_grid_pos[-1], M)[:, None]
    

# =============================================================================
# EGM solver 
# =============================================================================

class BaseEGM:
    """Endogenous-grid-method solver on top of `EconomicModel`."""

    def __init__(
            self, 
            econ_model,
            max_iter=1000,
            tol=1e-6,
        ):
        self.model = econ_model
        self.policy = None
        self.TOL = tol           # EGM convergence tolerance (sup-norm on policy)
        self.MAX_ITER = max_iter # EGM max iterations

    def solve(self, verbose=True):
        start = time.time()
        self.policy = self._egm_loop(self.model._initial_policy(), verbose)
        self.solve_time = time.time() - start
        if verbose:
            print(f"EGM finished in {self.solve_time:.2f} seconds.")
        return self.policy
    
    def _egm_loop(self, policy_old, verbose):
        diff = np.inf
        for iteration in range(self.MAX_ITER):
            policy_new = self._egm_step(policy_old)
            diff = np.max(np.abs(policy_new - policy_old))
            policy_old = policy_new

            if diff < self.TOL:
                if verbose:
                    print(f"\nConverged after {iteration + 1} iterations (sup-norm diff = {diff:.2e})")
                return policy_new

            if verbose:
                print(f"Iteration {iteration}: sup-norm diff = {diff:.2e}", end='\r', flush=True)
        if verbose:
            print(f"\nDid not converge after {self.MAX_ITER} iterations (sup-norm diff = {diff:.2e})")
        return policy_new

    def _egm_step(self, policy_old):
        # Next-period c(m', ξ') at every (ξ, a, shock) node.
        c_next = self._interp_policy(policy_old, self.model.m_next, self.model.xi_next)
        # Solve the Euler FOC for c (model-specific integration).
        c_endo = self.model.inv_marginal_utility(self.model._euler_rhs(c_next, **self._transition_grid()))
        # Endogenous cash-on-hand from the budget constraint.
        m_endo = self.model.a_grid + c_endo
        # Map back onto the fixed m-grid (per ξ-slice if applicable).
        return self._interp_back(m_endo, c_endo)

    def _interp_policy(self, policy, m_next, xi_next):
            """Evaluate policy on regular (xi_grid, m_grid) mesh at (m', ξ') nodes."""
            if self.model.STATE_DIM == 1:
                return np.interp(m_next.ravel(), self.model.m_grid, policy).reshape(m_next.shape)
            
            # 1. Clip coordinates to prevent negative extrapolation (which causes NaNs later)
            xi_cl = np.clip(xi_next, self.model.xi_grid[0], self.model.xi_grid[-1])
            m_cl = np.clip(m_next, self.model.m_grid[0], self.model.m_grid[-1])
            
            interp = RegularGridInterpolator(
                (self.model.xi_grid, self.model.m_grid), policy,
                bounds_error=False, fill_value=None,
            )
            
            # 2. Stack coordinates safely (older SciPy crashes if passed a tuple)
            pts = np.stack([xi_cl, m_cl], axis=-1)
            return interp(pts)

    def _interp_back(self, m_endo, c_endo):
        """Map endogenous (m, c) pairs back onto m_grid, per ξ-slice if applicable."""
        if self.model.STATE_DIM == 1:
            return self._interp_one(m_endo, c_endo)
        policy = np.empty((len(self.model.xi_grid), len(self.model.m_grid)))
        for i in range(len(self.model.xi_grid)):
            policy[i] = self._interp_one(m_endo[i], c_endo[i])
        return policy

    def _interp_one(self, m_endo, c_endo):
        """Single-slice map-back; prepends (0, 0) kink under a borrowing constraint."""
        if self.model.CONSTRAINED:
            m_endo = np.insert(m_endo, 0, 0.0)
            c_endo = np.insert(c_endo, 0, 0.0)
        return np.interp(self.model.m_grid, m_endo, c_endo)

    @abstractmethod
    def _transition_grid(self):
        """Return pre-cached transition arrays as a dict for the EGM solver.

        Called once per EGM iteration inside `_egm_step` and **-unpacked
        directly into `_euler_rhs`. The arrays (e.g. R_next, perm_next) are
        computed once in `_setup` on the fixed (a_grid, xi_grid, shock_nodes)
        mesh and reused across all iterations, making this the fast,
        grid-anchored path. For off-grid evaluation use
        `next_state_distribution` instead.
        """
        ...


class CassKoopmansEGM(BaseEGM):
    def _transition_grid(self):
        return {'R_next': self.model.R_next}
    
class BrockMirmanEGM(CassKoopmansEGM):
    """Same EGM structure as Cass-Koopmans, but with stochastic integration."""


class BufferstockEGM(BaseEGM):
    def __init__(self, econ_model, max_iter=1000, tol=1e-6):
        super().__init__(econ_model, max_iter, tol)
        self._refresh_post_decision()

    def solve(self, verbose=True):
        """Two-pass solve to refine the grid around the target m*"""
        start = time.time()

        # Pass 1: rough solve on the generous grid
        if verbose:
            print(f"Pass 1: m_max = {self.model.M_MAX_PASS1}")
        self.policy = self._egm_loop(self.model._initial_policy(), verbose)
        self.m_target = self.model._compute_target(self.policy, self.model.m_grid)
        self.model.m_ref = self.m_target
        if verbose:
            print(f"  -> target m* = {self.m_target:.4f}")

        # Pass 2: rebuild grid anchored on m* and re-solve
        m_max_refined = self.model.M_TARGET_FACTOR * self.m_target
        if verbose:
            print(f"Pass 2: m_max = {self.model.M_TARGET_FACTOR} * m* = {m_max_refined:.4f}")
            
        self.model.m_grid = self.model._build_m_grid(m_max_refined, self.model.M_GRID_SIZE)
        self._refresh_post_decision()
        self.policy = self._egm_loop(self.model._initial_policy(), verbose)
        self.m_target = self.model._compute_target(self.policy, self.model.m_grid)
        self.model.m_ref = self.m_target
        self.solve_time = time.time() - start
        
        if verbose:
            print(f"  -> target m* = {self.m_target:.4f}")
            print(f"EGM finished in {self.solve_time:.2f} seconds.")
        return self.policy
    
    def _refresh_post_decision(self):
        """Recompute (m_next, perm_next) for the current m-grid."""
        self.model.a_grid = self.model.m_grid.copy()
        self.model.m_next, self.perm_next = self.model._transition_arrays(self.model.a_grid)
        self.model.xi_next = None

    def _transition_grid(self):
        return {'perm_next': self.perm_next}

# =============================================================================
# Entry point: solve all three models and report Euler error stats
# =============================================================================

if __name__ == "__main__":

    models = [
        ("Cass-Koopmans (deterministic growth)",              CassKoopmans(CassKoopmansParams()), CassKoopmansEGM),
        ("Brock-Mirman (stochastic growth)",                  BrockMirman(BrockMirmanParams()), BrockMirmanEGM),
        ("Buffer stock (occasionally binding constraint)",    BufferStock(BufferStockParams()), BufferstockEGM),
    ]
    solved = []
    for label, model, egm_class in models:
        print(label)
        solver = egm_class(model)
        solver.solve()
        solved.append((label, model, solver))

    header = f"{'Model':<50} {'Mean log10|Euler error|':>22} {'Solve time (s)':>14}"
    print(header)
    print("=" * len(header))
    for label, model, solver in solved:
        model.policy = solver.policy
        mean_err = np.mean(accuracy_metrics.euler_errors(model, model))
        print(f"{label:<50} {mean_err:>22.4f} {solver.solve_time:>14.2f}")
    print("=" * len(header))