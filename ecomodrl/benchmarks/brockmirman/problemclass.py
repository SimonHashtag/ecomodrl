import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import RegularGridInterpolator

class PlanningProblem:
    def __init__(self,
                 ALPHA=0.36, DELTA=1.0,
                 BETA=0.99, GAMMA=1.0,
                 RHO=0.9, SIGMA=0.04):
        self.ALPHA = ALPHA
        self.DELTA = DELTA
        self.BETA = BETA
        self.GAMMA = GAMMA
        self.RHO = RHO
        self.SIGMA = SIGMA

        # Will be filled by build_grids()
        self.Nz = self.Nk = self.Nkp = self.Ngh = None
        self.z_grid = None
        self.k_grid = None
        self.kp_grid = None
        self.Z = self.K = None
        self.r_grid = None
        self.gh_x = self.gh_w = None
        self.z_next_all = None

    # --------- primitives ----------
    def u(self, c):
        return (c ** (1-self.GAMMA)) / (1-self.GAMMA) if self.GAMMA != 1 else np.log(c)

    def u_prime(self, c):
        return c ** (-self.GAMMA)

    def inv_u_prime(self, mu):
        return mu ** (-1.0 / self.GAMMA)

    def f(self, z, k):
        return z * (k ** self.ALPHA)

    def f_prime(self, z, k):
        return self.ALPHA * z * (k ** (self.ALPHA - 1))

    def cash_on_hand(self, z, k):
        return self.f(z, k) + (1.0 - self.DELTA) * k

    # --------- setup ----------
    def build_grids(self,
                    Nz=10, Nk=100, Ngh=7,
                    k_min=0.05, k_max=None,
                    Nkprime=None):
        """
        Build grids, GH nodes/weights, cash_on_hand grid, and z_{t+1} panels.
        Set Nkprime to use a distinct k'-grid for EGM.
        """
        self.Nz, self.Nk, self.Ngh, self.Nkp = Nz, Nk, Ngh, Nkprime if Nkprime is not None else Nk

        # Shock grid
        sig_stat = self.SIGMA / np.sqrt(1.0 - self.RHO**2)
        self.z_grid = np.exp(np.linspace(-2.5 * sig_stat, 2.5 * sig_stat, Nz))

        # Capital grid bounds
        if k_max is None:
            k_max = 2.0 * (self.ALPHA * self.z_grid[-1] /
                           (1.0 / self.BETA - (1.0 - self.DELTA))) ** (1.0 / (1.0 - self.ALPHA))

        self.k_grid  = np.exp(np.linspace(np.log(k_min), np.log(k_max), Nk))
        self.kp_grid = np.exp(np.linspace(np.log(k_min), np.log(k_max), self.Nkp))

        # Mesh Z, K + cash_on_hand on grid
        self.Z, self.K = np.meshgrid(self.z_grid, self.k_grid, indexing="ij")
        self.r_grid = self.cash_on_hand(self.Z, self.K)

        # GH nodes & weights for N(0,1)
        gh_x, gh_w = hermgauss(Ngh)
        self.gh_x = gh_x * np.sqrt(2.0)
        self.gh_w = gh_w / np.sqrt(np.pi)

        # Precompute z_{t+1} for each z_t and GH node -> (Nz, Ngh)
        self.z_next_all = np.exp(self.RHO * np.log(self.z_grid)[:, None] + self.SIGMA * self.gh_x[None, :])

    # --------- helper: policy interpolator ----------
    def policy_interpolator(self, a):
        """
        RegularGridInterpolator for a(z,k).
        """
        return RegularGridInterpolator(
            (self.z_grid, self.k_grid), a, bounds_error=False, fill_value=None
        )

    # --------- Euler residual ----------
    def euler_residual(self, a_t, r_t, z_next_vec, gh_w, rgi):
        """
        Residual: u'(c_t) - beta * E[ u'(c_{t+1}) * ((1-delta) + f_k(z',k')) ]
        Choice variable: a_t in (0,1) (consumption share today).
        """
        # current period
        c_t = a_t * r_t
        k_next = r_t - c_t

        # a'(z',k') via interpolation
        points = np.column_stack((z_next_vec, np.full_like(z_next_vec, k_next)))
        a_next_vec = rgi(points)

        # next-period consumption and MU
        c_next_vec = a_next_vec * self.cash_on_hand(z_next_vec, k_next)
        up_next_vec = self.u_prime(c_next_vec)

        rhs = self.BETA * np.dot(
            gh_w, up_next_vec * ((1.0 - self.DELTA) + self.f_prime(z_next_vec, k_next))
        )

        return self.u_prime(c_t) - rhs

    # --------- convergence check ----------
    def check_convergence(self, a_new, a_old, tol, it, tag=""):
        """
        Print iteration diagnostics and return (next_a_old, converged).
        """
        diff = np.max(np.abs(a_new - a_old))
        print(f"[{tag}] Iteration {it}: max policy diff = {diff:.2e}")
        converged = diff < tol
        if converged:
            print(f"[{tag}] Converged!")
        return a_new, converged

    # --------- shares -> total consumption ----------
    def share_to_consumption(self, a):
        """
        Map consumption shares a(z,k) to total consumption c(z,k) using cached r_grid.
        """
        return a * self.r_grid