import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.optimize import brentq
import time

from ecomodrl.benchmarks.brockmirman.problemclass import PlanningProblem

# -----------------------
# Time iteration
# -----------------------
def time_iteration(problem: PlanningProblem,
                   Nz=10, Nk=100, Ngh=7,
                   a_init=0.2, a_MIN=1e-6,
                   tol=1e-5, max_iter=100,
                   time_limit=None):
    """
    Solve for consumption-share policy a(z,k) via time iteration.
    """
    # Set up grids
    problem.build_grids(Nz=Nz, Nk=Nk, Ngh=Ngh)

    z_grid, k_grid = problem.z_grid, problem.k_grid
    gh_w = problem.gh_w
    r_grid = problem.r_grid
    z_next_all = problem.z_next_all

    # Initial policy
    a_old = np.full((Nz, Nk), a_init, dtype=float)

    start = time.time() if time_limit is not None else None

    for it in range(max_iter):
        a_new = np.empty_like(a_old)

        # Interpolator over a_old
        rgi = problem.policy_interpolator(a_old)

        # Grid loop
        for i, z in enumerate(z_grid):
            z_next_vec = z_next_all[i, :]
            for j, k in enumerate(k_grid):
                r_t = r_grid[i, j]
                a_new[i, j] = brentq(
                    problem.euler_residual,
                    a_MIN, 1.0 - a_MIN,
                    args=(r_t, z_next_vec, gh_w, rgi)
                )

        if time_limit is not None:
            # Time-based stopping
            a_old = a_new
            if (time.time() - start) >= time_limit:
                break
        else:
            # Tolerance-based stopping
            a_old, done = problem.check_convergence(a_new, a_old, tol, it, tag="TI")
            if done:
                break

    a = a_old
    c = problem.share_to_consumption(a)
    return a, c, z_grid, k_grid, r_grid




# -----------------------
# EGM
# -----------------------
def egm(problem: PlanningProblem,
        Nz=10, Nk=1000, Ngh=7, Nkprime=None,
        a_init=0.2, tol=1e-5, max_iter=1_000,
        time_limit=None):
    """
    Endogenous Grid Method for consumption-share policy a(z,k).
    """
    # Set up grids (separate one for k' if needed)
    problem.build_grids(Nz=Nz, Nk=Nk, Ngh=Ngh, Nkprime=Nkprime)

    z_grid, k_grid, kp_grid = problem.z_grid, problem.k_grid, problem.kp_grid
    gh_w = problem.gh_w
    z_next_all = problem.z_next_all

    # Initial policy
    a_old = np.full((Nz, Nk), a_init, dtype=float)

    start = time.time() if time_limit is not None else None

    for it in range(max_iter):
        a_new = np.empty_like(a_old)

        # Interpolator for previous policy a(z,k), used for c_{t+1}
        rgi = problem.policy_interpolator(a_old)

        # Loop over current z_t; vectorize across all k' at once
        for i, z in enumerate(z_grid):

            # 1) Build quadrature panels (z', k')
            z_next_vec = z_next_all[i, :]                            # (Ngh,)
            Kp = np.tile(kp_grid, (problem.Ngh, 1))                  # (Ngh, Nkp)
            Zp = np.tile(z_next_vec[:, None], (1, problem.Nkp))      # (Ngh, Nkp)

            # 2) Evaluate a'(z', k') on all GH nodes & k'
            pts = np.column_stack((Zp.ravel(), Kp.ravel()))          # ((Ngh*Nkp), 2)
            a_next = rgi(pts).reshape(problem.Ngh, problem.Nkp)      # (Ngh, Nkp)

            # 3) Construct next-period objects
            r_next = problem.cash_on_hand(Zp, Kp)
            c_next = a_next * r_next
            up_next = problem.u_prime(c_next)
            R_next = (1.0 - problem.DELTA) + problem.f_prime(Zp, Kp)
            rhs = problem.BETA * (gh_w[:, None] * (up_next * R_next)).sum(axis=0)  # (Nkp,)

            # 4) Euler inversion at each k': u'(c_t) = beta * E[u'(c_{t+1}) R_{t+1}]
            c_today = problem.inv_u_prime(rhs)                                     # (Nkp,)

            # 5) Endogenous cash_on_hand implied by (c_t, k'): r_endo = c_t + k'
            r_endo = c_today + kp_grid                                             # (Nkp,)

            # 6) Invert the true resource function r(z,k) to get k = k(r)
            r_func = problem.cash_on_hand(z, k_grid)                                  # (Nk,)
            cs_inv = CubicSpline(r_func, k_grid, bc_type="natural")                # inverse m -> k
            k_endo = cs_inv(r_endo)

            # 7) On the endogenous grid, the policy is a = c / r
            a_endo = c_today / r_endo

            # 8) Interpolate back to the fixed state grid k_grid
            order = np.argsort(k_endo)
            k_sorted, a_sorted = k_endo[order], a_endo[order]
            k_unique, idx = np.unique(k_sorted, return_index=True)
            a_unique = a_sorted[idx]
            cs = CubicSpline(k_unique, a_unique, bc_type="natural")
            a_new[i, :] = cs(k_grid)

        if time_limit is not None:
            # Time-based stopping
            a_old = a_new
            if (time.time() - start) >= time_limit:
                break
        else:
            # Tolerance-based stopping
            a_old, done = problem.check_convergence(a_new, a_old, tol, it, tag="EGM")
            if done:
                break

    # Final objects
    a = a_old
    c = problem.share_to_consumption(a)
    return a, c, z_grid, k_grid, problem.r_grid


def get_V_from_c(problem: PlanningProblem, c_policy, tol=1e-6, max_iter=10_000):
    
    """
    Policy evaluation for a given consumption policy c(z,k) on (problem.z_grid, problem.k_grid).

    Computes V^π as the fixed point of:
        V(z,k) = u(c(z,k)) + BETA * E_z'[ V(z', k') ],
        k' = f(z,k) + (1-δ)k - c(z,k)

    Returns
    -------
    V : ndarray, shape (Nz, Nk)
        The evaluated value function on the grid.
    """

    Nz, Nk = problem.Nz, problem.Nk
    z_grid, k_grid = problem.z_grid, problem.k_grid
    gh_w, z_next_all = problem.gh_w, problem.z_next_all

    # Precompute: cash_on_hand and next-period capital implied by the policy
    r_grid = problem.r_grid                       # r(z,k) = f(z,k) + (1-δ)k
    k_next = r_grid - c_policy                    # implied k'(z,k)

    # Initialize and iterate the Bellman policy-evaluation operator
    V_old = np.zeros((Nz, Nk), dtype=float)
    u = problem.u
    BETA = problem.BETA
    Ngh = problem.Ngh

    for _ in range(max_iter):
        # Interpolator for V_old over (z,k)
        V_interp = RegularGridInterpolator(
            (z_grid, k_grid), V_old, bounds_error=False, fill_value=None
        )

        V_new = np.empty_like(V_old)

        # Loop over current z; vectorize across all k for that z
        for i in range(Nz):
            z_next_vec = z_next_all[i, :]                        # (Ngh,)

            # Build evaluation panels (z', k') with shape (Ngh, Nk)
            Zp = np.tile(z_next_vec[:, None], (1, Nk))           # (Ngh, Nk)
            Kp = np.tile(k_next[i, :][None, :], (Ngh, 1))        # (Ngh, Nk)

            # Evaluate V_old(z', k') off-grid via 2D interpolation
            pts = np.column_stack((Zp.ravel(), Kp.ravel()))      # ((Ngh*Nk), 2)
            V_next = V_interp(pts).reshape(Ngh, Nk)              # (Ngh, Nk)

            # Expectation over shocks with GH weights
            EV_next = (gh_w[:, None] * V_next).sum(axis=0)       # (Nk,)

            # Bellman update given fixed policy c(z,k)
            V_new[i, :] = u(c_policy[i, :]) + BETA * EV_next

        # Convergence check
        diff = np.max(np.abs(V_new - V_old))
        if diff < tol:
            return V_new

        V_old = V_new

    # If not converged within max_iter, return last iterate
    return V_old


# -----------------------
# Simulation
# -----------------------
def simulate_euler_errors(problem: PlanningProblem, a_policy, T=1000, k0=None, z0=None, seed=0):
    """
    Simulate a trajectory and return log10|Euler residual| series.
    - a_policy: array policy a(z,k) on (z_grid, k_grid)
    - Uses GH-based expectation for Euler error (same as the solvers).
    """
    rng = np.random.default_rng(seed)
    rgi = problem.policy_interpolator(a_policy)

    # Initial state
    if k0 is None:
        k_t = rng.choice(problem.k_grid)
    else:
        k_t = float(k0)
    if z0 is None:
        z_t = float(problem.z_grid[len(problem.z_grid)//2])  # median z
    else:
        z_t = float(z0)

    err_log10 = np.empty(T)
    gh_x, gh_w = problem.gh_x, problem.gh_w
    RHO, SIGMA = problem.RHO, problem.SIGMA

    for t in range(T):
        # cash_on_hand and current policy
        r_t = problem.cash_on_hand(z_t, k_t)
        a_t = float(rgi((z_t, k_t)))

        # GH nodes for z_{t+1} around current z_t
        z_next_vec = np.exp(RHO * np.log(z_t) + SIGMA * gh_x)

        # Euler residual at (z_t, k_t) using expected next-period terms
        res = float(problem.euler_residual(a_t, r_t, z_next_vec, gh_w, rgi))
        # Record log10 absolute error
        err_log10[t] = np.log10(max(1e-18, abs(res)))

        # State transition: realize one shock for the path
        eps = rng.standard_normal()
        z_t = np.exp(RHO * np.log(z_t) + SIGMA * eps)
        # Capital transition
        c_t = a_t * r_t
        k_t = r_t - c_t

    return err_log10[int(0.1*T):] # use first 10% as burn-in to reduce dependency on starting value


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    problem = PlanningProblem(ALPHA=0.36, DELTA=0.1, BETA=0.99,
                              GAMMA=1.0, RHO=0.9, SIGMA=0.04)

    # Equal setup for both
    Nz, Nk, Ngh = 10, 100, 7
    a_init, max_iter = 0.2, 200
    time_limit, tol = None, 1e-5

    # TI
    a_ti, c_ti, z_grid, k_grid, r_grid = time_iteration(
        problem, Nz=Nz, Nk=Nk, Ngh=Ngh,
        a_init=a_init, a_MIN=1e-3,
        tol=tol, max_iter=max_iter,
        time_limit=time_limit
    )

    # EGM
    a_egm, c_egm, z_grid2, k_grid2, r_grid2 = egm(
        problem, Nz=Nz, Nk=Nk, Ngh=Ngh,
        a_init=a_init, tol=tol, max_iter=max_iter,
        time_limit=time_limit
    )

    ### Euler Error Histograms
    # Simulate and collect Euler errors
    T, n_bins = 1_000, 50
    errs_ti = simulate_euler_errors(problem, a_ti, T=T, seed=42)
    errs_egm = simulate_euler_errors(problem, a_egm, T=T, seed=42)

    # Define common bins over the union of both ranges
    all_errs = np.concatenate([errs_ti, errs_egm])
    bins = np.linspace(all_errs.min(), all_errs.max(), n_bins+1)

    # Plot hsitograms
    plt.figure(figsize=(8, 5))
    plt.hist(errs_ti,  bins=bins, alpha=0.6, label="TI")
    plt.hist(errs_egm, bins=bins, alpha=0.6, label="EGM")
    plt.title("Histogram of log10 |Euler residual|")
    plt.xlabel("log10 |residual|")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    ### Value Function Plots
    # Value functions for the TI and EGM policies
    V_ti = get_V_from_c(problem, c_ti,  tol=1e-6, max_iter=10_000, verbose=False)
    V_egm = get_V_from_c(problem, c_egm, tol=1e-6, max_iter=10_000, verbose=False)

    # Plot: one subplot per z
    Nz = len(z_grid)
    nrows = int(np.ceil(np.sqrt(Nz)))
    ncols = int(np.ceil(Nz / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.2*nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i in range(Nz):
        ax = axes[i]
        ax.plot(k_grid, V_ti[i, :],  label="TI", linewidth=3)
        ax.plot(k_grid, V_egm[i, :], label="EGM")
        ax.set_title(f"z = {z_grid[i]:.3f}")
        if i % ncols == 0:
            ax.set_ylabel("V(z,k)")
        if i >= (nrows-1)*ncols:
            ax.set_xlabel("k")
        if i == 0:
            ax.legend()

    # Turn off any unused subplots
    for j in range(Nz, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Value functions by shock state z")
    plt.tight_layout()
    plt.show()