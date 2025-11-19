import numpy as np
import matplotlib.pyplot as plt
from time import time

from ecomodrl.benchmarks.brockmirman.problemclass import PlanningProblem, PlanningSolvers

if __name__ == "__main__":

    # Model and solvers
    model   = PlanningProblem()                     # delta=1.0, gamma=1.0 for closed-form sanity check
    solvers = PlanningSolvers(k_points=100,         # k grid points
                              z_points=10,          # z grid points
                              tol=1e-5,             # optimization tolerance (root finding in TI)
                              conv=1e-5)            # convergence tolerance (policy)

    # Solve with timing
    def timed_solve(method_name, solver, model):
        t0 = time()
        policy = solver(model)
        t1 = time()
        print(f'{method_name} converged in {t1 - t0:.5f} seconds.')
        return policy

    # Policies as bilinear interpolators
    policy_ti  = timed_solve('TI',  solvers.TI,  model)
    policy_egm = timed_solve('EGM', solvers.EGM, model)

    # ------------------------------------------------------------------
    # Plot 1: Optimal consumption share policies a(k | z = median)
    # ------------------------------------------------------------------

    k_grid = model.make_k_grid(points=solvers.k_points)
    z_grid = model.make_z_grid(points=solvers.z_points)
    z_mid = z_grid[len(z_grid)//2]

    a_ti, a_egm = (p(np.column_stack([k_grid, np.full_like(k_grid, z_mid)])) for p in (policy_ti, policy_egm))

    plt.figure()
    plt.plot(k_grid, a_ti,  linewidth=5, label=f"TI policy a(k, z={z_mid:.3f})")
    plt.plot(k_grid, a_egm, linewidth=2, label=f"EGM policy a(k, z={z_mid:.3f})")
    plt.xlabel("Capital k")
    plt.ylabel("Consumption share a(k,z)")
    plt.title("Brock–Mirman: Optimal Consumption Share Policies (TI, EGM)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Euler errors on the full (k,z) grid, simulate T=100 from each node
    euler_errors_ti,  states_ti,  U_ti   = model.get_euler_errors(k_grid, z_grid, policy_ti,  episodes=100)
    euler_errors_egm, states_egm, U_egm  = model.get_euler_errors(k_grid, z_grid, policy_egm, episodes=100)

    ti_err, egm_err = np.max(np.abs(euler_errors_ti)), np.max(np.abs(euler_errors_egm))
    print("Max absolute Euler equation error")
    print(f"TI:  {ti_err:.9f}")
    print(f"EGM: {egm_err:.9f}")

    print("\nAverage discounted episode return")
    print(f"TI:  {U_ti:.9f}")
    print(f"EGM: {U_egm:.9f}")

    # ------------------------------------------------------------------
    # Plot 2: Consumption policies (z = median) and Euler error histograms
    # ------------------------------------------------------------------

    # Consumption policies on k grid at z_mid
    c_ti, c_egm = (model.consumption_from_share(a, k_grid, z_mid) for a in (a_ti, a_egm))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: consumption policies
    ax[0].plot(k_grid, c_ti,  linewidth=5, label='Time iteration')
    ax[0].plot(k_grid, c_egm, linewidth=2, label='EGM')
    ax[0].set_title(f'Policy function (consumption) at z={z_mid:.3f}')
    ax[0].set_xlabel('Capital')
    ax[0].set_ylabel('Consumption')

    # Right: histograms of log10 absolute Euler errors, restricted to the CI
    def log_errors_ci(errors, width=0.95):
        logs = np.log10(np.abs(errors))
        alpha = (1 - width) / 2
        lo, hi = np.quantile(logs, [alpha, 1-alpha])
        return logs[(logs >= lo) & (logs <= hi)]

    logs_ti, logs_egm = map(log_errors_ci, (euler_errors_ti, euler_errors_egm))
    ax[1].hist(logs_ti,  density=True, bins=50, alpha=0.5, label='Time iteration')
    ax[1].hist(logs_egm, density=True, bins=50, alpha=0.5, label='EGM')
    ax[1].set_title('Log10 |Euler error|')
    ax[1].set_xlabel('Log10 absolute Euler equation error')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.legend()
    plt.show()