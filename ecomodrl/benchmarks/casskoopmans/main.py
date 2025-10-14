import numpy as np
import matplotlib.pyplot as plt
from time import time

from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningSolvers, make_grid

if __name__ == "__main__":

    # Model and solvers
    model   = PlanningProblem()
    solvers = PlanningSolvers(grid_points=1000,
                              lower_limit=0.1,
                              upper_limit=15.0,
                              tol=1e-5,         # optimization tolerance
                              conv=1e-5)        # convergence tolerance (consumption share policy)

    # Solve with timing
    def timed_solve(method_name, solver, model):
        t0 = time()
        policy = solver(model)
        t1 = time()
        print(f'{method_name} converged in {t1 - t0:.3f} seconds.')
        return policy

    # Policies as spline objects
    policy_vfi = timed_solve('VFI', solvers.VFI, model)
    policy_ti  = timed_solve('TI',  solvers.TI,  model)
    policy_egm = timed_solve('EGM', solvers.EGM, model)

    # Evaluate policies on their grid
    grid  = make_grid(solvers.grid_points, solvers.lower_limit, solvers.upper_limit)
    a_vfi, a_ti, a_egm = (p(grid) for p in (policy_vfi, policy_ti, policy_egm))

    # Plot
    plt.figure()
    plt.plot(grid, a_vfi, linewidth=5, label="VFI policy a(k)")
    plt.plot(grid, a_ti,  linewidth=3, label="TI policy a(k)")
    plt.plot(grid, a_egm, linewidth=1, label="EGM policy a(k)")
    plt.xlabel("Capital k")
    plt.ylabel("Consumption share a(k)")
    plt.title("Cass–Koopmans: Optimal Consumption Share Policies (VFI, TI, EGM)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Euler equation errors & return comparison (CI histograms)
    # ------------------------------------------------------------------
    euler_errors_ti,  _, U_ti   = model.get_euler_errors(grid, policy_ti,  n_eps=100)
    euler_errors_vfi, _, U_vfi  = model.get_euler_errors(grid, policy_vfi, n_eps=100)
    euler_errors_egm, _, U_egm  = model.get_euler_errors(grid, policy_egm, n_eps=100)

    print("Max absolute Euler equation error:")
    print(f"TI:  {np.max(np.abs(euler_errors_ti)):.6f}")
    print(f"VFI: {np.max(np.abs(euler_errors_vfi)):.6f}")
    print(f"EGM: {np.max(np.abs(euler_errors_egm)):.6f}")

    print("\nAverage discounted episode return:")
    print(f"TI:  {U_ti:.6f}")
    print(f"VFI: {U_vfi:.6f}")
    print(f"EGM: {U_egm:.6f}")

    # Consumption policies on the same grid for reference
    c_vfi, c_ti, c_egm = (
        model.consumption_from_share(a, grid) for a in (a_vfi, a_ti, a_egm)
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: consumption policies
    ax[0].plot(grid, c_vfi, linewidth=5, label='Value function iteration')
    ax[0].plot(grid, c_ti,  linewidth=3, label='Time iteration')
    ax[0].plot(grid, c_egm, linewidth=1, label='EGM')
    ax[0].set_title('Policy function (consumption)')
    ax[0].set_xlabel('Capital')
    ax[0].set_ylabel('Consumption')

    # Right: histograms of log10 absolute Euler errors, restricted to the 95% CI
    def log_errors_ci(errors, width=0.95):
        logs = np.log10(np.abs(errors))
        alpha = (1 - width) / 2
        lo, hi = np.quantile(logs, [alpha, 1-alpha])
        return logs[(logs >= lo) & (logs <= hi)]

    logs_vfi, logs_ti, logs_egm = map(log_errors_ci, (euler_errors_vfi, euler_errors_ti, euler_errors_egm))

    ax[1].hist(logs_vfi, density=True, bins=50, alpha=0.5, label='Value function iteration')
    ax[1].hist(logs_ti,  density=True, bins=50, alpha=0.5, label='Time iteration')
    ax[1].hist(logs_egm, density=True, bins=50, alpha=0.5, label='EGM')
    ax[1].set_title('Log10 |Euler error|')
    ax[1].set_xlabel('Log10 absolute Euler equation error')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.legend()
    plt.show()