import numpy as np
import matplotlib.pyplot as plt
from time import time
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem, PlanningSolvers

if __name__ == "__main__":

    # Model and solvers
    model   = PlanningProblem()                 # delta=1.0, gamma=1.0 for closed-form sanity check
    solvers = PlanningSolvers(grid_points=1000, # number of grid points for state space
                              tol=1e-5,         # optimization tolerance (root finding in time iteration)
                              conv=1e-5)        # convergence tolerance (consumption share policy)

    # Solve with timing
    def timed_solve(method_name, solver, model):
        t0 = time()
        policy = solver(model)
        t1 = time()
        print(f'{method_name} converged in {t1 - t0:.5f} seconds.')
        return policy

    # Policies as spline objects
    policy_ti  = timed_solve('TI',  solvers.TI,  model)
    policy_egm = timed_solve('EGM', solvers.EGM, model)

    # ------------------------------------------------------------------
    # Plot 1: Optimal consumption share policies a(k)
    # ------------------------------------------------------------------

    grid = model.make_grid(points=solvers.grid_points)
    a_ti, a_egm = (p(grid) for p in (policy_ti, policy_egm))

    plt.figure()
    plt.plot(grid, a_ti,  linewidth=5, label="TI policy a(k)")
    plt.plot(grid, a_egm, linewidth=2, label="EGM policy a(k)")
    plt.xlabel("Capital k")
    plt.ylabel("Consumption share a(k)")
    plt.title("Cass–Koopmans: Optimal Consumption Share Policies (TI, EGM)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    euler_errors_ti,  states_ti,  U_ti   = model.get_euler_errors(grid, policy_ti,  episodes=100)
    euler_errors_egm, states_egm, U_egm  = model.get_euler_errors(grid, policy_egm, episodes=100)

    ti_err, egm_err = np.max(np.abs(euler_errors_ti)), np.max(np.abs(euler_errors_egm))
    print("Max absolute Euler equation error")
    print(f"TI:  {ti_err:.9f}")
    print(f"EGM: {egm_err:.9f}")

    print("\nAverage discounted episode return")
    print(f"TI:  {U_ti:.9f}")
    print(f"EGM: {U_egm:.9f}")

    # ------------------------------------------------------------------
    # Plot 2: Consumption policies and Euler error histograms
    # ------------------------------------------------------------------

    # Consumption policies on the same grid for reference
    c_ti, c_egm = (model.consumption_from_share(a, grid) for a in (a_ti, a_egm))

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: consumption policies
    ax[0].plot(grid, c_ti,  linewidth=5, label='Time iteration')
    ax[0].plot(grid, c_egm, linewidth=2, label='EGM')
    ax[0].set_title('Policy function (consumption)')
    ax[0].set_xlabel('Capital')
    ax[0].set_ylabel('Consumption')

    # Right: histograms of log10 absolute Euler errors, restricted to the CI
    def log_errors_ci(errors, width=0.95):
        logs = np.log10(np.abs(errors))
        alpha = (1 - width) / 2
        lo, hi = np.quantile(logs, [alpha, 1-alpha])
        return logs[(logs >= lo) & (logs <= hi)]

    # Compute log10 Euler errors with CI
    logs_ti, logs_egm = map(log_errors_ci, (euler_errors_ti, euler_errors_egm))

    ax[1].hist(logs_ti,  density=True, bins=50, alpha=0.5, label='Time iteration')
    ax[1].hist(logs_egm, density=True, bins=50, alpha=0.5, label='EGM')
    ax[1].set_title('Log10 |Euler error|')
    ax[1].set_xlabel('Log10 absolute Euler equation error')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.legend()
    plt.show()