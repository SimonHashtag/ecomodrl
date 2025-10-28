import numpy as np
import matplotlib.pyplot as plt
from time import time

from ecomodrl.benchmarks.portfoliochoice.problemclass import PlanningProblem, PlanningSolvers

if __name__ == "__main__":

    # Model and solvers
    model   = PlanningProblem(M=2**9)                     # delta=1.0, gamma=1.0 for closed-form sanity check
    solvers = PlanningSolvers(m_points=20,         # k grid points
                              z_points=20,          # z grid points
                              tol=1e-5,             # optimization tolerance (root finding in TI)
                              conv=1e-5,            # convergence tolerance (policy)
                              max_drop_iters=20)    # max drop iterations in node solver            

    # Solve with timing
    def timed_solve(method_name, solver, model):
        t0 = time()
        a_policy, w_policy = solver(model, verbose=True)
        t1 = time()
        print(f'{method_name} converged in {t1 - t0:.5f} seconds.')
        return a_policy, w_policy

    # Policies as bilinear interpolators
    a_policy_ti, w_policy_ti  = timed_solve('TI',  solvers.TI,  model)

    # ------------------------------------------------------------------
    # Plot 1: Optimal consumption share policies a(k | z = median)
    # ------------------------------------------------------------------

    m_grid = model.make_m_grid(points=solvers.m_points)
    z_grids = model.make_z_grids(points=solvers.z_points)
    z_mid = z_grids[0][len(z_grids[0])//2]

    a_ti = a_policy_ti(np.column_stack([m_grid, np.full_like(m_grid, z_mid)]))
    plt.figure()
    plt.plot(m_grid, a_ti,  linewidth=5, label=f"TI policy a(k, z={z_mid:.3f})")
    plt.xlabel("Capital k")
    plt.ylabel("Consumption share a(k,z)")
    plt.title("Brock–Mirman: Optimal Consumption Share Policies (TI)")
    plt.legend()
    plt.tight_layout()
    plt.show()