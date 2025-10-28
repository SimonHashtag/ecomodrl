import numpy as np
import matplotlib.pyplot as plt
from time import time

from ecomodrl.benchmarks.portfoliochoice.problemclass import PlanningProblem, PlanningSolvers

if __name__ == "__main__":

    # Model and solvers
    model   = PlanningProblem(M=2**9)                     # delta=1.0, gamma=1.0 for closed-form sanity check
    solvers = PlanningSolvers(m_points=10,         # k grid points
                              z_points=10,          # z grid points
                              tol=1e-5,             # optimization tolerance (root finding in TI)
                              conv=1e-5)            # convergence tolerance (policy)

    # Solve with timing
    def timed_solve(method_name, solver, model):
        t0 = time()
        a_policy, w_policy = solver(model)
        t1 = time()
        print(f'{method_name} converged in {t1 - t0:.5f} seconds.')
        return a_policy, w_policy

    # Policies as bilinear interpolators
    a_policy_ti, w_policy_ti  = timed_solve('TI',  solvers.TI,  model)