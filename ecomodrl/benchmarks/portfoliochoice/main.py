import numpy as np
import matplotlib.pyplot as plt
from time import time

from ecomodrl.benchmarks.portfoliochoice.problemclass import PlanningProblem, PlanningSolvers

if __name__ == "__main__":

    # Model and solvers
    model   = PlanningProblem(
        # sigma_R=[0.28, 0.18],   # idiosyncratic/short-run risk: growth riskier
        # sigma_z=[0.08, 0.10],   # slow-moving uncertainty: conservative a touch higher
        # rho_z=[0.90, 0.75],
        # kappa_R=[0.00, 0.00],
        # xi_R=[+0.045, -0.030],
        M=2**11
        )                     
    solvers = PlanningSolvers(
        m_points=82,          # m grid points
        rs_points=41,
        z_points=15,          # z grid points
        tol=1e-5,             # optimization tolerance (root finding in TI)
        conv=1e-4,            # convergence tolerance (policy)
    )                     

    # Solve with timing
    def timed_solve(method_name, solver, model):
        t0 = time()
        a_policy, w_policy = solver(model, verbose=True)
        t1 = time()
        print(f'{method_name} converged in {t1 - t0:.5f} seconds.')
        return a_policy, w_policy

    # Policies as bilinear interpolators
    a_policy_egm, w_policy_egm  = timed_solve('EGM',  solvers.EGM,  model) 

    m_grid = model.make_m_grid(points=solvers.rs_points, span=(0.2, 30.), dense_low=False) # TODO: m_points or rs_points
    z_grids = model.make_z_grids(points=solvers.z_points)

    # Euler equation errors & return comparison
    euler_errors_egm, states_egm, U_egm = model.get_euler_errors(m_grid, z_grids, a_policy_egm, w_policy_egm, episodes=100)
    value_fun_egm = model.V_from_a(m_grid, z_grids, a_policy_egm, w_policy_egm, conv=1e-5)

    print(
        "Mean absolute Euler equation error: \n"
        f"EGM: {np.nanmean(euler_errors_egm)}"
    )
    print(
        "The optimal average discounted episode return is: \n"
        f"EGM: {U_egm}"
    )

    # -------------------------------------------------------
    # EGM Plots
    # -------------------------------------------------------
    z_grid = z_grids[0]  # Only considering the first risky asset for 3D plot
    M, Z = np.meshgrid(m_grid, z_grid, indexing='ij')
    pts = np.column_stack([M.ravel(), Z.ravel()])
    A_egm = a_policy_egm(pts).reshape(M.shape)
    W_egm = w_policy_egm(pts)[:,1].reshape(M.shape)
    V_egm = value_fun_egm(pts).reshape(M.shape)

    fig3 = plt.figure(figsize=(12, 5))

    # --- 3D Policy plot ---
    ax1 = fig3.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(M, Z, A_egm, cmap='viridis', edgecolor='none', alpha=0.9)

    ax1.set_title('Policy function $a_t(m, z)$', fontsize=12)
    ax1.set_xlabel('Cash-on-hand $m$')
    ax1.set_ylabel('Shock $z$')
    ax1.set_zlabel('Consumption share $a_t$')
    fig3.colorbar(surf, ax=ax1, shrink=0.6, aspect=10, label='Consumption share')

    ax2 = fig3.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(M, Z, W_egm, cmap='viridis', edgecolor='none', alpha=0.9)

    ax2.set_title('Policy function $w_t(m, z)$', fontsize=12)
    ax2.set_xlabel('Cash-on-hand $m$')
    ax2.set_ylabel('Shock $z$')
    ax2.set_zlabel('Risky asset share $w_t$')
    fig3.colorbar(surf, ax=ax2, shrink=0.6, aspect=10, label='Risky asset share')

    # --- Histogram displaying log10 absolute (maximum) Euler errors ---
    plt.figure(figsize=(8, 5))
    plt.hist(euler_errors_egm, density=True, bins=50, color="forestgreen", alpha=0.5, label='EGM')
    plt.title('Log10 absolute (maximum) Euler equation errors', fontsize=12)
    plt.xlabel('Log10 |Euler residual|')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Value function---
    fig2 = plt.figure(figsize=(12, 5))

    ax = fig2.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(M, Z, V_egm, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_title('Value Function $V(m, z)$', fontsize=12)
    ax.set_xlabel('Cash-on-hand $m$')
    ax.set_ylabel('Shock $z$')
    ax.set_zlabel('$V$')
    fig2.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='Value function')
    plt.tight_layout()
    plt.show()

    