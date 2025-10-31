import numpy as np
import matplotlib.pyplot as plt
from time import time

from ecomodrl.benchmarks.portfoliochoice.problemclass import PlanningProblem, PlanningSolvers

if __name__ == "__main__":

    # Model and solvers
    model   = PlanningProblem(M=2**11)                     
    solvers = PlanningSolvers(
        m_points=20,          # m grid points
        z_points=20,          # z grid points
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
    a_policy_vfi, w_policy_vfi  = timed_solve('VFI',  solvers.VFI,  model)
    a_policy_ti, w_policy_ti  = timed_solve('TI',  solvers.TI,  model)

    # ------------------------------------------------------------------
    # Plot 1: Optimal consumption share policies a(k | z = median)
    # ------------------------------------------------------------------

    m_grid = model.make_m_grid(points=solvers.m_points)
    z_grids = model.make_z_grids(points=solvers.z_points)

    # Euler equation errors & return comparison
    euler_errors_vfi, states_vfi, U_vfi = model.get_euler_errors(m_grid, z_grids, a_policy_vfi, w_policy_vfi, episodes=100)
    euler_errors_ti, states_ti, U_ti = model.get_euler_errors(m_grid, z_grids, a_policy_ti, w_policy_ti, episodes=100)
    print(np.mean(euler_errors_ti))
    print(U_ti)

    print(
        "Mean absolute Euler equation error: \n"
        f"VFI: {np.mean(euler_errors_vfi)};  TI: {np.mean(euler_errors_ti)}"
    )
    print(
        "The optimal average discounted episode return is: \n"
        f"VFI: {U_vfi}; TI: {U_ti}"
    )

    z_grid = z_grids[0]  # Only considering the first risky asset for 3D plot

    M, Z = np.meshgrid(m_grid, z_grid, indexing='ij')
    pts = np.column_stack([M.ravel(), Z.ravel()])
    A_ti = a_policy_ti(pts).reshape(M.shape)
    W_ti = w_policy_ti(pts)[:,1].reshape(M.shape)

    fig1 = plt.figure(figsize=(12, 5))

    # --- 3D Policy plot ---
    ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(M, Z, A_ti, cmap='viridis', edgecolor='none', alpha=0.9)

    ax1.set_title('Policy function $a_t(m, z)$', fontsize=12)
    ax1.set_xlabel('Cash-on-hand $m$')
    ax1.set_ylabel('Shock $z$')
    ax1.set_zlabel('Consumption share $a_t$')
    fig1.colorbar(surf, ax=ax1, shrink=0.6, aspect=10, label='Consumption share')

    ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(M, Z, W_ti, cmap='viridis', edgecolor='none', alpha=0.9)

    ax2.set_title('Policy function $w_t(m, z)$', fontsize=12)
    ax2.set_xlabel('Cash-on-hand $m$')
    ax2.set_ylabel('Shock $z$')
    ax2.set_zlabel('Risky asset share $w_t$')
    fig1.colorbar(surf, ax=ax2, shrink=0.6, aspect=10, label='Risky asset share')

    # --- Histogram displaying log10 absolute (maximum) Euler errors ---
    plt.figure(figsize=(8, 5))
    plt.hist(euler_errors_ti, density=True, bins=50, color="purple", alpha=0.5, label='Time iterations')
    plt.title('Log10 absolute (maximum) Euler equation errors', fontsize=12)
    plt.xlabel('Log10 |Euler residual|')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

    A_vfi = a_policy_vfi(pts).reshape(M.shape)
    W_vfi = w_policy_vfi(pts)[:,1].reshape(M.shape)

    fig1 = plt.figure(figsize=(12, 5))

    # --- 3D Policy plot ---
    ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(M, Z, A_vfi, cmap='viridis', edgecolor='none', alpha=0.9)

    ax1.set_title('VFI Policy function $a_t(m, z)$', fontsize=12)
    ax1.set_xlabel('Cash-on-hand $m$')
    ax1.set_ylabel('Shock $z$')
    ax1.set_zlabel('Consumption share $a_t$')
    fig1.colorbar(surf, ax=ax1, shrink=0.6, aspect=10, label='Consumption share')

    ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(M, Z, W_vfi, cmap='viridis', edgecolor='none', alpha=0.9)

    ax2.set_title('VFI Policy function $w_t(m, z)$', fontsize=12)
    ax2.set_xlabel('Cash-on-hand $m$')
    ax2.set_ylabel('Shock $z$')
    ax2.set_zlabel('Risky asset share $w_t$')
    fig1.colorbar(surf, ax=ax2, shrink=0.6, aspect=10, label='Risky asset share')

    # --- Histogram displaying log10 absolute (maximum) Euler errors ---
    plt.figure(figsize=(8, 5))
    plt.hist(euler_errors_vfi, density=True, bins=50, color="orange", alpha=0.5, label='Value Function iterations')
    plt.title('Log10 absolute (maximum) Euler equation errors', fontsize=12)
    plt.xlabel('Log10 |Euler residual|')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()