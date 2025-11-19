import numpy as np
import matplotlib.pyplot as plt
from time import time

from ecomodrl.benchmarks.portfoliochoice.problemclass import PlanningProblem, PlanningSolvers

if __name__ == "__main__":

    # Model and solvers
    model   = PlanningProblem(
        sigma_R=[0.28, 0.18],   # idiosyncratic/short-run risk: growth riskier
        sigma_z=[0.08, 0.10],   # slow-moving uncertainty: conservative a touch higher
        rho_z=[0.90, 0.75],
        kappa_R=[0.00, 0.00],
        xi_R=[+0.045, -0.030],
        M=2**10
        )                     
    solvers = PlanningSolvers(
        m_points=200,         # m grid points (only used for interpolation to common rectangular grid)
        rs_points=51,         # rs grid points
        z_points=15,          # z grid points
        tol=1e-5,             # optimization tolerance (root finding for weights)
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

    m_grid = model.make_m_grid(points=30, span=(0.1, 30.), dense_low=False) # TODO: m_points or rs_points
    z_grids = model.make_z_grids(points=solvers.z_points)

    # Euler equation errors & return comparison
    euler_errors_egm, states_egm, U_egm = model.get_euler_errors(m_grid, z_grids, a_policy_egm, w_policy_egm, episode_len=100)

    print(
        "Mean absolute Euler equation error: \n"
        f"EGM: {np.nanmean(euler_errors_egm)}"
    )
    print(
        "The optimal average discounted episode return is: \n"
        f"EGM: {U_egm}"
    )

    # Value function evaluation
    value_fun_egm = model.V_from_a(m_grid, z_grids, a_policy_egm, w_policy_egm, conv=1e-4)

    # -------------------------------------------------------
    # EGM Plots
    # -------------------------------------------------------
    n_risky = len(z_grids)
    z_meds = np.array([zg[len(zg)//2] for zg in z_grids], dtype=float)

    # choose which z to vary in the m–z plot (first one)
    z_idx = 0
    z_grid = z_grids[z_idx]

    # build (m,z_k) mesh
    M, Z = np.meshgrid(m_grid, z_grid, indexing='ij')

    # assemble full points: columns = [m, z1, ..., zK]
    pts = np.empty((M.size, 1 + n_risky), dtype=float)
    pts[:, 0] = M.ravel()
    # start with all z at medians
    pts[:, 1:] = z_meds[None, :]
    # overwrite the varying z_k with the mesh values
    pts[:, 1 + z_idx] = Z.ravel()

    # evaluate policies/value
    A_egm = a_policy_egm(pts).reshape(M.shape)
    W_full = w_policy_egm(pts)                    # shape: (N, n_assets)
    W_egm  = W_full[:, 1].reshape(M.shape)        # risky-1 share (column 1)
    V_egm  = value_fun_egm(pts).reshape(M.shape)


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
    if n_risky > 1:
        # pick an m0 (median of m_grid)
        m0 = m_grid[len(m_grid)//2]
        Z1, Z2 = np.meshgrid(z_grids[0], z_grids[1], indexing='ij')
        pts2 = np.empty((Z1.size, 1 + n_risky), dtype=float)
        pts2[:, 0] = m0
        # set all z to medians first
        pts2[:, 1:] = z_meds[None, :]
        # overwrite first two z columns with mesh
        pts2[:, 1 + 0] = Z1.ravel()
        pts2[:, 1 + 1] = Z2.ravel()

        A_m0 = a_policy_egm(pts2).reshape(Z1.shape)

        fig4 = plt.figure(figsize=(6, 5))
        ax4 = fig4.add_subplot(1, 1, 1, projection='3d')
        surf = ax4.plot_surface(Z1, Z2, A_m0, cmap='viridis', edgecolor='none', alpha=0.9)
        ax4.set_title(r'Policy $a_t(m_0, z_1, z_2)$ with $m_0=%.3g$ (others at medians)' % m0)
        ax4.set_xlabel(r'$z_1$')
        ax4.set_ylabel(r'$z_2$')
        ax4.set_zlabel('Consumption share $a_t$')
        fig4.colorbar(surf, ax=ax4, shrink=0.6, aspect=10, label='Consumption share')
        plt.tight_layout()

    plt.show()

    