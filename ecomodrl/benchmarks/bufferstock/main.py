import time
import numpy as np
import matplotlib.pyplot as plt

from ecomodrl.benchmarks.bufferstock.problemclass import PlanningProblem, PlanningSolvers

if __name__ == "__main__":
    # Model and solvers
    model = PlanningProblem()
    solvers = PlanningSolvers(grid_points=100, conv=1e-5)

    # Time iterations
    start = time.time()
    a_ti = solvers.TI(model)
    print(f"Time iterations took {time.time() - start} seconds.")

    # EGM
    start = time.time()
    a_egm = solvers.EGM(model)
    print(f"EGM took {time.time() - start} seconds.")

    m_grid = np.linspace(1e-4, 15.0, 1000)

    # Euler equation errors & return comparison
    euler_errors_ti, m_states_ti, U_ti = model.get_euler_errors(m_grid, a_ti, episodes=10)
    euler_errors_egm, m_states_egm, U_egm = model.get_euler_errors(m_grid, a_egm, episodes=10)

    print(
        "Mean absolute Euler equation error: \n"
        f"TI: {np.mean(euler_errors_ti)} EGM: {np.mean(euler_errors_egm)}"
    )
    print(
        "The optimal average discounted episode return is: \n"
        f"TI: {U_ti} EGM: {U_egm}"
    )

    m_plot = np.concatenate(([0.0], m_grid))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(m_plot, a_ti(m_plot), linestyle='--', color='orange', label='Time iterations')
    ax[0].plot(m_plot, a_egm(m_plot), linestyle='--', color='green', label='EGM')
    ax[0].set_title('Policy function a(m)')
    ax[0].set_xlabel('Cash-on-hand m')
    ax[0].set_ylabel('Consumption share a_t')
    ax[1].hist(euler_errors_ti, density=True, bins=50, color="orange", alpha=0.5, label='Time iterations')
    ax[1].hist(euler_errors_egm, density=True, bins=50, color="green", alpha=0.5, label='EGM')
    ax[1].set_title('Log10 absolute Euler equation errors')
    ax[1].set_xlabel('Log10 |Euler residual|')
    ax[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.legend()
    plt.show()