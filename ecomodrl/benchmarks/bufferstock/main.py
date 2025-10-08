import time
import numpy as np
import matplotlib.pyplot as plt

from ecomodrl.benchmarks.bufferstock.problemclass import PlanningProblem

if __name__ == "__main__":
    pp = PlanningProblem()

    # Time iterations 
    start = time.time()
    _, policy_ti = pp.time_iterations(
        grid_points=50, n_knots=8, lower_limit=1.0, upper_limit=15., 
        tol=1e-5, conv=1e-6, verbose=False)
    print(f'Time iterations took {time.time()-start} seconds.')
    
    # EGM
    start = time.time()
    _, policy_egm = pp.egm(grid_points=100, n_knots=8, lower_limit=1e-6, upper_limit=15., 
                        conv=1e-10, verbose=True)
    print(f'EGM took {time.time()-start} seconds.')

    grid = np.linspace(1., 15., 1000)
    # Euler equation errors & return comparison
    euler_errors_ti, states_ti, U_ti = pp.get_euler_errors(grid, policy_ti, n_eps=10)
    euler_errors_egm, states_egm, U_egm = pp.get_euler_errors(grid, policy_egm, n_eps=10)
    print(f'Mean absolute Euler equation error: \r\n  TI: {np.mean(euler_errors_ti)}       EGM: {np.mean(euler_errors_egm)}')
    print(f'The optimal average discounted episode return is: \r\n  TI: {U_ti}      EGM: {U_egm}')
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(np.concatenate(([0], grid)), policy_ti(np.concatenate(([0], grid))), linestyle='--', color='orange', label='Time iterations')
    ax[0].plot(np.concatenate(([0], grid)), policy_egm(np.concatenate(([0], grid))), linestyle='--', color='green', label='EGM')
    ax[0].set_title('Policy function')
    ax[0].set_xlabel('Normalized gross wealth')
    ax[0].set_ylabel('Normalized Consumption')
    ax[1].hist(euler_errors_ti, density=True, bins=50, color="orange", alpha=0.5, label='Time iterations')
    ax[1].hist(euler_errors_egm, density=True, bins=50, color="green", alpha=0.5, label='EGM')
    ax[1].set_title('Log10 absolute Euler equation errors')
    ax[1].set_xlabel('Log10 absolute Euler equation error')
    ax[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.legend()
    plt.show()

