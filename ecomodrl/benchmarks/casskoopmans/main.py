import matplotlib.pyplot as plt
import numpy as np
import time

from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem

if __name__ == "__main__":
    pp = PlanningProblem()

    # Time iterations 
    start = time.time()
    policy_TI = pp.TI(
        grid_points=30, lower_limit=0.1, upper_limit=15., 
        tol=1e-5, conv=0.0001, verbose=False)
    print(f'Time iterations took {time.time()-start} seconds.')

    # VFI
    start = time.time()
    policy_VFI = pp.VFI(grid_points=100, lower_limit=0.1, upper_limit=15., 
                        tol=1e-5, conv=0.001, verbose=False)
    print(f'VFI took {time.time()-start} seconds.')
    
    # EGM
    start = time.time()
    grid, policy_EGM = pp.EGM(grid_points=100, lower_limit=0.1, upper_limit=15., 
                        conv=0.001, verbose=False)
    print(f'EGM took {time.time()-start} seconds.')
    
    # Euler equation errors & return comparison
    euler_errors_TI, _, U_TI = pp.get_euler_errors(grid, policy_TI, n_eps=100)
    euler_errors_VFI, _, U_VFI = pp.get_euler_errors(grid, policy_VFI, n_eps=100)
    euler_errors_EGM, _, U_EGM = pp.get_euler_errors(grid, policy_EGM, n_eps=100)
    print(f'Max absolute Euler equation error: \r\n  TI: {max(abs(euler_errors_TI))}      VFI: {max(abs(euler_errors_VFI))}      EGM: {max(abs(euler_errors_EGM))}')
    print(f'The optimal average discounted episode return is: \r\n  TI: {U_TI}      VFI: {U_VFI}      EGM: {U_EGM}')
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(grid, policy_VFI(grid)*pp.cash_on_hand(grid), label='Value function iteration')
    ax[0].plot(grid, policy_TI(grid)*pp.cash_on_hand(grid), linestyle='--', color='orange', label='Time iterations')
    ax[0].plot(grid, policy_EGM(grid)*pp.cash_on_hand(grid), linestyle='--', color='green', label='EGM')
    ax[0].set_TItle('Policy function')
    ax[0].set_xlabel('Capital')
    ax[0].set_ylabel('Consumption')
    ax[1].hist(np.log10(abs(euler_errors_VFI)), density=True, bins=50, label='Value function iteration')
    ax[1].hist(np.log10(abs(euler_errors_TI)), density=True, bins=50, color="orange", alpha=0.5, label='Time iterations')
    ax[1].hist(np.log10(abs(euler_errors_EGM)), density=True, bins=50, color="green", alpha=0.5, label='EGM')
    ax[1].set_TItle('Log10 absolute Euler equation errors')
    ax[1].set_xlabel('Log10 absolute Euler equation error')
    ax[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.legend()
    plt.show()