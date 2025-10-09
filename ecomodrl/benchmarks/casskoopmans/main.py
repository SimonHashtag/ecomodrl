import matplotlib.pyplot as plt
import numpy as np

from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem

if __name__ == "__main__":
    pp = PlanningProblem()
    K_REF = 5.0
    TT_REF = 100

    # Time iterations 
    import time
    start = time.time()
    _, policy_ti = pp.time_iterations(
        grid_points=30, lower_limit=0.1, upper_limit=15., 
        tol=1e-5, conv=0.0001, verbose=False)
    print(f'Time iterations took {time.time()-start} seconds.')

    # VFI
    start = time.time()
    grid, _, policy_vfi = pp.vfi(grid_points=100, lower_limit=0.1, upper_limit=15., 
                        tol=1e-5, conv=0.001, verbose=False)
    print(f'VFI took {time.time()-start} seconds.')
    
    # EGM
    start = time.time()
    grid, policy_egm = pp.egm(grid_points=100, lower_limit=0.1, upper_limit=15., 
                        conv=0.001, verbose=False)
    print(f'EGM took {time.time()-start} seconds.')
    
    # Euler equation errors & return comparison
    euler_errors_ti, _, U_ti = pp.get_euler_errors(grid, policy_ti, n_eps=100)
    euler_errors_vfi, _, U_vfi = pp.get_euler_errors(grid, policy_vfi, n_eps=100)
    euler_errors_egm, _, U_egm = pp.get_euler_errors(grid, policy_egm, n_eps=100)
    print(f'Max absolute Euler equation error: \r\n  TI: {max(abs(euler_errors_ti))}      VFI: {max(abs(euler_errors_vfi))}      EGM: {max(abs(euler_errors_egm))}')
    print(f'The optimal average discounted episode return is: \r\n  TI: {U_ti}      VFI: {U_vfi}      EGM: {U_egm}')
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(grid, policy_vfi(grid)*pp.pre_decision_next_k(grid), label='Value function iteration')
    ax[0].plot(grid, policy_ti(grid)*pp.pre_decision_next_k(grid), linestyle='--', color='orange', label='Time iterations')
    ax[0].plot(grid, policy_egm(grid)*pp.pre_decision_next_k(grid), linestyle='--', color='green', label='EGM')
    ax[0].set_title('Policy function')
    ax[0].set_xlabel('Capital')
    ax[0].set_ylabel('Consumption')
    ax[1].hist(np.log10(abs(euler_errors_vfi)), density=True, bins=50, label='Value function iteration')
    ax[1].hist(np.log10(abs(euler_errors_ti)), density=True, bins=50, color="orange", alpha=0.5, label='Time iterations')
    ax[1].hist(np.log10(abs(euler_errors_egm)), density=True, bins=50, color="green", alpha=0.5, label='EGM')
    ax[1].set_title('Log10 absolute Euler equation errors')
    ax[1].set_xlabel('Log10 absolute Euler equation error')
    ax[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.legend()
    plt.show()