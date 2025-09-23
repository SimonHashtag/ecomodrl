import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem

#-------------------------------------------------------------------
# Shooting and bisection methods
# Source of shooting algorithm (changes were made): https://python.quantecon.org/cass_koopmans_1.html#equation-feasibility-constraint
#-------------------------------------------------------------------
def shooting(pp, c0, k0, T=10):
    '''
    Given the initial condition of capital k0 and an initial guess
    of consumption c0, computes the whole paths of c and k
    using the state transition law and Euler equation for T periods.
    '''
    if c0 > pp.f(k0):
        print("initial consumption is not feasible")

        return None

    # initialize vectors of c and k
    c_vec = np.empty(T+1)
    k_vec = np.empty(T+2)

    c_vec[0] = c0
    k_vec[0] = k0

    for t in range(T):
        k_vec[t+1], c_vec[t+1] = pp.next_k_c(k_vec[t], c_vec[t])

    k_vec[T+1] = pp.f(k_vec[T]) + (1 - pp.δ) * k_vec[T] - c_vec[T]

    return c_vec, k_vec

def bisection(pp, c0, k0, T=10, tol=1e-4, max_iter=500, k_ter=0, verbose=True):
    '''
    Implementation of bisection method. 
    1. Make an initial guess of consumption c0 (between 0 and f(k0)). 
    2. Compute path via shooting algorithm. 
    3. If k_{t+1}>0: make c0 lower bound of next guess; If k_{t+1}<0: make c0 upper bound of next guess
    4. Repeat until k_{t+1}~0
    '''
    # initial boundaries for guess c0
    c0_upper = pp.f(k0)
    c0_lower = 0

    i = 0
    while True:
        c_vec, k_vec = shooting(pp, c0, k0, T)
        error = k_vec[-1] - k_ter

        # check if the terminal condition is satisfied
        if np.abs(error) < tol:
            if verbose:
                print('Converged successfully on iteration ', i+1)
            return c_vec, k_vec

        i += 1
        if i == max_iter:
            if verbose:
                print('Convergence failed.')
            return c_vec, k_vec

        # if iteration continues, updates boundaries and guess of c0
        if error > 0:
            c0_lower = c0
        else:
            c0_upper = c0

        c0 = (c0_lower + c0_upper) / 2

def saving_rate(pp, c_path, k_path):
    'Given paths of c and k, computes the path of saving rate.'
    production = pp.f(k_path[:-1])

    return (production - c_path) / production

def calculate_paths(pp, c0, k0, T, k_ter=0):
    'Calculate paths of consumption, capital, lagrange multiplier and savings rate by bisection method'
    c_vec, k_vec = bisection(pp, c0, k0, T, k_ter=k_ter, verbose=False)
    μ_vec = pp.u_prime(c_vec)
    s_vec = saving_rate(pp, c_vec, k_vec)
    
    return c_vec, k_vec, s_vec, μ_vec

def plot_paths(pp, c0, k0, T, k_ter=0, c_ss=None, k_ss=None, s_ss=None):
    'Plot paths of consumption, capital, lagrange multiplier and savings rate'
    
    c_vec, k_vec, s_vec, μ_vec = calculate_paths(pp, c0, k0, T, k_ter)
    
    _, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs = axs.flatten()
    ylabels = ['$c_t$', '$k_t$', '$s_t$', '$mu_t$']
    titles = ['Consumption', 'Capital', 'Saving rate', 'Lagrange Multiplier']
    
    paths = [c_vec, k_vec, s_vec, μ_vec]
    for i in range(4):
        axs[i].plot(paths[i], color="#333F48")
        axs[i].set(xlabel='t', ylabel=ylabels[i], title=titles[i])
        
        # Plot steady state value of consumption rate
        if c_ss is not None:
            axs[0].hlines(c_ss, 0, np.max(T), linestyle='--', color="#333F48")
        
        # Plot steady state value of capital
        if k_ss is not None:
            axs[1].axhline(k_ss, c='k', ls='--', lw=1, color="#333F48")
        
        # Plot steady state value of savings rate
        if s_ss is not None:
            axs[2].hlines(s_ss, 0, np.max(T), linestyle='--', color="#333F48")

        axs[1].axvline(T+1, c='k', ls='--', lw=1, color="#333F48")
        axs[1].scatter(T+1, paths[1][-1], s=80, color="#AF0078")
        
if __name__ == "__main__":
    pp = PlanningProblem()
    K_REF = 5.0
    TT_REF = 100
    # Bisection search and shooting method
    ρ = 1 / pp.β - 1
    k_ss = pp.f_prime_inv(ρ+pp.δ)
    s_ss = pp.δ * k_ss / pp.f(k_ss)
    c_ss = pp.f(k_ss)*(1-s_ss)
    start_time = time.time()
    c_path, k_path, s_path, μ_path = calculate_paths(pp, 1.0, K_REF, TT_REF, k_ter=k_ss)
    elapsed_time = time.time() - start_time
    # Compute optimal discounted episode return
    U = sum(np.power(pp.β, range(0, len(c_path))) *  pp.u(c_path))
    # Euler equation errors
    euler_errors = pp.calculate_euler_errors(c_path, k_path)
    euler_errors[euler_errors==0.0] += abs(euler_errors[euler_errors>0]).min()
    plt.hist(np.log10(abs(euler_errors)), bins=50)
    plt.title('Log10 absolute Euler equation errors')
    plt.show()
    print(f'Bisection method took {elapsed_time} seconds to compute the paths.')
    print(f'max absolute Euler equation error is: {max(abs(euler_errors))}')
    print(f'steady state for capital is: {k_ss}')
    print(f'the steady state savings rate is: {s_ss}')
    print(f'steady state consumption is: {c_ss}')
    print(f'the optimal discounted episode return is: {U}')
    copt = CubicSpline(k_path[0:101], c_path, bc_type="natural", extrapolate=True)
    plt.plot(k_path[0:101], copt(k_path[0:101]))
    plt.show()

    plot_paths(pp, 1.0, K_REF, TT_REF, k_ter=k_ss, c_ss=c_ss, k_ss=k_ss, s_ss=s_ss)
    plt.show()