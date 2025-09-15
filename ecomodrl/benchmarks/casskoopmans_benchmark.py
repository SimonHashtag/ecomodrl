# Source of shooting algorithm (changes were made): https://python.quantecon.org/cass_koopmans_1.html#equation-feasibility-constraint
import time
import matplotlib.pyplot as plt
from numba import njit, float64
from numba.experimental import jitclass
import numpy as np
from typing import Union
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize

# %%

planning_data = [
    ('γ', float64),    # Coefficient of relative risk aversion
    ('β', float64),    # Discount factor
    ('δ', float64),    # Depreciation rate on capital
    ('α', float64),    # Return to capital per capita
    ('A', float64)     # Technology
]

@jitclass(planning_data)
class PlanningProblem():

    def __init__(self, γ=2, β=0.95, δ=0.02, α=0.33, A=1):

        self.γ, self.β = γ, β
        self.δ, self.α, self.A = δ, α, A

    def u(self, c):
        '''
        Utility function
        ASIDE: If you have a utility function that is hard to solve by hand
        you can use automatic or symbolic differentiation
        See https://github.com/HIPS/autograd
        '''
        γ = self.γ

        return c ** (1 - γ) / (1 - γ) if γ!= 1 else np.log(c)

    def u_prime(self, c):
        'Derivative of utility'
        γ = self.γ

        return c ** (-γ)

    def u_prime_inv(self, c):
        'Inverse of derivative of utility'
        γ = self.γ

        return c ** (-1 / γ)

    def f(self, k):
        'Production function'
        α, A = self.α, self.A

        return A * k ** α

    def f_prime(self, k):
        'Derivative of production function'
        α, A = self.α, self.A

        return α * A * k ** (α - 1)

    def f_prime_inv(self, k):
        'Inverse of derivative of production function'
        α, A = self.α, self.A

        return (k / (A * α)) ** (1 / (α - 1))

    def next_k_c(self, k, c):
        ''''
        Given the current capital Kt and an arbitrary feasible
        consumption choice Ct, computes Kt+1 by state transition law
        and optimal Ct+1 by Euler equation.
        '''
        β, δ = self.β, self.δ
        u_prime, u_prime_inv = self.u_prime, self.u_prime_inv
        f, f_prime = self.f, self.f_prime

        k_next = f(k) + (1 - δ) * k - c
        c_next = u_prime_inv(u_prime(c) / (β * (f_prime(k_next) + (1 - δ))))

        return k_next, c_next

@njit
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

@njit
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

@njit
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


def calculate_euler_errors(pp, c_path, k_path):
    'Calculate Euler equation errors given paths of consumption and capital'
    euler_errors = pp.β * (c_path[:-1]/c_path[1:])**(pp.γ) * (pp.f_prime(k_path[1:-1]) + (1 - pp.δ)) - 1
    
    return euler_errors

## Dynamic Programming (VFI)
class problem():
    def __init__(self, γ=2, β=0.95, δ=0.02, α=0.33, A=1):
        self.γ, self.β = γ, β
        self.δ, self.α, self.A = δ, α, A
    
    def u(self, c):
        '''
        Utility function
        '''
        γ = self.γ

        return c ** (1 - γ) / (1 - γ) if γ!= 1 else np.log(c)
    
    def u_prime(self, c):
        'Derivative of utility'
        γ = self.γ

        return c ** (-γ)
    
    def f(self, k):
        'Production function'
        α, A = self.α, self.A

        return A * k ** α
    
    def f_prime(self, k):
        'Derivative of production function'
        α, A = self.α, self.A

        return α * A * k ** (α - 1)
    
    def next_k(self, k, c):
        ''''
        Given the current capital Kt and an arbitrary feasible
        consumption choice Ct, computes Kt+1 by state transition law.
        '''
        δ = self.δ
        f = self.f

        k_next = f(k) + (1 - δ) * k - c
        
        return k_next
    
    def bellman(self, c, k, value, cons_share=True):
        'Bellman equation for c = consumption share'
        if cons_share:
            c = c * k
        new_k = self.next_k(k, c)
        if new_k<0:
            return np.inf
        else:
            return -(self.u(c) + self.β * value(new_k))
    
    def pre_decision_next_k(self, k):
        ''''
        Given the current capital Kt, computes Kt+1 by state transition law
        assuming no consumption (for upper bound of consumption)
        '''
        δ = self.δ
        f = self.f

        k_next = f(k) + (1 - δ) * k

        return k_next
    
    def sq_euler_error(self, c, k, policy):
        '''
        Given current consumption and capital, computes the Euler equation error
        '''
        β, δ = self.β, self.δ
        u_prime = self.u_prime
        f_prime = self.f_prime

        k_next = self.next_k(k, c)
        if k_next < 0:
            return np.inf
        c_next = policy(k_next)

        return (β * u_prime(c_next/c) * (f_prime(k_next) + (1 - δ)) - 1)**2
        
        
def vfi(problem, grid_points: int = 1000, lower_limit = 0.1, upper_limit= 15., tolerance: float = 0.25, conv: float = 0.1, verbose=True):
    grid = np.linspace(lower_limit, upper_limit, grid_points)
    V = CubicSpline(grid, np.zeros(grid_points), bc_type="natural", extrapolate=True)
    Vn = np.zeros(grid_points)
    copt = np.linspace(0.001, 1, grid_points)
    
    while True:
        for i in range(0, grid_points):
            obj = minimize(
                problem.bellman,
                x0 = copt[i],
                bounds=((0.001, 1.0),),
                args=(
                    grid[i],
                    V,
                ),
                method="Powell",
                tol=tolerance,
            )
            Vn[i] = -obj.fun
            copt[i] = obj.x[0]
            if abs(obj.x - 0.0001) < 0.0001 or abs(obj.x - grid[i]) < 0.00001:
                print(f"Corner point solution! State: {grid[i]}; Action: {obj.x[0]}")
        if max(abs(Vn - V(grid))) < conv:
            break
        if verbose:
            print(
                "Convergence: {}.".format(max(abs(Vn - V(grid)))),
                "Convergence criterion: {}.".format(conv),
            )
        V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
    return grid, Vn, CubicSpline(grid, copt, bc_type="natural", extrapolate=True)

def time_iterations(problem, grid_points=1000, lower_limit=0.1, upper_limit=15., tol=0.25, conv=0.1, verbose=True):
    grid = np.linspace(lower_limit, upper_limit, grid_points)
    sq_euler = np.zeros(grid_points)
    copt = CubicSpline(grid, grid*0.2, bc_type="natural", extrapolate=True)
    cs = copt(grid)
    
    while True:
        for i in range(0, grid_points):
            obj = minimize(
                problem.sq_euler_error,
                x0 = cs[i],
                bounds=((0.001, problem.pre_decision_next_k(grid[i])-0.001),),
                args=(
                    grid[i],
                    copt,
                ),
                method="Powell",
                tol=tol,
            )
            sq_euler[i] = obj.fun
            cs[i] = obj.x[0]
            if abs(obj.x - 0.0001) < 0.0001 or abs(obj.x - grid[i]) < 0.00001:
                print(f"Corner point solution! State: {grid[i]}; Action: {obj.x[0]}")
        if max(abs(copt(grid)-cs)) < conv:
            break
        if verbose:
            print(
                "Convergence: {}.".format(max(abs(copt(grid)-cs))),
                "Absolute Euler Error: {}.".format(max(np.sqrt(sq_euler))),
                "Convergence criterion: {}.".format(conv),
            )
        
        copt = CubicSpline(grid, cs, bc_type="natural", extrapolate=True)
    return grid, copt

def get_vfrom_copt(problem, grid, copt, conv):
    Vn = np.zeros(len(grid))
    V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
    while True:
        for i in range(len(grid)):
            Vn[i] = -problem.bellman(copt(grid[i]), grid[i], V, cons_share=False)
        if max(abs(Vn - V(grid))) < conv:
            break
        V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
    return CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)

def simulation_cfrac(problem, copt, periods=100, start_value=5.0):
    """Simulate consumption path over number of periods"""
    state = np.empty(periods+1)
    state[0] = start_value
    cons = np.empty(periods)
    U = 0
    power = 1.0
    for t in range(periods):
        action = copt(state[t])
        cons[t] = state[t]*action
        state[t+1] = problem.next_k(state[t], cons[t])
        U += power * problem.u(cons[t])
        power *= problem.β
    return cons, state, U

def simulation(problem, copt, periods=100, start_value=5.0):
    """Simulate consumption path over number of periods"""
    state = np.empty(periods+1)
    state[0] = start_value
    cons = np.empty(periods)
    U = 0
    power = 1.0
    for t in range(periods):
        cons[t] = copt(state[t])
        state[t+1] = problem.next_k(state[t], cons[t])
        U += power * problem.u(cons[t])
        power *= problem.β
    return cons, state, U

if __name__ == "__main__":
    # Bisection search and shooting method
    pp = PlanningProblem()

    ρ = 1 / pp.β - 1
    k_ss = pp.f_prime_inv(ρ+pp.δ)
    s_ss = pp.δ * k_ss / pp.f(k_ss)
    c_ss = pp.f(k_ss)*(1-s_ss)
    start_time = time.time()
    c_path, k_path, s_path, μ_path = calculate_paths(pp, 1.0, 5.0, 100, k_ter=k_ss)
    elapsed_time = time.time() - start_time
    # Compute optimal discounted episode return
    U = sum(np.power(pp.β, range(0, len(c_path))) *  pp.u(c_path))
    # Euler equation errors
    euler_errors = calculate_euler_errors(pp, c_path, k_path)
    plt.hist(euler_errors, bins=50)
    plt.title('Euler equation errors')
    plt.show()
    print(f'Bisection method took {elapsed_time} seconds to compute the paths.')
    print(f'max Euler equation error is: {max(abs(euler_errors))}')
    print(f'steady state for capital is: {k_ss}')
    print(f'the steady state savings rate is: {s_ss}')
    print(f'steady state consumption is: {c_ss}')
    print(f'the optimal discounted episode return is: {U}')
    copt = CubicSpline(k_path[0:101], c_path, bc_type="natural", extrapolate=True)
    plt.plot(k_path[0:101], copt(k_path[0:101]))
    plt.show()

    plot_paths(pp, 1.0, 5.0, 100, k_ter=k_ss, c_ss=c_ss, k_ss=k_ss, s_ss=s_ss)
    plt.show()

    # Time iterations 
    p = problem()
    start_time = time.time()
    grid, c_optimal = time_iterations(
        p, grid_points=30, lower_limit=0.1, upper_limit=15., 
        tol=1e-5, conv=0.0001, verbose=False)
    V = get_vfrom_copt(p, grid, c_optimal, conv=0.001)
    elapsed_time = time.time() - start_time
    cons_path, state_path, U = simulation(p, c_optimal, 100)
    plt.plot(grid, c_optimal(grid))
    plt.show()
    # Euler equation errors
    euler_errors = calculate_euler_errors(p, cons_path, state_path)
    plt.hist(euler_errors, bins=50)
    plt.title('Euler equation errors')
    plt.show()
    print(f'Time iteration took {elapsed_time} seconds to compute the policy function.')
    print(f'max Euler equation error is: {max(abs(euler_errors))}')
    print(f'the optimal discounted episode return is: {U}')

    # VFI
    start_time = time.time()
    grid, V, copt = vfi(p, grid_points=100, 
                        conv=0.001, tolerance=1e-5, verbose=False)
    elapsed_time = time.time() - start_time
    cons_path, state_path, U = simulation_cfrac(p, copt, 100)
    plt.plot(grid, copt(grid)*grid)
    plt.show()
    euler_errors = calculate_euler_errors(p, cons_path, state_path)
    plt.hist(euler_errors, bins=50)
    plt.title('Euler equation errors')
    plt.show()
    print(f'Value function iteration took {elapsed_time} seconds to compute the policy function.')
    print(f'max Euler equation error is: {max(abs(euler_errors))}')
    print(f'the optimal discounted episode return is: {U}')
