# Source (changes were made): https://python.quantecon.org/cass_koopmans_1.html#equation-feasibility-constraint

import matplotlib.pyplot as plt
from numba import njit, float64
from numba.experimental import jitclass
import numpy as np
from typing import Union
from scipy.interpolate import interp1d
from scipy.optimize import minimize

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

pp = PlanningProblem()

ρ = 1 / pp.β - 1
k_ss = pp.f_prime_inv(ρ+pp.δ)
s_ss = pp.δ * k_ss / pp.f(k_ss)
c_ss = pp.f(k_ss)*(1-s_ss)

c_path, k_path, s_path, μ_path = calculate_paths(pp, 1.0, 5.0, 100, k_ter=k_ss)

U = sum(pp.u(c_path))
print(f'steady state for capital is: {k_ss}')
print(f'the steady state savings rate is: {s_ss}')
print(f'steady state consumption is: {c_ss}')
print(f'the optimal episode return is: {U}')

plot_paths(pp, 1.0, 5.0, 100, k_ter=k_ss, c_ss=c_ss, k_ss=k_ss, s_ss=s_ss)
plt.show()

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
    
    def f(self, k):
        'Production function'
        α, A = self.α, self.A

        return A * k ** α
    
    def eom(self, c, k):
        'Equation of motion'
        new_k = self.f(k) + (1-self.δ) * k - c * k
        
        return new_k
    
    def bellman(self, c, k, value):
        new_k = self.eom(c, k)
        if any(new_k)<0:
            return np.inf
        else:
            return -(self.u(c*k) + self.β * value(new_k))
        
        
def vfi(problem, tolerance: float = 0.25, conv: float = 0.1, grid_points: int = 1000, lower_limit = 0.1, upper_limit= 15., verbose=True):
    grid = np.linspace(lower_limit, upper_limit, grid_points)
    V = interp1d(grid, np.zeros(grid_points), fill_value="extrapolate")
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
                method="nelder-mead",
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
        V = interp1d(grid, Vn, fill_value="extrapolate")
    return grid, Vn, copt

def simulation(problem, grid, copt, periods=100, start_value=5.0):
    """Simulate consumption path over number of periods"""
    policy = interp1d(grid, copt, fill_value="extrapolate")
    state = []
    action = []
    U = 0
    state.append(start_value)
    for t in range(periods):
        action.append(policy(state[t]))
        state.append(problem.eom(action[t], state[t]))
        U += problem.u(state[t]*action[t])
    return action, state, U

p = problem()

grid, V, copt = vfi(p, conv=0.01, tolerance=0.01, verbose=False)
action_path, state_path, U = simulation(p, grid, copt, 100)
print(f'the optimal episode return is: {U}')