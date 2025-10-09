from typing import Union

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

#-------------------------------------------------------------------
# Cass-Koopmans planning problem
#-------------------------------------------------------------------
class PlanningProblem():
    def __init__(
        self, 
        σ:float=2,      # Coefficient of relative risk aversion
        β:float=0.95,   # Discount factor
        δ:float=0.02,   # Depreciation rate on capital
        α:float=0.33,   # Return to capital per capita
        A:int=1):       # Technology

        self.σ, self.β = σ, β
        self.δ, self.α, self.A = δ, α, A

    def u(self, c):
        '''
        Utility function
        '''
        σ = self.σ

        return c ** (1 - σ) / (1 - σ) if σ!= 1 else np.log(c)

    def u_prime(self, c):
        'Derivative of utility'
        σ = self.σ

        return c ** (-σ)

    def u_prime_inv(self, c):
        'Inverse of derivative of utility'
        σ = self.σ

        return c ** (-1 / σ)

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
    
    def pre_decision_next_k(self, k):
        ''''
        Given the current capital Kt, computes Kt+1 by state transition law
        assuming no consumption (for upper bound of consumption)
        '''
        δ = self.δ
        f = self.f

        k_next = f(k) + (1 - δ) * k

        return k_next
        
    def next_k(self, k, c):
        ''''
        Given the current capital Kt and an arbitrary feasible
        consumption choice Ct, computes Kt+1 by state transition law.
        '''
        k_next = self.f(k) + (1 - self.δ) * k - c
        
        return k_next

    def next_k_c(self, k, c):
        ''''
        Given the current capital Kt and an arbitrary feasible
        consumption choice Ct, computes Kt+1 by state transition law
        and optimal Ct+1 by Euler equation.
        '''
        β, δ = self.β, self.δ
        u_prime, u_prime_inv = self.u_prime, self.u_prime_inv
        f_prime = self.f_prime

        k_next = self.next_k(k, c)
        c_next = u_prime_inv(u_prime(c) / (β * (f_prime(k_next) + (1 - δ))))

        return k_next, c_next
    
    def bellman(self, c, k, value):
        'Bellman equation for c = consumption share'
        c = c * self.pre_decision_next_k(k)
        new_k = self.next_k(k, c)
        if new_k<0:
            return np.inf
        else:
            return -(self.u(c) + self.β * value(new_k))
    
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
    
    def euler_egm(self, c_next, k_next):
        β, δ = self.β, self.δ
        u_prime = self.u_prime
        f_prime = self.f_prime
        u_prime_inv = self.u_prime_inv
        
        return u_prime_inv(β*u_prime(c_next)*(f_prime(k_next)+(1-δ)))
    
    def calculate_euler_errors(self, c_path, k_path):
        'Calculate Euler equation errors given paths of consumption and capital'
        euler_errors = self.β * (c_path[:-1]/c_path[1:])**(self.σ) * (self.f_prime(k_path[1:-1]) + (1 - self.δ)) - 1
        
        return euler_errors
    
    #-------------------------------------------------------------------
    # Value function iteration
    #-------------------------------------------------------------------

    def vfi(self, grid_points: int = 1000, lower_limit: float = 0.1, upper_limit: float= 15., tol: float = 1e-5, conv: float = 0.1, verbose: bool=True):
        """ 
        Computes optimal consumption policy by value function iteration
        grid_points: number of grid points
        lower_limit: lower limit of grid
        upper_limit: upper limit of grid
        tol: tolerance level for minimization
        conv: convergence criterion
        verbose: if True, prints convergence information
        
        return: grid, value function and optimal consumption policy as CubicSpline objects"""
        grid = np.linspace(lower_limit, upper_limit, grid_points)
        V = CubicSpline(grid, np.zeros(grid_points), bc_type="natural", extrapolate=True)
        Vn = np.zeros(grid_points)
        copt = np.linspace(0.001, 1, grid_points)
        
        while True:
            for i in range(0, grid_points):
                obj = minimize(
                    self.bellman,
                    x0 = copt[i],
                    bounds=((0.001, 1.0),),
                    args=(
                        grid[i],
                        V,
                    ),
                    method="Powell",
                    tol=tol,
                )
                Vn[i] = -obj.fun
                copt[i] = obj.x[0]
                if abs(obj.x - 0.0001) < 0.0001 or abs(obj.x - grid[i]) < 0.00001:
                    print(f"Corner point solution! State: {grid[i]}; Action: {obj.x[0]}")
            if max(abs(Vn - V(grid))) < conv:
                return grid, V, CubicSpline(grid, copt, bc_type="natural", extrapolate=True)
            if verbose:
                print(
                    "Convergence: {}.".format(max(abs(Vn - V(grid)))),
                    "Convergence criterion: {}.".format(conv),
                )
            V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)

    #-------------------------------------------------------------------
    # Time iterations
    #-------------------------------------------------------------------

    def time_iterations(self, grid_points:int = 1000, lower_limit:float=0.1, upper_limit:float=15., tol:float=1e-5, conv:float=0.1, verbose:bool=True) -> Union[np.ndarray, CubicSpline]:
        """ 
        Computes optimal consumption policy by time iterations
        
        grid_points: number of grid points
        lower_limit: lower limit of grid
        upper_limit: upper limit of grid
        tol: tolerance level for minimization
        conv: convergence criterion
        verbose: if True, prints convergence information
        
        return: grid and optimal consumption policy as CubicSpline object
        """
        grid = np.linspace(lower_limit, upper_limit, grid_points)
        sq_euler = np.zeros(grid_points)
        copt = CubicSpline(grid, grid*0.2, bc_type="natural", extrapolate=True)
        cs = copt(grid)
        
        while True:
            for i in range(0, grid_points):
                obj = minimize(
                    self.sq_euler_error,
                    x0 = cs[i],
                    bounds=((0.001, self.pre_decision_next_k(grid[i])-0.001),),
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
            if verbose:
                print(
                    "Convergence: {}.".format(max(abs(copt(grid)-cs))),
                    "Absolute Euler Error: {}.".format(max(np.sqrt(sq_euler))),
                    "Convergence criterion: {}.".format(conv),
                )
            if max(abs(copt(grid)-cs)) < conv:
                return grid, CubicSpline(grid, cs/self.pre_decision_next_k(grid), bc_type="natural", extrapolate=True)
            copt = CubicSpline(grid, cs, bc_type="natural", extrapolate=True)
        

    #-------------------------------------------------------------------
    # EGM
    #-------------------------------------------------------------------

    def egm(self, grid_points:int = 1000, lower_limit:float=0.1, upper_limit:float=15., conv:float=0.1, verbose:bool=True):
        """
        Computes optimal consumption policy by endogenous grid method (EGM)
        
        grid_points: number of grid points
        lower_limit: lower limit of grid
        upper_limit: upper limit of grid
        conv: convergence criterion
        verbose: if True, prints convergence information
        
        return: grid and optimal consumption policy as CubicSpline object
        """
        cash_on_hand = self.pre_decision_next_k
        euler_egm = self.euler_egm
        
        k_next_grid = np.linspace(lower_limit, upper_limit, grid_points)
        sopt = CubicSpline(np.concatenate(([0.0], k_next_grid)), np.concatenate(([0.0], k_next_grid*0.2)), bc_type="natural", extrapolate=True)
        ms = np.empty(grid_points)
        while True:
            for i, k_next in enumerate(k_next_grid):
                m_next = cash_on_hand(k_next)
                c_next = m_next - sopt(m_next)
                c = euler_egm(c_next, k_next)
                ms[i] = c + k_next
            if verbose:
                print(
                    "Convergence: {}.".format(max(abs(sopt(ms)-k_next_grid))),
                    "Convergence criterion: {}.".format(conv),
                )
            if max(abs(sopt(ms)-k_next_grid)) < conv:
                c_of_m = CubicSpline(np.concatenate(([0.0], ms)), np.concatenate(([0.0], ms-k_next_grid)), bc_type="natural", extrapolate=True)
                m_grid = cash_on_hand(k_next_grid)
                copt = CubicSpline(k_next_grid, c_of_m(m_grid)/m_grid, bc_type="natural", extrapolate=True) 
                return k_next_grid, copt 
            sopt = CubicSpline(np.concatenate(([0.0], ms)), np.concatenate(([0.0], k_next_grid)), bc_type="natural", extrapolate=True)
        

    #-------------------------------------------------------------------
    # Simulation
    #-------------------------------------------------------------------
    def simulation(self, copt, periods=100, start_value=5.0):
        """
        Simulate consumption path over number of periods
        
        copt: consumption policy as CubicSpline object
        periods: number of periods to simulate
        start_value: starting value of capital
        """
        next_k, pre_decision_next_k = self.next_k, self.pre_decision_next_k
        u = self.u
        β = self.β
        
        state = np.empty(periods+1)
        state[0] = start_value
        cons = np.empty(periods)
        U = 0
        power = 1.0
        for t in range(periods):
            action = copt(state[t])
            cons[t] = pre_decision_next_k(state[t])*action
            state[t+1] = next_k(state[t], cons[t])
            U += power * u(cons[t])
            power *= β
        return cons, state, U
    
    def get_vfrom_copt(self, grid, copt, conv):
        """
        Computes value function on grid by iterating on Bellman equation
        
        grid: grid on which value function is computed
        copt: optimal consumption policy as CubicSpline object
        conv: convergence criterion
        
        return: value function as CubicSpline object
        """
        bellman = self.bellman
        Vn = np.zeros(len(grid))
        V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
        while True:
            for i in range(len(grid)):
                Vn[i] = -bellman(copt(grid[i]), grid[i], V)
            if max(abs(Vn - V(grid))) < conv:
                break
            V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
        return CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)

    #-------------------------------------------------------------------
    # Euler equation errors
    #-------------------------------------------------------------------
    def get_euler_errors(self, grid, policy, n_eps=100):
        '''
        Given a grid of capital values and a consumption policy, computes Euler equation errors
        over n_eps periods for each initial capital value in the grid.
        '''
        calculate_euler_errors = self.calculate_euler_errors
        grid_len = len(grid)
        states = np.empty((grid_len, n_eps+1), dtype=float)
        euler_errors = np.empty((grid_len, n_eps-1), dtype=float)
        avg_return = 0
        for i, k in enumerate(grid):
            cons_path, state_path, U = self.simulation(policy, n_eps, start_value=k)
            states[i,:] = state_path
            euler_errors[i,:] = calculate_euler_errors(cons_path, state_path)
            avg_return += U/grid_len
        return euler_errors.ravel(), states.ravel(), avg_return