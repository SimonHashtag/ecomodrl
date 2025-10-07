from typing import Union

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

#-------------------------------------------------------------------
# Buffer stock saving planning problem
#-------------------------------------------------------------------
class PlanningProblem():
    def __init__(
        self, 
        σ:float=2,          # Coefficient of relative risk aversion
        β:float=0.95,       # Discount factor
        r:float=0.02,       # Interest reate
        g:float=0.01,       # Growth rate of permanent income
        σ_tran:float=0.1,   # Std. dev. of transitory shock (log)
        σ_perm:float=0.1,   # Std. dev. of permanent shock (log)
        prob:float=0.005,   # Probability of unemployment
        θ_low:float=0.3,    # Earnings in unemployment
    ):
    
        self.σ, self.β = σ, β
        self.R = 1 + r
        self.G = 1 + g
        self.σ_tran, self.σ_perm, self.prob, self.θ_low = σ_tran, σ_perm, prob, θ_low
        self.μ_z = np.log((1 - self.prob*self.θ_low)/(1 - self.prob)) - 0.5*self.σ_tran^2   # Mean of log Z such that E[θ]=1

    def u(self, c):
        '''
        Utility function
        '''
        σ = self.σ

        return (c ** (1 - σ) - 1) / (1 - σ) if σ!= 1 else np.log(c)
    
    def u_prime(self, c):
        'Derivative of utility'
        σ = self.σ

        return c ** (-σ)

    def u_prime_inv(self, c):
        'Inverse of derivative of utility'
        σ = self.σ

        return c ** (-1 / σ)
    
    def draw_shocks(self, n_knots:int):
        gh_x, gh_w = hermgauss(n_knots)
        gh_x *= np.sqrt(2.0)
        gh_w /= np.sqrt(np.pi)

        # Transitory shocks
        theta_continuous = np.exp(self.σ_tran * gh_x + self.μ_z)      # exp(N(mu_z, sigma_transitory^2))
        theta_points = np.concatenate([
            np.full(n_knots, self.θ_low),
            np.repeat(theta_continuous, repeats=n_knots)
        ])

        # Permanent shocks
        psi_continuous = np.exp(self.σ_perm * gh_x - 0.5 * self.σ_perm**2)  # exp(N(-σ^2/2, σ^2))
        psi_points = np.tile(psi_continuous, reps=(n_knots + 1))                   # length (n_points+1)*n_points

        # Combine probabilities for each (θ, ψ) pair
        probs_theta = np.concatenate([
            np.full(n_knots, self.prob),
            (1.0 - self.prob) * np.repeat(gh_w, repeats=n_knots)
        ])
        probs = probs_theta * np.tile(gh_w, reps=(n_knots + 1))

        return (theta_points, psi_points), probs


    def eom(self, a:float, θ_next:np.ndarray, ψ_next:np.ndarray):
        'Equation of motion for normalized state variables with post-decision state'
        R, G = self.R, self.G

        return R/(G*ψ_next) * a + θ_next
    
    def euler_egm(self, c_next, ψ_next, probs):
        return self.u_prime_inv(self.β * self.R/self.G * np.dot(probs, self.u_prime(c_next/ψ_next)))
    
    
    #-------------------------------------------------------------------
    # EGM
    #-------------------------------------------------------------------

    def egm(self, grid_points:int = 1000, n_knots:int=8, lower_limit:float=0.1, upper_limit:float=15., conv:float=0.1, verbose:bool=True):
        """
        Computes optimal consumption policy by endogenous grid method (EGM)
        
        grid_points: number of grid points
        lower_limit: lower limit of grid
        upper_limit: upper limit of grid
        conv: convergence criterion
        verbose: if True, prints convergence information
        
        return: grid and optimal consumption policy as CubicSpline object
        """
        shocks, probs = self.draw_shocks(n_knots)
        θ, ψ = shocks
        eom = self.eom
        euler_egm = self.euler_egm
        
        a_grid = np.linspace(lower_limit, upper_limit, grid_points)
        copt = CubicSpline(np.concatenate(([0.0], a_grid)), np.concatenate(([0.0], a_grid*0.2)), bc_type="natural", extrapolate=True)
        xs = np.empty(grid_points)
        while max(abs(xs-copt(xs)-a)) < conv:
            for i, a in enumerate(a_grid):
                x_next = eom(a, θ, ψ)
                c_next = copt(x_next)
                c = euler_egm(c_next, ψ, probs)
                xs[i] = a + c
            if verbose:
                print(
                    "Convergence: {}.".format(max(abs(xs-copt(xs)-a))),
                    "Convergence criterion: {}.".format(conv),
                )
            copt = CubicSpline(np.concatenate(([0.0], xs)), np.concatenate(([0.0], c)), bc_type="natural", extrapolate=True)
        return xs, copt

    # #-------------------------------------------------------------------
    # # Simulation
    # #-------------------------------------------------------------------
    # def simulation(self, copt, periods=100, start_value=5.0, cons_share=False):
    #     """
    #     Simulate consumption path over number of periods
        
    #     copt: consumption policy as CubicSpline object
    #     periods: number of periods to simulate
    #     start_value: starting value of capital
    #     cons_share: if True, copt returns consumption share of pre-decision capital next period
    #     """
    #     next_k, pre_decision_next_k = self.next_k, self.pre_decision_next_k
    #     u = self.u
    #     β = self.β
        
    #     state = np.empty(periods+1)
    #     state[0] = start_value
    #     cons = np.empty(periods)
    #     U = 0
    #     power = 1.0
    #     for t in range(periods):
    #         if cons_share:
    #             action = copt(state[t])
    #             cons[t] = pre_decision_next_k(state[t])*action
    #         else:
    #             cons[t] = copt(state[t])
    #         state[t+1] = next_k(state[t], cons[t])
    #         U += power * u(cons[t])
    #         power *= β
    #     return cons, state, U
    
    # def get_vfrom_copt(self, grid, copt, conv):
    #     """
    #     Computes value function on grid by iterating on Bellman equation
        
    #     grid: grid on which value function is computed
    #     copt: optimal consumption policy as CubicSpline object
    #     conv: convergence criterion
        
    #     return: value function as CubicSpline object
    #     """
    #     bellman = self.bellman
    #     Vn = np.zeros(len(grid))
    #     V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
    #     while True:
    #         for i in range(len(grid)):
    #             Vn[i] = -bellman(copt(grid[i]), grid[i], V, cons_share=False)
    #         if max(abs(Vn - V(grid))) < conv:
    #             break
    #         V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
    #     return CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)

    # #-------------------------------------------------------------------
    # # Euler equation errors
    # #-------------------------------------------------------------------
    # def get_euler_errors(self, grid, policy, n_eps=100, cons_share=False):
    #     '''
    #     Given a grid of capital values and a consumption policy, computes Euler equation errors
    #     over n_eps periods for each initial capital value in the grid.
    #     '''
    #     calculate_euler_errors = self.calculate_euler_errors
    #     grid_len = len(grid)
    #     states = np.empty((grid_len, n_eps+1), dtype=float)
    #     euler_errors = np.empty((grid_len, n_eps-1), dtype=float)
    #     avg_return = 0
    #     for i, k in enumerate(grid):
    #         cons_path, state_path, U = self.simulation(policy, n_eps, start_value=k, cons_share=cons_share)
    #         states[i,:] = state_path
    #         euler_errors[i,:] = calculate_euler_errors(cons_path, state_path)
    #         avg_return += U/grid_len
    #     return euler_errors.ravel(), states.ravel(), avg_return