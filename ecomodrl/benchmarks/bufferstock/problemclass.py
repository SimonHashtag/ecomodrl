from typing import Union

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import PchipInterpolator  # monotone, linear extrapolation
from scipy.optimize import brentq

#-------------------------------------------------------------------
# Buffer stock saving planning problem
#-------------------------------------------------------------------
class PlanningProblem():
    def __init__(
        self, 
        σ:float=2.,          # Coefficient of relative risk aversion
        β:float=0.95,       # Discount factor
        r:float=0.02,       # Interest reate
        g:float=0.01,       # Growth rate of permanent income
        σ_tran:float=0.1,   # Std. dev. of transitory shock (log)
        σ_perm:float=0.1,   # Std. dev. of permanent shock (log)
        prob:float=0.005,   # Probability of unemployment
        θ_low:float=0.3,    # Earnings in unemployment
    ):
    
        self.σ, self.β = σ, β
        self.R = 1.0 + r
        self.G = 1.0 + g
        self.σ_tran, self.σ_perm, self.prob, self.θ_low = σ_tran, σ_perm, prob, θ_low
        self.μ_z = np.log((1 - self.prob*self.θ_low)/(1 - self.prob)) - 0.5*self.σ_tran**2   # Mean of log Z such that E[θ]=1

    def u(self, c):
        '''
        Utility function
        '''
        σ = self.σ

        return (c ** (1 - σ)) / (1 - σ) if σ!= 1 else np.log(c)
    
    def u_prime(self, c):
        'Derivative of utility'
        σ = self.σ

        return c ** (-σ)

    def u_prime_inv(self, c):
        'Inverse of derivative of utility'
        σ = self.σ

        return c ** (-1 / σ)
    
    def bellman(self, action, state, value):
        'Bellman equation for c = consumption share'
        consumption = state * action
        a = state * (1.0 - action)
        new_state = self.eom(a)
        if new_state<0:
            return np.inf
        else:
            return self.u(consumption) + self.β * value(new_state)
    
    def set_gh(self, n_knots:int):
        gh_x, gh_w = hermgauss(n_knots)
        gh_x *= np.sqrt(2.0)
        gh_w /= np.sqrt(np.pi)

        # Transitory shocks
        theta_continuous = np.exp(self.σ_tran * gh_x + self.μ_z)      # exp(N(mu_z, sigma_transitory^2))
        self.θ = np.concatenate([
            np.full(n_knots, self.θ_low),
            np.repeat(theta_continuous, repeats=n_knots)
        ])

        # Permanent shocks
        psi_continuous = np.exp(self.σ_perm * gh_x - 0.5 * self.σ_perm**2)  # exp(N(-σ^2/2, σ^2))
        self.ψ = np.tile(psi_continuous, reps=(n_knots + 1))                   # length (n_points+1)*n_points

        # Combine probabilities for each (θ, ψ) pair
        probs_theta = np.concatenate([
            np.full(n_knots, self.prob),
            (1.0 - self.prob) * np.repeat(gh_w, repeats=n_knots)
        ])
        self.probs = probs_theta * np.tile(gh_w, reps=(n_knots + 1))

    def draw_shocks(self):
        if np.random.uniform(0., 1.) < self.prob:
            θ = self.θ_low
        else:
            θ = np.exp(np.random.normal(self.μ_z, self.σ_tran))
        ψ = np.exp(np.random.normal(-0.5*self.σ_perm**2, self.σ_perm))
        return θ, ψ

    def eom(self, a:float, θ:Union[None, float, np.ndarray]=None, ψ:Union[None, float, np.ndarray]=None):
        'Equation of motion for normalized state variables with post-decision state'
        R, G = self.R, self.G
        θ = self.θ if θ is None else θ
        ψ = self.ψ if ψ is None else ψ

        return R/(G*ψ) * a + θ
    
    def euler_egm(self, c_next):
        return self.u_prime_inv(self.β * self.R/self.G * np.dot(self.probs, self.u_prime(c_next/self.ψ)))
    
    def euler_residual(self, c_share:float, x:float, policy:PchipInterpolator):
        eom = self.eom
        u_prime = self.u_prime

        a = x * (1.0-c_share)
        x_next = eom(a)
        c_next = x_next * policy(x_next)

        rhs = self.β * self.R/self.G * np.dot(self.probs, u_prime(c_next/self.ψ))
        return u_prime(x*c_share) - rhs
    
    def calculate_euler_errors(self, c_path, x_path, policy):
        'Calculate Euler equation errors given paths of consumption and wealth and a policy'
        euler_residual = self.euler_residual
        euler_errors = np.empty(len(c_path))
        for i, c, x in zip(range(0, len(c_path)), c_path, x_path[:-1]):
            if 1-c/x < 10e-3:
                euler_errors[i] = None
            else:
                euler_errors[i] = np.log10(max(1e-18, abs(euler_residual(c/x, x, policy))))
        return euler_errors
    
    #-------------------------------------------------------------------
    # EGM
    #-------------------------------------------------------------------

    def egm(self, 
            grid_points:int = 1000, 
            n_knots:int=8, 
            lower_limit:float=0.1, 
            upper_limit:float=15., 
            conv:float=0.1,
            verbose:bool=True) -> Union[np.ndarray, PchipInterpolator]:
        """
        Computes optimal consumption policy by endogenous grid method (EGM)
        
        grid_points: number of grid points
        n_knots: number of Gauss Hermite points
        lower_limit: lower limit of grid
        upper_limit: upper limit of grid
        conv: convergence criterion
        verbose: if True, prints convergence information
        
        return: grid and optimal consumption policy as CubicSpline object
        """
        self.set_gh(n_knots)
        eom = self.eom
        euler_egm = self.euler_egm
        
        a_grid = np.linspace(lower_limit, upper_limit, grid_points)
        copt = PchipInterpolator(np.concatenate(([0.0], a_grid)), np.concatenate(([1.0], np.ones(grid_points)*0.1)), extrapolate=True)
        copt_old = PchipInterpolator(np.concatenate(([0.0], a_grid)), np.concatenate(([1.0], np.ones(grid_points)*0.2)), extrapolate=True)
        xs = a_grid.copy()
        cs = np.empty(grid_points)
        while max(abs(copt_old(xs)-copt(xs))) > conv:
            copt_old = copt
            for i, a in enumerate(a_grid):
                xs_next = eom(a)
                cs_next = copt(xs_next) * xs_next
                cs[i] = euler_egm(cs_next)
                xs[i] = a + cs[i]
            order = np.argsort(xs)
            xs_sorted, cs_sorted = xs[order], cs[order]
            copt = PchipInterpolator(np.concatenate(([0.0], xs_sorted)), np.concatenate(([1.0], cs_sorted/xs_sorted)), extrapolate=True)
            if verbose:
                print(
                    "Convergence: {}.".format(max(abs(copt_old(xs)-copt(xs)))),
                    "Convergence criterion: {}.".format(conv),
                )
        return a_grid, PchipInterpolator(np.concatenate(([0.0], xs_sorted)), np.concatenate(([1.0], cs_sorted/xs_sorted)), extrapolate=True)
    
    #-------------------------------------------------------------------
    # Time iterations
    #-------------------------------------------------------------------

    def time_iterations(
            self, 
            grid_points:int = 1000, 
            n_knots:int=8, 
            lower_limit:float=0.1, 
            upper_limit:float=15., 
            tol:float=1e-5, 
            conv:float=0.1, 
            verbose:bool=True) -> Union[np.ndarray, PchipInterpolator]:
        """ 
        Computes optimal consumption policy by time iterations
        
        grid_points: number of grid points
        n_knots: number of Gauss Hermite points
        lower_limit: lower limit of grid
        upper_limit: upper limit of grid
        tol: tolerance level for minimization
        conv: convergence criterion
        verbose: if True, prints convergence information
        
        return: grid and optimal consumption policy as CubicSpline object
        """
        TOL = 1e-3
        assert lower_limit > TOL, f"The lower limit of the grid must be larger than {TOL}. Current lower limit {lower_limit}."

        self.set_gh(n_knots)
        grid = np.linspace(lower_limit, upper_limit, grid_points)
        copt = PchipInterpolator(np.concatenate(([0.0], grid)), np.concatenate(([1.0], np.ones(grid_points)*0.1)), extrapolate=True)
        cs = copt(grid)
        euler_residual = self.euler_residual
        
        while True:
            for i, x in enumerate(grid):
                lo = 1e-6
                hi = max(1e-6, 1.0)  # ensure lo < hi

                flo = euler_residual(lo, x, copt)
                fhi = euler_residual(hi, x, copt)

                if flo * fhi > 0:
                    # No interior root. Choose corner consistent with the sign:
                    # Residual>0 everywhere -> increase c -> consume at upper corner.
                    # Residual<0 everywhere -> decrease c -> consume at lower corner.
                    cs[i] = hi if flo > 0 else lo
                else:
                    cs[i] = brentq(euler_residual, lo, hi, xtol=tol, args=(x, copt))
            if max(abs(copt(grid)-cs)) < conv:
                break
            if verbose:
                print(
                    "Convergence: {}.".format(max(abs(copt(grid)-cs))),
                    "Convergence criterion: {}.".format(conv),
                )
            
            copt = PchipInterpolator(np.concatenate(([0.], grid)), np.concatenate(([1.0], cs)), extrapolate=True)
        return grid, PchipInterpolator(np.concatenate(([0.], grid)), np.concatenate(([1.0], cs)), extrapolate=True)
    
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
        V = PchipInterpolator(grid, Vn, extrapolate=True)
        while True:
            for i in range(len(grid)):
                Vn[i] = bellman(copt(grid[i]), grid[i], V)
            V = PchipInterpolator(grid, Vn, extrapolate=True)
            if max(abs(Vn - V(grid))) < conv:
                return V

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
        eom = self.eom
        draw_shocks = self.draw_shocks
        u = self.u
        β = self.β
        
        state = np.empty(periods+1)
        state[0] = start_value
        cons = np.empty(periods)
        U = 0
        power = 1.0
        for t in range(periods):
            cons[t] = state[t] * copt(state[t])
            a = state[t]-cons[t]
            θ, ψ = draw_shocks()
            state[t+1] = eom(a, θ, ψ)
            U += power * u(cons[t])
            power *= β
        return cons, state, U
    
    #-------------------------------------------------------------------
    # Euler equation errors
    #-------------------------------------------------------------------
    def get_euler_errors(self, grid, policy, n_eps=100):
        '''
        Given a grid of capital values and a consumption policy, computes Euler equation errors
        over n_eps periods for each initial capital value in the grid.
        '''
        calculate_euler_errors = self.calculate_euler_errors
        simulation = self.simulation
        grid_len = len(grid)
        states = np.empty((grid_len, n_eps+1), dtype=float)
        euler_errors = np.empty((grid_len, n_eps), dtype=float)
        avg_return = 0
        for i, k in enumerate(grid):
            cons_path, state_path, U = simulation(policy, n_eps, start_value=k)
            states[i,:] = state_path
            euler_errors[i,:] = calculate_euler_errors(cons_path, state_path, policy)
            avg_return += U/grid_len
        return euler_errors[~np.isnan(euler_errors)].ravel(), states.ravel(), avg_return