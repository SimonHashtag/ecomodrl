import matplotlib.pyplot as plt
import numpy as np

from typing import Union
from scipy.interpolate import CubicSpline
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem
from scipy.optimize import minimize

#-------------------------------------------------------------------
# Dynamic Programming (VFI)
#-------------------------------------------------------------------

def vfi(pp, grid_points: int = 1000, lower_limit: float = 0.1, upper_limit: float= 15., tol: float = 1e-5, conv: float = 0.1, verbose: bool=True):
    """ 
    Given a PlanningProblem object, computes optimal consumption policy by value function iteration
    pp: PlanningProblem object
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
                pp.bellman,
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
            break
        if verbose:
            print(
                "Convergence: {}.".format(max(abs(Vn - V(grid)))),
                "Convergence criterion: {}.".format(conv),
            )
        V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
    return grid, V, CubicSpline(grid, copt, bc_type="natural", extrapolate=True)

#-------------------------------------------------------------------
# Time iterations
#-------------------------------------------------------------------

def time_iterations(pp, grid_points:int = 1000, lower_limit:float=0.1, upper_limit:float=15., tol:float=1e-5, conv:float=0.1, verbose:bool=True) -> Union[np.ndarray, CubicSpline]:
    """ 
    Given a PlanningProblem object, computes optimal consumption policy by time iterations
    
    pp: PlanningProblem object
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
                pp.sq_euler_error,
                x0 = cs[i],
                bounds=((0.001, pp.pre_decision_next_k(grid[i])-0.001),),
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

def get_vfrom_copt(pp, grid, copt, conv):
    """
    Given optimal consumption policy, computes value function on grid by iterating on Bellman equation
    
    pp: PlanningProblem object
    grid: grid on which value function is computed
    copt: optimal consumption policy as CubicSpline object
    conv: convergence criterion
    
    return: value function as CubicSpline object
    """
    bellman = pp.bellman
    Vn = np.zeros(len(grid))
    V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
    while True:
        for i in range(len(grid)):
            Vn[i] = -bellman(copt(grid[i]), grid[i], V, cons_share=False)
        if max(abs(Vn - V(grid))) < conv:
            break
        V = CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)
    return CubicSpline(grid, Vn, bc_type="natural", extrapolate=True)

#-------------------------------------------------------------------
# EGM
#-------------------------------------------------------------------

def egm(pp, grid_points:int = 1000, lower_limit:float=0.1, upper_limit:float=15., conv:float=0.1, verbose:bool=True):
    """
    Given a PlanningProblem object, computes optimal consumption policy by endogenous grid method (EGM)
    
    pp: PlanningProblem object
    grid_points: number of grid points
    lower_limit: lower limit of grid
    upper_limit: upper limit of grid
    conv: convergence criterion
    verbose: if True, prints convergence information
    
    return: grid and optimal consumption policy as CubicSpline object
    """
    cash_on_hand = pp.pre_decision_next_k
    euler_egm = pp.euler_egm
    
    k_next_grid = np.linspace(lower_limit, upper_limit, grid_points)
    sopt = CubicSpline(np.concatenate(([0.0], k_next_grid)), np.concatenate(([0.0], k_next_grid*0.2)), bc_type="natural", extrapolate=True)
    ms = np.empty(grid_points)
    while True:
        for i, k_next in enumerate(k_next_grid):
            m_next = cash_on_hand(k_next)
            c_next = m_next - sopt(m_next)
            c = euler_egm(c_next, k_next)
            ms[i] = c + k_next
        if max(abs(sopt(ms)-k_next_grid)) < conv:
            c_of_m = CubicSpline(np.concatenate(([0.0], ms)), np.concatenate(([0.0], ms-k_next_grid)), bc_type="natural", extrapolate=True)
            m_grid = cash_on_hand(k_next_grid)
            copt = CubicSpline(k_next_grid, c_of_m(m_grid), bc_type="natural", extrapolate=True) 
            break
        if verbose:
            print(
                "Convergence: {}.".format(max(abs(sopt(ms)-k_next_grid))),
                "Convergence criterion: {}.".format(conv),
            )
        sopt = CubicSpline(np.concatenate(([0.0], ms)), np.concatenate(([0.0], k_next_grid)), bc_type="natural", extrapolate=True)
    return k_next_grid, copt 

#-------------------------------------------------------------------
# Simulation
#-------------------------------------------------------------------
def simulation(pp, copt, periods=100, start_value=5.0, cons_share=False):
    """
    Simulate consumption path over number of periods
    
    pp: PlanningProblem object
    copt: consumption policy as CubicSpline object
    periods: number of periods to simulate
    start_value: starting value of capital
    cons_share: if True, copt returns consumption share of pre-decision capital next period
    """
    next_k, pre_decision_next_k = pp.next_k, pp.pre_decision_next_k
    u = pp.u
    β = pp.β
    
    state = np.empty(periods+1)
    state[0] = start_value
    cons = np.empty(periods)
    U = 0
    power = 1.0
    for t in range(periods):
        if cons_share:
            action = copt(state[t])
            cons[t] = pre_decision_next_k(state[t])*action
        else:
            cons[t] = copt(state[t])
        state[t+1] = next_k(state[t], cons[t])
        U += power * u(cons[t])
        power *= β
    return cons, state, U

#-------------------------------------------------------------------
# Euler equation errors
#-------------------------------------------------------------------
def get_euler_errors(p, grid, policy, n_eps=100, cons_share=False):
    '''
    Given a grid of capital values and a consumption policy, computes Euler equation errors
    over n_eps periods for each initial capital value in the grid.
    '''
    calculate_euler_errors = p.calculate_euler_errors
    grid_len = len(grid)
    states = np.empty((grid_len, n_eps+1), dtype=float)
    euler_errors = np.empty((grid_len, n_eps-1), dtype=float)
    for i, k in enumerate(grid):
        cons_path, state_path, _ = simulation(p, policy, n_eps, start_value=k, cons_share=cons_share)
        states[i,:] = state_path
        euler_errors[i,:] = calculate_euler_errors(cons_path, state_path)
    return euler_errors.ravel(), states.ravel()

if __name__ == "__main__":
    pp = PlanningProblem()
    K_REF = 5.0
    TT_REF = 100

    # Time iterations 
    import time
    start = time.time()
    _, policy_ti = time_iterations(
        pp, grid_points=30, lower_limit=0.1, upper_limit=15., 
        tol=1e-5, conv=0.0001, verbose=False)
    print(f'Time iterations took {time.time()-start} seconds.')

    # VFI
    start = time.time()
    grid, _, policy_vfi = vfi(pp, grid_points=100, lower_limit=0.1, upper_limit=15., 
                        tol=1e-5, conv=0.001, verbose=False)
    print(f'VFI took {time.time()-start} seconds.')
    
    # EGM
    start = time.time()
    grid, policy_egm = egm(pp, grid_points=100, lower_limit=0.1, upper_limit=15., 
                        conv=0.001, verbose=False)
    print(f'EGM took {time.time()-start} seconds.')
    
    # Return comparison
    _, _, U_vfi = simulation(pp, policy_vfi, TT_REF, start_value=K_REF, cons_share=True)
    _, _, U_ti = simulation(pp, policy_ti, TT_REF, start_value=K_REF, cons_share=False)
    _, _, U_egm = simulation(pp, policy_egm, TT_REF, start_value=K_REF, cons_share=False)
    print(f'The optimal discounted episode return is: \r\n  TI: {U_ti}      VFI: {U_vfi}      EGM: {U_egm}')
    
    # Euler equation errors
    euler_errors_ti, _ = get_euler_errors(pp, grid, policy_ti, n_eps=100)
    euler_errors_vfi, _ = get_euler_errors(pp, grid, policy_vfi, n_eps=100, cons_share=True)
    euler_errors_egm, _ = get_euler_errors(pp, grid, policy_egm, n_eps=100, cons_share=False)
    print(f'Max absolute Euler equation error: \r\n  TI: {max(abs(euler_errors_ti))}      VFI: {max(abs(euler_errors_vfi))}      EGM: {max(abs(euler_errors_egm))}')
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(grid, policy_vfi(grid)*pp.pre_decision_next_k(grid), label='Value function iteration')
    ax[0].plot(grid, policy_ti(grid), linestyle='--', color='orange', label='Time iterations')
    ax[0].plot(grid, policy_egm(grid), linestyle='--', color='green', label='EGM')
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