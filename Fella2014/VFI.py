"""VFI Replication of Fella (2014).
For every state (a, d, y) and every durable d', 
try every a' on the asset grid and keep the best pair (d', a'). 

Fella's VFI1 (Sec. 5.2) adds two improvements:
- MONOTONE: a' increases with cash-on-hand z (Prop. 1), so the search over a' starts at the
  optimum of the previous asset level. Same solution, less time.
- REFINE: the best grid point is refined off the grid, between its two neighbours, with Ṽ
  linearly interpolated. More accurate savings, more time.

Additional choices by Fella (the code marks them with "# Fella:")
1. Given d', the savings problem depends on (a, d, y) only through cash-on-hand z (Eqs. 15-17),
   and measuring assets as a = w + ξ d turns the collateral constraint into a' >= 0 (Eq. 8).
2. The initial guess is 𝕍^0 = u(r a + y, d) instead of 𝕍^0 = 0, and iteration stops on the
   relative, not the absolute, sup-norm change of 𝕍.
3. An unaffordable durable (z <= 0) gets the value PENALTY.
"""

## Imports
import time # timing
import numpy as np # numerical operations
import matplotlib.pyplot as plt # plotting
from numba import njit # compiles functions to machine code
from scipy.stats import norm # normal CDF for Tauchen's method

## Economic Parameters (Table 1, Section 5.1; values as in Fella's Fortran code)
BETA = 0.93 # discount factor
R = 0.06 # NET interest rate r (not 1 + r)
THETA = 0.77 # weight of non-durable consumption in utility
KAPPA = 0.075 # service flow per unit of durable stock
PHI = 0.06 # fixed adjustment cost: pay PHI * d' whenever d' != d
XI = 0.8 # collateralizable share of the durable (20% down payment)

## Income Parameters: log y = p + e (the paper's z + ε; z is cash-on-hand here)
RHO = 0.977 # persistence of the AR(1) component p
SIGMA_ETA = 0.024 # std dev of the innovation to p
SIGMA_EPS = 0.063 # std dev of the iid component e
N_P, N_E = 7, 7 # Tauchen nodes for p and e (49 income states)
TAUCHEN_WIDTH = 3.0 # grid covers +/- 3 unconditional std devs

## Grid Parameters
N_A = 200 # number of asset levels (the smallest grid of Fig. 4)
N_D = 7 # number of durable levels
D_MIN, D_MAX = 0.01, 10.0 # durable grid bounds (D_MIN > 0 keeps log(KAPPA * d') finite)
A_MAX = 25.0 # upper bound of the asset grid

## Numerical Parameters
TOL = 1e-5 # tolerance on the relative sup-norm change of 𝕍
MAX_ITER = 1000 # safety cap on Bellman iterations
PENALTY = -1e7 # value of an unaffordable durable choice (z <= 0)
REFINE_TOL = 1e-8 # stop refining once the bracket is this narrow

## Switches: both False is textbook VFI, both True is Fella's VFI1
MONOTONE = True # start the search over a' at the previous optimum (Prop. 1)
REFINE = True # refine the best grid point off the grid

## Simulation Parameters (Section 5.2 and Fella's code)
T_SIM = 50_000 # number of simulated periods
SEED = 0 # seed of the income draws (the same history for every variant)
A0 = 0.0 # initial assets
D0_INDEX = 0 # initial durable: smallest level d = 0.01
Y0_INDEX = 24 # initial income: median state y = 1

## Plot Parameters
BIN_WIDTH = 0.2 # histogram bin width, as in Fig. 4
Y_TICKS = (0, 1, 5, 10, 100, 8000) # counts shown on the log(1 + count) axis


# --- Economic primitives -----------------------------------------------------

@njit
def utility(c, d_next):
    """u(c, d') = θ log c + (1 - θ) log(κ d'), Eq. (22) with τ = 0."""
    return THETA * np.log(c) + (1 - THETA) * np.log(KAPPA * d_next)

@njit
def cash_on_hand(a, d, y, d_next):
    """z(d'; a, d, y), Eq. (6) with γ = 0: resources left for c + a' after choosing the durable d'.

    Fella: a = w + ξ d is wealth plus borrowing capacity (Eq. 5), so the borrowing constraint
    w' >= -ξ d' simply reads a' >= 0 (Eq. 8): the same asset grid works for every d'.
    """
    adjustment_cost = PHI * d_next * (d_next != d)
    return y + (1 + R) * a - (1 - XI) * (d_next - d) - adjustment_cost - R * XI * d


# --- Grids -------------------------------------------------------------------

def tauchen(n, rho, sigma):
    """Tauchen (1986): n-point Markov chain approximating x' = ρ x + σ ε."""
    std_x = sigma / np.sqrt(1 - rho**2)
    x = np.linspace(-TAUCHEN_WIDTH * std_x, TAUCHEN_WIDTH * std_x, n)
    half_step = (x[1] - x[0]) / 2
    # P[i, j] = Prob(x' in [x_j - half_step, x_j + half_step] | x = x_i)
    upper = norm.cdf((x[None, :] + half_step - rho * x[:, None]) / sigma)
    lower = norm.cdf((x[None, :] - half_step - rho * x[:, None]) / sigma)
    upper[:, -1] = 1.0 # last interval extends to +infinity
    lower[:, 0] = 0.0 # first interval extends to -infinity
    return x, upper - lower

def asset_grid(n_a):
    """Double-exponential grid on [0, A_MAX]: dense near the borrowing limit a = 0."""
    x = np.linspace(0.0, np.log(np.log(A_MAX + 1) + 1), n_a)
    return np.exp(np.exp(x) - 1) - 1

def make_grids(n_a):
    """Asset grid G_a, durable grid D, income states Y and their transition matrix P."""
    a_grid = asset_grid(n_a)
    d_grid = np.linspace(D_MIN, D_MAX, N_D)
    p, P_p = tauchen(N_P, RHO, SIGMA_ETA)
    e, P_e = tauchen(N_E, 0.0, SIGMA_EPS)
    y_grid = np.exp(p[:, None] + e[None, :]).ravel() # state index = N_E * i_p + i_e
    P = np.kron(P_p, P_e) # (49, 49)
    return a_grid, d_grid, y_grid, P


# --- Numerical helpers -------------------------------------------------------

@njit
def interp(x, xp, fp):
    """Linear interpolation of fp(xp) at x, extrapolating linearly at both ends."""
    i = min(max(np.searchsorted(xp, x) - 1, 0), len(xp) - 2)
    weight = (x - xp[i]) / (xp[i + 1] - xp[i])
    return fp[i] + weight * (fp[i + 1] - fp[i])


# --- Fella's refinement: a' off the grid (REFINE) ----------------------------

@njit
def objective(a_next, z, d_next, a_grid, V_tilde):
    """u(z - a', d') + Ṽ(a'), the maximand of Eq. (16), with Ṽ linearly interpolated."""
    return utility(z - a_next, d_next) + interp(a_next, a_grid, V_tilde)

@njit
def refine(lo, hi, best, f_best, z, d_next, a_grid, V_tilde):
    """Fella's bracket search: evaluate the midpoint of [lo, hi] and cut the bracket."""
    while hi - lo > REFINE_TOL:
        mid = 0.5 * (lo + hi)
        f_mid = objective(mid, z, d_next, a_grid, V_tilde)
        if f_mid > f_best: # mid is better: the max lies on mid's side of best
            if mid < best:
                hi = best
            else:
                lo = best
            best, f_best = mid, f_mid
        else: # mid is worse: the max lies on best's side of mid
            if mid < best:
                lo = mid
            else:
                hi = mid
    return best, f_best


# --- The Bellman operator, Eq. (15) ------------------------------------------

def expectations(V, P):
    """Continuation value Ṽ(a', d', y) = β E[𝕍(a', d', y') | y], Eq. (11)."""
    # Ṽ does not depend on today's (a, d), so it is computed once per iteration and only
    # looked up in the maximization: same values as β Σ_y' P(y, y') 𝕍 inside the max, less work.
    # Read 𝕍(a, d, y) as tomorrow's 𝕍(a', d', y'); @ P.T sums over y' with weights P[y, y']
    return BETA * V @ P.T

@njit
def bellman_update(a_grid, d_grid, y_grid, V_tilde):
    """New 𝕍 at every state (a, d, y), Eq. (15), and 𝕍^{d'}, a'^{d'} on G_a x D x Y x D'.

    𝕍(a, d, y) = max over d' in D of [max over a' in [0, z) of u(z - a', d') + Ṽ(a', d', y)],
    where z = z(d'; a, d, y) is cash-on-hand and c = z - a' is consumption.
    """
    n_a, n_d, n_y = V_tilde.shape
    V = np.empty((n_a, n_d, n_y))
    V_cond = np.empty((n_a, n_d, n_y, n_d)) # value if the household picks d'
    a_cond = np.empty((n_a, n_d, n_y, n_d)) # savings if it picks d'
    for i_y in range(n_y):
        for i_d in range(n_d):
            # Fella: a' rises with z only for a given d' (Prop. 1), so each d' has its own start
            j_start = np.zeros(n_d, dtype=np.int64)
            for i_a in range(n_a):
                v_best = -np.inf
                for i_dn in range(n_d):
                    d_next = d_grid[i_dn]
                    # Fella: given d', the savings problem depends on (a, d, y) only through z
                    z = cash_on_hand(a_grid[i_a], d_grid[i_d], y_grid[i_y], d_next)
                    if z <= 0.0: # Fella: d' is not affordable, so it gets a very low value
                        v, a_next = PENALTY, 0.0
                    else:
                        # 1: Grid search: try every a' in G_a that leaves c = z - a' > 0
                        j_best, v = j_start[i_dn], -np.inf
                        for j in range(j_start[i_dn], n_a):
                            if a_grid[j] >= z: # c = z - a' must be positive
                                break
                            f = utility(z - a_grid[j], d_next) + V_tilde[j, i_dn, i_y]
                            if f > v:
                                j_best, v = j, f
                        a_next = a_grid[j_best]
                        if MONOTONE: # Fella: z rises with a, so the next search starts here
                            j_start[i_dn] = j_best
                        # 2: Fella: refine between the grid neighbours of the best node
                        if REFINE:
                            lo = a_grid[max(j_best - 1, 0)]
                            hi = min(a_grid[min(j_best + 1, n_a - 1)], z) # below z, so c > 0
                            a_next, v = refine(
                                lo, hi, a_next, v, z, d_next, a_grid, V_tilde[:, i_dn, i_y]
                            )
                    V_cond[i_a, i_d, i_y, i_dn] = v
                    a_cond[i_a, i_d, i_y, i_dn] = a_next
                    # 3: Discrete choice: keep the best d' so far
                    v_best = max(v_best, v)
                V[i_a, i_d, i_y] = v_best
    return V, V_cond, a_cond


# --- Value function iteration ------------------------------------------------

def initial_guess(a_grid, d_grid, y_grid):
    """Fella: 𝕍^0(a, d, y) = u(r a + y, d), i.e. consume interest and income (textbook: 𝕍^0 = 0)."""
    a, d, y = np.meshgrid(a_grid, d_grid, y_grid, indexing="ij")
    return utility(R * a + y, d)

def solve(n_a, tol=TOL, max_iter=MAX_ITER):
    """Iterate on the Bellman equation until 𝕍 converges (pseudo-code of Sec. 3.3)."""
    # 1: Grids and initial guess
    a_grid, d_grid, y_grid, P = make_grids(n_a)
    V = initial_guess(a_grid, d_grid, y_grid)

    for iteration in range(1, max_iter + 1):
        # 2: Continuation value Ṽ
        V_tilde = expectations(V, P)
        # 3: Bellman update: the best d' and a' at every state
        V_new, V_cond, a_cond = bellman_update(a_grid, d_grid, y_grid, V_tilde)
        # 4: Fella: relative sup-norm distance; the textbook uses max |𝕍_new - 𝕍|
        distance = np.max(np.abs(V_new - V)) / np.max(np.abs(V))
        V = V_new
        if distance < tol:
            break

    # The d'-contingent solution is all the simulation needs
    return V_cond, a_cond, iteration


# --- Policy functions off the grid -------------------------------------------

@njit
def policy(a, i_d, i_y, a_grid, d_grid, y_grid, V_cond, a_cond):
    """Choices (d', a', c) at any assets a: interpolate in a, then pick the best d'."""
    # 1: Discrete choice: the d' with the highest interpolated value, Eq. (14)
    i_best, v_best = 0, -np.inf
    for i_dn in range(len(d_grid)):
        v = interp(a, a_grid, V_cond[:, i_d, i_y, i_dn])
        if v > v_best:
            i_best, v_best = i_dn, v
    # 2: Savings and consumption given that d'
    a_next = interp(a, a_grid, a_cond[:, i_d, i_y, i_best])
    c = cash_on_hand(a, d_grid[i_d], y_grid[i_y], d_grid[i_best]) - a_next
    return i_best, a_next, c

@njit
def euler_error(a_next, i_dn, i_y, c, a_grid, d_grid, y_grid, P, V_cond, a_cond):
    """log10 |1 - c*/c|, where u_c(c*) = β(1 + r) E[u_c(c')], Eqs. (24)-(25)."""
    # 1: Expected marginal utility tomorrow, E[θ / c'], over all income states y'
    expected_mu = 0.0
    for i_yn in range(len(y_grid)):
        _, _, c_next = policy(a_next, i_dn, i_yn, a_grid, d_grid, y_grid, V_cond, a_cond)
        expected_mu += P[i_y, i_yn] * THETA / c_next
    # 2: The consumption that would satisfy the Euler equation exactly
    c_star = THETA / (BETA * (1 + R) * expected_mu)
    return np.log10(np.abs(1 - c_star / c))


# --- Simulation --------------------------------------------------------------

@njit
def simulate(a_grid, d_grid, y_grid, P_cum, P, V_cond, a_cond, draws):
    """Simulate one income history; return the Euler error of every unconstrained period."""
    errors = np.empty(len(draws))
    n_errors = 0
    a, i_d, i_y = A0, D0_INDEX, Y0_INDEX
    for t in range(len(draws)):
        # 1: Today's choices
        i_dn, a_next, c = policy(a, i_d, i_y, a_grid, d_grid, y_grid, V_cond, a_cond)
        # 2: Euler error only "off corners": a' at or above the first positive grid point
        if a_next >= a_grid[1]:
            errors[n_errors] = euler_error(
                a_next, i_dn, i_y, c, a_grid, d_grid, y_grid, P, V_cond, a_cond
            )
            n_errors += 1
        # 3: Tomorrow's state; y' is drawn from row i_y of P by inverting its CDF
        a, i_d = a_next, i_dn
        i_y = min(np.searchsorted(P_cum[i_y], draws[t]), len(y_grid) - 1) # min(): rounding guard
    return errors[:n_errors]

def euler_errors(V_cond, a_cond, n_a):
    """log10 Euler errors of a solution along a T_SIM-period simulated history."""
    a_grid, d_grid, y_grid, P = make_grids(n_a)
    P_cum = np.cumsum(P, axis=1) # cumulative transition probabilities
    draws = np.random.default_rng(SEED).random(T_SIM)
    return simulate(a_grid, d_grid, y_grid, P_cum, P, V_cond, a_cond, draws)


# --- Output ------------------------------------------------------------------

def plot_euler_errors(errors, cpu):
    """One panel of Fig. 4: histogram of log10 Euler errors, with average, maximum and CPU time."""
    bins = np.arange(np.floor(errors.min()), np.ceil(errors.max()) + BIN_WIDTH, BIN_WIDTH)

    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.hist(errors, bins=bins, color="0.2")
        ax.axvline(errors.mean(), color="red", linewidth=1)
        ax.text(0.03, 0.97, f"Avg.: {errors.mean():.2f}\nMax: {errors.max():.2f}\n"
                f"CPU: {cpu:.0f} s", transform=ax.transAxes, va="top")
        ax.set_yscale("function", functions=(np.log1p, np.expm1))
        ax.set_yticks(Y_TICKS)
        ax.set_ylim(0, 3 * max(Y_TICKS))
        ax.set_title(f"{N_A} grid points")
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(f"MONOTONE = {MONOTONE}, REFINE = {REFINE}", rotation=270, labelpad=15)
        fig.supxlabel("Euler errors (log10)")
        fig.supylabel("Count")
        fig.tight_layout()
    plt.show()


def main():
    print(f"Solving on {N_A} grid points, MONOTONE = {MONOTONE}, REFINE = {REFINE} ...", flush=True)
    solve(n_a=10, max_iter=1) # warm-up: compile the Numba code before timing
    start = time.process_time() # CPU time, as in the paper
    V_cond, a_cond, n_iter = solve(N_A)
    cpu = time.process_time() - start
    print(f"Converged after {n_iter} iterations, {cpu:.1f} s CPU")
    plot_euler_errors(euler_errors(V_cond, a_cond, N_A), cpu)


if __name__ == "__main__":
    main()
