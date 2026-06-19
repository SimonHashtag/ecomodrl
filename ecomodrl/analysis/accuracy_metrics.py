# ATTENTION: Formerly, analysis.py

# -----------------------------------------------------------------------------
# Model-agnostic analysis routines. Every function takes (model, policy, ...).
# The shared `policy` interface (see policies.py) is the only contract.
#
# This module owns the policy-evaluation layer:
#   - `euler_errors`         : log10 |1 - c_implied / c| over test states,
#                              using `model.next_state_distribution` and the
#                              same `_euler_rhs` used inside the EGM step.
#   - `policy_value_on_grid` : Bellman fixed point V = u + β·E[V'] under a
#                              fixed policy; used for CEV.
#   - `make_eval_pack`       : assemble the per-run rollout pack from
#                              `model.eval_start_states` + `model.sample_shocks`.
#   - `eval_value_grid`      : reward-based mean discounted return across the
#                              eval_pack's starting states; consumed by the
#                              training loop's checkpoint evaluator.
#   - `cev`                  : consumption-equivalent variation closed form.
#
# Per-model physics primitives (`_euler_rhs`, `next_state_distribution`,
# `sample_shocks`, `step`, `eval_start_states`, ...) live on the
# `EconomicModel` base in models.py; this module only consumes them.
# -----------------------------------------------------------------------------

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import wasserstein_distance


# ---------------------------------------------------------------------------
# Euler-error driver
# ---------------------------------------------------------------------------

def euler_errors(model, policy, n_sim=None, burn_in=1_000, seed=0):
    """Evaluate log10 |1 - c_implied / c| at a set of test states.

    For deterministic models (no shock weights) we sweep a dense log-spaced
    m-grid; for stochastic models we forward-simulate the policy under
    pre-drawn shocks with a burn-in. `policy(state) -> c` is any callable
    consuming flat (..., STATE_DIM) states.
    """
    if model.weights is None:
        if n_sim is None:
            n_sim = 10_000
        states = _test_states_deterministic(model, n_sim)
    else:
        if n_sim is None:
            n_sim = 100_000
        states = _test_states_simulated(model, policy, n_sim, burn_in, seed)

    c_today = policy(states)
    m = states[..., 0]
    a = m - c_today
    next_states, trans = model.next_state_distribution(states, a)
    c_next = policy(next_states)
    c_implied = model.inv_marginal_utility(model._euler_rhs(c_next, **trans))
    err = np.abs(1.0 - c_implied / c_today)
    if model.CONSTRAINED:
        # Euler equation does not hold where c = m.
        err = np.where(c_today / m > 0.999, 0.0, err)
    return np.log10(err[err > 0])


def _test_states_deterministic(model, n_sim):
    """Dense log-spaced m-grid across the model's m-grid span."""
    m_test = model._log_grid(model.m_grid[0], model.m_grid[-1], n_sim)
    return m_test[..., None]


def _test_states_simulated(model, policy, n_sim, burn_in, seed):
    """Forward-simulate under `policy` with pre-drawn shocks; drop burn-in.

    Initial state anchors m at `model.m_ref` and (for STATE_DIM > 1) z at 1
    (the AR(1) stationary mean for BrockMirman).
    """
    rng = np.random.default_rng(seed)
    total = n_sim + burn_in
    shocks = model.sample_shocks(rng, (total,))

    state = np.empty((total + 1, model.STATE_DIM))
    state[0, 0] = model.m_ref
    if model.STATE_DIM > 1:
        state[0, 1] = 1.0

    for t in range(total):
        c_t = policy(state[t])
        shock_t = None if shocks is None else shocks[t]
        state[t + 1] = model.transition(state[t], c_t, shock_t)

    return state[burn_in:-1]


def simulate_states(model, policy, n_sim, burn_in=1_000, seed=0):
    """Public wrapper around the Euler-error state generators.

    Returns the (n_sim, STATE_DIM) array of states used by `euler_errors`,
    so callers can derive auxiliary diagnostics (e.g. visited-m
    distribution) without duplicating the seeding/burn-in logic. Same
    behaviour: deterministic models get a dense log-spaced m-grid;
    stochastic models get a forward simulation seeded by `seed`.
    """
    if model.weights is None:
        return _test_states_deterministic(model, int(n_sim))
    return _test_states_simulated(model, policy, int(n_sim), int(burn_in), int(seed))


def wasserstein_euler(ee_rl, ee_egm):
    """1-Wasserstein distance between two empirical distributions of
    log10|Euler errors|. Both inputs are 1-D arrays; non-finite entries
    are filtered defensively.
    """
    a = np.asarray(ee_rl, dtype=np.float64).ravel()
    b = np.asarray(ee_egm, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


# ---------------------------------------------------------------------------
# Aggregation utilities (IQM + two-way crossed bootstrap CI)
# ---------------------------------------------------------------------------

def iqm(x):
    """Interquartile mean: mean of values in [Q25, Q75]. NaNs dropped.
    Returns NaN if input is empty after filtering.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    q1, q3 = np.percentile(a, [25, 75])
    mask = (a >= q1) & (a <= q3)
    if not mask.any():
        return float(np.mean(a))
    return float(np.mean(a[mask]))


def iqm_ci_crossed(matrix, B=2000, alpha=0.05, seed=42):
    """Two-way crossed (pigeonhole) bootstrap 95% CI for the IQM.

    `matrix` is the (n_train, n_test) grid of a metric (NaNs allowed for
    missing cells). Each replicate resamples train-seed rows AND test-seed
    columns with replacement and recomputes the IQM over the induced
    submatrix, so both the train-seed and test-seed variance components
    enter the interval. A flat cell-level resample would instead treat the
    n_train*n_test correlated cells as independent and badly understate the
    width -- the cells come from a fully-crossed design with only n_train
    independent train seeds and n_test independent test seeds, not
    n_train*n_test independent draws.

    For a deterministic single-column grid (n_test == 1) this degenerates
    to a one-way train-seed bootstrap, which is exactly right for CK.

    References: Owen (2007), "The pigeonhole bootstrap"; McCullagh (2000),
    "Resampling and exchangeable arrays".
    """
    M = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    finite = M[np.isfinite(M)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = iqm(finite)
    R, C = M.shape
    rng = np.random.default_rng(int(seed))
    iqms = np.empty(int(B), dtype=np.float64)
    for b in range(int(B)):
        ri = rng.integers(0, R, size=R)
        ci = rng.integers(0, C, size=C)
        # iqm() flattens, drops NaNs, and falls back to the plain mean when
        # the resampled IQR box is empty.
        iqms[b] = iqm(M[np.ix_(ri, ci)])
    iqms = iqms[np.isfinite(iqms)]
    if iqms.size == 0:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(iqms, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Bellman policy-evaluation on the grid (for CEV)
# ---------------------------------------------------------------------------

def policy_value_on_grid(model, policy, tol=1e-8, max_iter=10_000):
    """Iterate V = u + β·E[V'] to convergence under a fixed policy.

    Builds the evaluation grid from `model.m_grid` (and `model.xi_grid` when
    STATE_DIM > 1); excludes m = 0 for CONSTRAINED models. Off-grid transition
    data is fetched once via `model.next_state_distribution`; V interpolation
    uses `np.interp` (1-D) or `RegularGridInterpolator` (2-D). Shock
    integration uses the shared `model._integrate`.

    Returns `(grid, V)`. For STATE_DIM > 1 models `grid` is the tuple
    `(m_grid, xi_grid)` and V is shaped (n_z, n_m); for 1-D models
    `grid` is the bare `m_grid` array and V is shaped (n_m,).
    """
    if model.STATE_DIM > 1:
        m_axis, xi_axis = model.m_grid, model.xi_grid
        Z, M = np.meshgrid(xi_axis, m_axis, indexing='ij')   # (n_z, n_m)
        states = np.stack([M, Z], axis=-1)
        interp_axes = (xi_axis, m_axis)
        display_grid = (m_axis, xi_axis)
    else:
        m_axis = model.m_grid[1:] if model.CONSTRAINED else model.m_grid
        states = m_axis[..., None]
        interp_axes = (m_axis,)
        display_grid = m_axis

    c = policy(states)
    a = states[..., 0] - c
    next_states, _ = model.next_state_distribution(states, a)
    u = model.utility(c)

    V = np.zeros_like(u)
    for _ in range(max_iter):
        V_next = _interpolate_value(V, interp_axes, next_states)
        EV = model._integrate(model.value_kernel() * V_next)
        V_new = u + model.BETA_EFF * EV
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return display_grid, V


def _interpolate_value(V, interp_axes, next_states):
    """Evaluate V (defined on `interp_axes`) at `next_states` coordinates.

    `interp_axes` order matches V's index order. For STATE_DIM > 1 models V is
    indexed (z, m) but the state convention is (m, z), so the coord tuple
    swaps the last two state components.
    """
    if len(interp_axes) == 1:
        return np.interp(next_states[..., 0].ravel(),
                         interp_axes[0], V).reshape(next_states.shape[:-1])
    interp = RegularGridInterpolator(
        interp_axes, V, bounds_error=False, fill_value=None,
    )
    return interp((next_states[..., 1], next_states[..., 0]))


# ---------------------------------------------------------------------------
# Reward-based mean discounted return (used by training-loop checkpoint eval)
# ---------------------------------------------------------------------------

def make_eval_pack(model, seed, M, T):
    """Assemble the per-run rollout pack.

    init_states come from `model.eval_start_states(M)` (the model may return
    M_eff != M, e.g. BM's meshgrid layout). shocks come from
    `model.sample_shocks(rng, (M_eff, T))`; None for deterministic models.
    The same pack is reused across every checkpoint within a run (and across
    algos), so the score is paired across checkpoints.
    """
    rng = np.random.default_rng(int(seed))
    init_states = model.eval_start_states(int(M))
    M_eff = init_states.shape[0]
    shocks = model.sample_shocks(rng, (M_eff, int(T)))
    return {'seed': int(seed), 'M': M_eff, 'T': int(T), 'init_states': init_states, 'shocks': shocks}


def eval_value_grid(model, policy, eval_pack):
    """Mean discounted return across M fixed starting states.

    eval_pack must contain 'M', 'T', 'init_states' of shape
    (M, STATE_DIM), and 'shocks' of shape (M, T, ...) (or None for
    deterministic models) consumed by `model.transition`. Returns a Python
    float; may be -inf if the policy ever drives c <= 0 in any rollout
    (caller treats that as ineligible).
    """
    M = int(eval_pack['M'])
    T = int(eval_pack['T'])
    state = np.asarray(eval_pack['init_states'], dtype=np.float64).copy()
    shocks = eval_pack['shocks']
    assert state.shape == (M, model.STATE_DIM)

    V = np.zeros(M, dtype=np.float64)
    disc = 1.0
    for t in range(T):
        c = np.asarray(policy(state), dtype=np.float64)
        V += disc * model.utility(c)
        disc *= model.BETA_EFF
        shock_t = None if shocks is None else shocks[:, t]
        state = model.transition(state, c, shock_t)
    return float(np.mean(V))


# ---------------------------------------------------------------------------
# Consumption-equivalent variation lambda(s) on the grid
#   lambda(s) = (V_egm(s) / V_rl(s)) ** (1/(1-gamma)) - 1     (CRRA closed form)
# Convention: V is negative under CRRA with gamma > 1, so the ratio is positive.
# ---------------------------------------------------------------------------

def cev(model, V_rl, V_egm):
    gamma = model.GAMMA
    if gamma == 1.0:
        # log utility: lambda = exp((V_egm - V_rl) * (1 - beta)) - 1
        return np.exp((V_egm - V_rl) * (1.0 - model.BETA_EFF)) - 1.0
    return (V_egm / V_rl) ** (1.0 / (1.0 - gamma)) - 1.0