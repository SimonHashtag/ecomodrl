# -----------------------------------------------------------------------------
# Formerly analysis.py
#
# Model-agnostic analysis routines.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import wasserstein_distance


# =============================================================================
# Policy helper
# =============================================================================

def _consumption(policy, state):
    """Evaluate policy consumption.

    Preferred interface:
        policy.consumption(state)

    Fallback:
        policy(state)
    """
    if hasattr(policy, "consumption"):
        return policy.consumption(state)

    if callable(policy):
        return policy(state)

    raise TypeError(
        "policy must expose `.consumption(state)` or be callable."
    )


def _transition(model, state, c, shocks):
    """Transition using consumption as the current control.

    The rewritten models expect:
        model.transition(state, controls, shocks)

    where:
        controls = {'consumption': c}
    """
    return model.transition(state, {"consumption": c}, shocks)


def _is_constrained(model):
    """Detect occasionally binding borrowing constraint."""
    return model.interpolation_kink() is not None


# =============================================================================
# Euler errors
# =============================================================================

def euler_errors(model, policy, n_sim=None, burn_in=1_000, seed=0):
    """Evaluate log10 absolute Euler errors.

    Computes:

        log10 |1 - c_implied / c_policy|

    where `c_implied` is obtained from the Euler equation.

    Deterministic models:
        use a dense deterministic m-grid.

    Stochastic models:
        simulate test states under the supplied policy after burn-in.
    """
    if model.weights is None:
        if n_sim is None:
            n_sim = 10_000
        states = _test_states_deterministic(model, int(n_sim))
    else:
        if n_sim is None:
            n_sim = 100_000
        states = _test_states_simulated(
            model=model,
            policy=policy,
            n_sim=int(n_sim),
            burn_in=int(burn_in),
            seed=int(seed),
        )

    c_today = np.asarray(_consumption(policy, states), dtype=np.float64)
    m_today = states[..., 0]
    a_today = m_today - c_today

    next_states, trans = model.next_state_distribution(states, a_today)
    c_next = _consumption(policy, next_states)

    rhs = model.euler_rhs(c_next, **trans)
    c_implied = model.inv_marginal_utility(rhs)

    err = np.abs(1.0 - c_implied / c_today)

    if _is_constrained(model):
        # Euler equation does not hold at constrained states. In the current
        # buffer-stock model, this region is characterized by c approximately m.
        constrained = c_today / np.maximum(m_today, 1e-300) > 0.999
        err = np.where(constrained, 0.0, err)

    err = err[np.isfinite(err) & (err > 0.0)]

    if err.size == 0:
        return np.array([], dtype=np.float64)

    return np.log10(err)


def _test_states_deterministic(model, n_sim):
    """Dense log-spaced m-grid for deterministic one-state models."""
    if model.exogenous_dim == 0:
        m_test = model.log_grid(model.m_grid[0], model.m_grid[-1], int(n_sim))
        return m_test[:, None]

    # Fallback for future deterministic models with exogenous states.
    return model.eval_start_states(int(n_sim))

# TODO: Vectorize?
def _test_states_simulated(model, policy, n_sim, burn_in, seed):
    """Forward-simulate states under `policy` and discard burn-in.
    
    Arguments:
    ----
    model: EconomicModel    The economic model instance.
    policy: callable        The policy to evaluate.
    n_sim: int              Number of post-burn-in states to return.
    burn_in: int            Number of initial states to discard.
    seed: int               Random seed for reproducibility.
    """
    rng = np.random.default_rng(int(seed))

    total = int(n_sim) + int(burn_in)
    shocks = model.sample_shocks(rng, (total,))

    states = np.empty((total + 1, model.state_dim), dtype=np.float64)
    states[0] = model.sample_initial_state(rng)

    for t in range(total):
        c_t = _consumption(policy, states[t])
        shock_t = None if shocks is None else shocks[t]
        states[t + 1] = _transition(model, states[t], c_t, shock_t)

    return states[burn_in:-1]


def simulate_states(model, policy, n_sim, burn_in=1_000, seed=0):
    """Return the states used by `euler_errors`."""
    if model.weights is None:
        return _test_states_deterministic(model, int(n_sim))

    return _test_states_simulated(
        model=model,
        policy=policy,
        n_sim=int(n_sim),
        burn_in=int(burn_in),
        seed=int(seed),
    )


def wasserstein_euler(ee_rl, ee_egm):
    """1-Wasserstein distance between two Euler-error distributions."""
    a = np.asarray(ee_rl, dtype=np.float64).ravel()
    b = np.asarray(ee_egm, dtype=np.float64).ravel()

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        return float("nan")

    return float(wasserstein_distance(a, b))


# =============================================================================
# Aggregation utilities
# =============================================================================

def iqm(x):
    """Interquartile mean with NaNs dropped."""
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
    """Two-way crossed bootstrap confidence interval for the IQM."""
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
        iqms[b] = iqm(M[np.ix_(ri, ci)])

    iqms = iqms[np.isfinite(iqms)]

    if iqms.size == 0:
        return float(point), float("nan"), float("nan")

    lo, hi = np.percentile(iqms, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return float(point), float(lo), float(hi)


# =============================================================================
# Bellman policy evaluation on the grid
# =============================================================================

def policy_value_on_grid(model, policy, tol=1e-8, max_iter=10_000):
    """Iterate the Bellman equation under a fixed policy.

    Computes:

        V(s) = u(c(s)) + beta * E[ value_kernel(s') * V(s') ]

    using the model's quadrature distribution.

    Important for BufferStock:
        This grid-based evaluation uses the true shock distribution and
        `model.value_kernel()`, so the discount factor is `model.params.beta`.

        The tilted simulation transition uses `model.beta_eff`, but that is for
        Monte Carlo rollout evaluation, not this quadrature Bellman operator.
    """
    axes = _value_grid_axes(model)
    states = _state_mesh_from_axes(axes)

    c = np.asarray(_consumption(policy, states), dtype=np.float64)
    a = states[..., 0] - c

    next_states, _ = model.next_state_distribution(states, a)
    u = model.utility(c)

    V = np.zeros_like(u, dtype=np.float64)
    beta = model.params.beta

    for _ in range(int(max_iter)):
        V_next = _interpolate_value(V, axes, next_states)
        EV = model.integrate(model.value_kernel() * V_next)

        V_new = u + beta * EV

        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break

        V = V_new

    display_grid = axes[0] if len(axes) == 1 else axes
    return display_grid, V


def _value_grid_axes(model):
    """Axes for grid-based value evaluation.

    The order matches state order:

        (m_grid, x1_grid, ..., xd_grid)

    For constrained models, m=0 is removed to avoid utility at zero
    consumption.
    """
    m_axis = model.m_grid

    if _is_constrained(model) and m_axis[0] <= 0.0:
        m_axis = m_axis[1:]

    return (m_axis,) + tuple(model.exogenous_grids)


def _state_mesh_from_axes(axes):
    """Create state mesh in state order.

    Example:
        axes = (m_grid, z_grid)

    returns:
        states.shape == (n_m, n_z, 2)
    """
    meshes = np.meshgrid(*axes, indexing="ij")
    return np.stack(meshes, axis=-1)


def _interpolate_value(V, axes, next_states):
    """Interpolate V at next-state coordinates.

    axes:
        tuple of grids in state order, e.g. (m_grid, z_grid)

    next_states:
        array with final dimension equal to len(axes).
    """
    next_states = np.asarray(next_states, dtype=np.float64)

    if len(axes) == 1:
        values = np.interp(
            next_states[..., 0].ravel(),
            axes[0],
            V,
        )
        return values.reshape(next_states.shape[:-1])

    coords = _clip_to_axes(next_states, axes)

    interp = RegularGridInterpolator(
        axes,
        V,
        bounds_error=False,
        fill_value=None,
    )

    return interp(coords)


def _clip_to_axes(coords, axes):
    coords = np.asarray(coords, dtype=np.float64).copy()

    for j, grid in enumerate(axes):
        coords[..., j] = np.clip(coords[..., j], grid[0], grid[-1])

    return coords


# =============================================================================
# Rollout-based return evaluation
# =============================================================================

@dataclass(frozen=True)
class EvaluationSample:
    """Fixed common-random-numbers sample for policy evaluation.

    Contains initial states and shock paths used to evaluate all policies on the
    same simulation design.
    """

    seed: int
    initial_states: np.ndarray
    shocks: object
    horizon: int

    @property
    def n_states(self) -> int:
        return int(self.initial_states.shape[0])


def make_evaluation_sample(model, seed, n_states, horizon) -> EvaluationSample:
    """Create a fixed evaluation sample for rollout-based policy comparison."""
    rng = np.random.default_rng(int(seed))

    initial_states = np.asarray(
        model.eval_start_states(int(n_states)),
        dtype=np.float64,
    )

    n_eff = initial_states.shape[0]

    shocks = model.sample_shocks(
        rng,
        (n_eff, int(horizon)),
    )

    return EvaluationSample(
        seed=int(seed),
        initial_states=initial_states,
        shocks=shocks,
        horizon=int(horizon),
    )


def eval_return_grid(model, policy, evaluation_sample):
    """Mean discounted rollout return across fixed starting states.

    This uses the model's simulation transition. For BufferStock, the model's
    simulation transition uses the tilted permanent-shock distribution, so
    discounting by `model.beta_eff` is correct here.
    """
    M = int(evaluation_sample.n_states)
    T = int(evaluation_sample.horizon)

    state = np.asarray(evaluation_sample.initial_states, dtype=np.float64).copy()
    shocks = evaluation_sample.shocks

    if state.shape != (M, model.state_dim):
        raise ValueError(
            f"Expected init_states shape {(M, model.state_dim)}, "
            f"got {state.shape}."
        )

    U = np.zeros(M, dtype=np.float64)
    disc = 1.0

    for t in range(T):
        c = np.asarray(_consumption(policy, state), dtype=np.float64)

        U += disc * model.utility(c)
        disc *= model.beta_eff

        shock_t = None if shocks is None else shocks[:, t]
        state = _transition(model, state, c, shock_t)

    return float(np.mean(U))


# =============================================================================
# Consumption-equivalent variation
# =============================================================================

def cev(model, V_rl, V_egm, discount=None):
    """Consumption-equivalent variation.

    For CRRA with gamma != 1:

        lambda = (V_egm / V_rl) ** (1 / (1 - gamma)) - 1

    For log utility, a scalar discount is required. If not supplied, this uses
    `model.beta_eff`, which is appropriate for rollout values. For grid Bellman
    values using the true quadrature operator, pass `discount=model.params.beta`.
    """
    gamma = model.params.gamma

    if gamma == 1.0:
        if discount is None:
            discount = model.beta_eff
        return np.exp((V_egm - V_rl) * (1.0 - discount)) - 1.0

    return (V_egm / V_rl) ** (1.0 / (1.0 - gamma)) - 1.0