# =============================================================================
# Policy accuracy metrics (Euler errors, CEV, Bellman values, policy comparison)
#
# Evaluates learned RL policies against analytical/numerical benchmarks across
# multiple dimensions: Euler equation errors, consumption-equivalent variation,
# and value function agreement. Supports standard economic models and
# portfolio-choice environments with occasionaly binding constraints.
#
# Metrics are computed per training seed on fixed evaluation samples,
# then aggregated with the IQM and bootstrap confidence intervals.
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import NdBSpline
from scipy.sparse import csr_matrix, vstack


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

    raise TypeError("policy must expose `.consumption(state)` or be callable.")


def _policy_controls(policy, state):
    """Evaluate full policy controls.

    Preferred interface:
        policy.controls(state)

    Fallback:
        consumption-only policy.
    """
    if hasattr(policy, "controls"):
        controls = dict(policy.controls(state))
    else:
        controls = {"consumption": _consumption(policy, state)}

    if "consumption" not in controls:
        if "share" not in controls:
            raise KeyError(
                "Policy controls must contain either 'consumption' or 'share'."
            )
        state_arr = np.asarray(state, dtype=np.float64)
        controls["consumption"] = np.asarray(controls["share"]) * state_arr[..., 0]

    return controls


def _is_constrained(model):
    """Detect occasionally binding borrowing constraint."""
    return model.interpolation_kink() is not None


def _is_portfolio_choice(model) -> bool:
    """Detect portfolio-choice models by their economic interface."""
    return (
        hasattr(model, "n_assets")
        and hasattr(model, "n_risky")
        and hasattr(model, "portfolio_weights")
        and hasattr(model, "_eps_nodes")
        and hasattr(model, "_eta_nodes")
    )


# =============================================================================
# Euler errors
# =============================================================================

def euler_errors(
    model,
    policy,
    n_sim,
    burn_in=0,
    seed=0,
    *,
    max_shock_nodes=None,
):
    """Evaluate log10 Euler errors.

    For standard one-control models this computes
    log10 |1 - c_implied / c_policy|.

    For portfolio-choice models this computes the log10 maximum of
    assetwise consumption-equivalent Euler/KKT residuals
    """
    if _is_portfolio_choice(model):
        return _portfolio_euler_errors(
            model=model,
            policy=policy,
            n_sim=int(n_sim),
            burn_in=int(burn_in),
            seed=int(seed),
            max_shock_nodes=max_shock_nodes,
        )

    if model.weights is None:
        states = _test_states_deterministic(model, int(n_sim))
    else:
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
        constrained = c_today / np.maximum(m_today, 1e-300) > 0.999
        err = np.where(constrained, 0.0, err)

    err = err[np.isfinite(err) & (err > 0.0)]
    return np.log10(err) if err.size else np.array([], dtype=np.float64)


def _test_states_deterministic(model, n_sim):
    """Dense grid for deterministic one-state models."""
    return model.eval_start_states(int(n_sim))


def _test_states_simulated(model, policy, n_sim, burn_in, seed):
    """Forward-simulate states under the full policy controls."""
    rng = np.random.default_rng(int(seed))

    total = int(n_sim) + int(burn_in)
    shocks = model.sample_shocks(rng, (total,))

    states = np.empty((total + 1, model.state_dim), dtype=np.float64)
    states[0] = model.sample_initial_state(rng)

    for t in range(total):
        controls_t = _policy_controls(policy, states[t])
        shock_t = None if shocks is None else shocks[t]
        states[t + 1] = model.transition(states[t], controls_t, shock_t)

    return states[burn_in:-1]


def simulate_states(model, policy, n_sim, burn_in=0, seed=0):
    """Return the states used by `euler_errors`."""

    return _test_states_simulated(
        model=model,
        policy=policy,
        n_sim=int(n_sim),
        burn_in=int(burn_in),
        seed=int(seed),
    )


# =============================================================================
# Aggregation utilities
# =============================================================================

def iqm(x, axis: int | None = None):
    """Interquartile mean with non-finite values dropped.

    With ``axis=None``, all observations are reduced to one scalar. With an
    integer axis, the reduction is performed independently for each retained
    slice and returns an array with that axis removed. Slices containing no
    finite observations return ``NaN``.
    """
    if axis is None:
        a = np.asarray(x, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]

        if a.size == 0:
            return float("nan")

        q1, q3 = np.percentile(a, [25, 75])
        mask = (a >= q1) & (a <= q3)

        if not mask.any():
            return float(np.mean(a))

        return float(np.mean(a[mask]))

    values = np.moveaxis(np.asarray(x, dtype=np.float64), axis, 0)
    reduced_shape = values.shape[1:]
    flattened = values.reshape(values.shape[0], -1)
    result = np.full(flattened.shape[1], np.nan, dtype=np.float64)

    for index in range(flattened.shape[1]):
        column = flattened[:, index]
        if np.any(np.isfinite(column)):
            result[index] = iqm(column)

    return result.reshape(reduced_shape)


def iqm_ci_crossed(
    matrix: np.ndarray, 
    B:int = 2000, 
    alpha:float = 0.05, 
    seed:int = 42,
):
    """IQM with a percentile-bootstrap confidence interval.

    matrix: np.ndarray 
        The input matrix for which to compute the IQM with confidence intervals.
    
    B: int
        Number of bootstrap resamples.
        
    alpha: float
        Significance level for the confidence interval. The default is 0.05, which corresponds to a 95% confidence interval.
        
    seed: int
        Random seed for reproducibility of the bootstrap resampling.
    
    The bootstrap is crossed over both rows and columns and returns three
    scalars (point, lo, hi). This is the aggregate train x test-seed metric.
    """
    M = np.asarray(matrix, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]

    M2 = np.atleast_2d(M)
    finite = M2[np.isfinite(M2)]

    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")

    point = iqm(finite)

    R, C = M2.shape
    iqms = np.empty(int(B), dtype=np.float64)
    for b in range(int(B)):
        ri = rng.integers(0, R, size=R)
        ci = rng.integers(0, C, size=C)
        iqms[b] = iqm(M2[np.ix_(ri, ci)])

    iqms = iqms[np.isfinite(iqms)]
    if iqms.size == 0:
        return float(point), float("nan"), float("nan")

    lo, hi = np.percentile(iqms, q)
    return float(point), float(lo), float(hi)


# =============================================================================
# Bellman policy evaluation on the grid
# =============================================================================

def _value_grid_axes(model):
    """Use the model's state grids, excluding zero cash for CRRA utility."""
    cash = np.asarray(model.m_grid)
    return (cash[cash > 0],) + tuple(model.exogenous_grids)


def _linear_interpolation_matrix(axes, coordinates):
    """Sparse linear interpolation weights, clamping each coordinate to its axis."""
    points = np.asarray(coordinates, dtype=float).reshape(-1, len(axes))
    if not np.isfinite(points).all():
        raise RuntimeError("Value interpolation received nonfinite coordinates.")
    points = np.clip(points, [axis[0] for axis in axes], [axis[-1] for axis in axes])
    knots = tuple(np.r_[axis[0], axis, axis[-1]] for axis in axes)
    weights = NdBSpline.design_matrix(points, knots, k=1, extrapolate=False)
    # SciPy omits trailing columns when no point touches the upper grid cells.
    weights.resize((len(points), int(np.prod([len(axis) for axis in axes]))))
    return weights


def interpolate_value(axes, values, coordinates):
    """Linear/multilinear values with constant continuation outside every axis."""
    coordinates = np.asarray(coordinates, dtype=float)
    weights = _linear_interpolation_matrix(axes, coordinates)
    return (weights @ np.asarray(values).ravel()).reshape(coordinates.shape[:-1])


def _expected_interpolation_matrix(axes, next_states, mass):
    """Average the shock-weighted interpolation rows for each current state."""
    n_shocks = len(mass)
    n_states = np.asarray(next_states).size // (len(axes) * n_shocks)
    expectation = csr_matrix(
        (np.tile(mass, n_states), np.arange(n_states * n_shocks),
         np.arange(n_states + 1) * n_shocks),
        shape=(n_states, n_states * n_shocks),
    )
    return expectation @ _linear_interpolation_matrix(axes, next_states)


def _policy_continuation_matrix(model, axes, states, controls, max_shock_nodes):
    """Precompute E[value_kernel(s') V(s')] under the fixed policy."""
    consumption = np.asarray(controls["consumption"], dtype=float)
    n_states = consumption.size
    if _is_portfolio_choice(model):
        weights = np.maximum(np.asarray(controls["portfolio_weights"], dtype=float), 0)
        weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-14)
        shock_idx = _portfolio_shock_indices(model, max_shock_nodes)
        kernel = model.value_kernel()[shock_idx] / len(shock_idx)
        mass = np.r_[(1-model.unemp_prob)*kernel, model.unemp_prob*kernel]
        rows = []
        # Construct interpolation rows for all predictors at one cash node together.
        for state_row, consumption_row, weight_row in zip(states, consumption, weights):
            next_states = np.asarray([
                _portfolio_next_states_for_value(model, state, max(state[0]-c, 0), w, shock_idx)
                for state, c, w in zip(state_row.reshape(-1, model.state_dim),
                                       consumption_row.ravel(), weight_row.reshape(-1, model.n_assets))
            ])
            rows.append(_expected_interpolation_matrix(axes, next_states, mass))
        return vstack(rows, format="csr")

    next_states, _ = model.next_state_distribution(states, states[..., 0]-consumption)
    next_states = np.asarray(next_states)
    n_shocks = next_states.reshape(n_states, -1, len(axes)).shape[1]
    kernel = np.broadcast_to(np.asarray(model.value_kernel()), (n_shocks,))
    quadrature = np.ones(n_shocks) if model.weights is None else np.asarray(model.weights)
    return _expected_interpolation_matrix(axes, next_states, kernel * quadrature)


def policy_value_on_grid(
    model,
    policy,
    tol=1e-8,
    max_iter=10_000,
    *,
    max_shock_nodes=None,
    grid_axes=None,
):
    """Solve V = u(c) + beta E[value_kernel(s') V(s')] for a fixed policy.

    Use the true shock distribution and structural beta. Coordinates outside
    the evaluation axes are clamped to their nearest endpoint. By default the
    axes are the model's state grids, excluding zero cash.
    """
    axes = _value_grid_axes(model) if grid_axes is None else tuple(grid_axes)
    states = _state_mesh_from_axes(axes)
    controls = _policy_controls(policy, states)
    utility = model.utility(np.maximum(controls["consumption"], 1e-12)).reshape(-1)
    continuation = _policy_continuation_matrix(model, axes, states, controls, max_shock_nodes)
    value = np.zeros(utility.size)
    for _ in range(int(max_iter)):
        updated = utility + model.params.beta * (continuation @ value)
        delta = float(np.max(np.abs(updated-value)))
        value = updated
        if not np.all(np.isfinite(value)):
            raise RuntimeError("Fixed-policy Bellman evaluation produced nonfinite values.")
        if delta < tol:
            break
    else:
        raise RuntimeError(f"Fixed-policy Bellman evaluation did not converge: delta={delta:.3e}.")
    display_grid = axes[0] if len(axes) == 1 else axes
    return display_grid, value.reshape(states.shape[:-1])


def _state_mesh_from_axes(axes):
    """Create state mesh in state order.

    Example:
        axes = (m_grid, z_grid)

    returns:
        states.shape == (n_m, n_z, 2)
    """
    meshes = np.meshgrid(*axes, indexing="ij")
    return np.stack(meshes, axis=-1)


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

    This uses the model's simulation transition. For BufferStock and
    PortfolioChoice, the transition uses the tilted permanent-shock
    distribution, so discounting by `model.beta_eff` is correct here.
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
        controls = _policy_controls(policy, state)
        c = np.asarray(controls["consumption"], dtype=np.float64)

        U += disc * model.utility(np.maximum(c, 1e-12))
        disc *= model.beta_eff

        shock_t = None if shocks is None else shocks[:, t]
        state = model.transition(state, controls, shock_t)

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


def cev_on_states(model, grid_axes, V_rl, V_egm, states):
    """Mean consumption-equivalent variation over simulated states.

    Computes the state-wise CEV on the shared value grid and averages 
    it over the provided simulated/ergodic states.
    """
    axes = grid_axes if isinstance(grid_axes, tuple) else (grid_axes,)
    states = np.asarray(states, dtype=np.float64)
    vals = cev(model, interpolate_value(axes, V_rl, states),
               interpolate_value(axes, V_egm, states), discount=model.params.beta)

    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    return float(np.mean(vals)) if vals.size else float("nan")


# =============================================================================
# Portfolio-choice Euler/KKT diagnostics
# =============================================================================

def _portfolio_euler_errors(
    model,
    policy,
    n_sim,
    burn_in,
    seed,
    *,
    max_shock_nodes=None,
):
    """Joint consumption/portfolio errors on the policy's own simulated states.

    For each asset, q_j = beta * g_j / u'(c) = E[M' R'_j]. The Euler-implied
    consumption is c_j = inv_u_prime(beta * g_j) = c * q_j**(-1/gamma).
    Positive holdings require q_j = 1; zero holdings require q_j <= 1.
    Accordingly, residuals are abs(1-c_j/c) for held assets and
    max(1-c_j/c, 0) for unheld assets. Zero saving makes all conditions one-sided.

    Return one log10 maximum asset residual per state; exact zeros give -inf.
    """
    states = _test_states_simulated(
        model=model,
        policy=policy,
        n_sim=int(n_sim),
        burn_in=int(burn_in),
        seed=int(seed),
    )

    shock_idx = _portfolio_shock_indices(model, max_shock_nodes)

    errors = np.empty(states.shape[0], dtype=np.float64)

    for i, state in enumerate(states):
        controls = _policy_controls(policy, state)

        c = float(np.asarray(controls["consumption"]))
        m = float(state[0])
        rs = max(m - c, 0.0)

        # Euler/KKT equations do not bind in the consume-all region.
        if c / max(m, 1e-300) > 0.999:
            errors[i] = np.nan
            continue

        if "portfolio_weights" not in controls:
            raise KeyError(
                "Portfolio-choice Euler diagnostics require "
                "controls['portfolio_weights']."
            )

        w = np.asarray(controls["portfolio_weights"], dtype=np.float64)
        w = w / w.sum()

        g = _portfolio_sdf_moments(
            model=model,
            policy=policy,
            state=state,
            rs=rs,
            w=w,
            shock_idx=shock_idx,
        )

        c_implied = np.asarray(model.inv_marginal_utility(model.params.beta * g))
        discrepancy = 1.0 - c_implied / c
        held = (rs / m) * w > 1e-10
        asset_errors = np.where(held, np.abs(discrepancy), np.maximum(discrepancy, 0.0))
        errors[i] = np.max(asset_errors)

    return np.log10(errors)


def _portfolio_sdf_moments(model, policy, state, rs, w, shock_idx):
    """Compute SDF-weighted expected asset returns for one state.

    g_j = E[ u'(c') / Gamma'^gamma * R'_j ]

    Employment and unemployment are integrated analytically as a two-point
    transitory-income mixture.
    """
    state = np.asarray(state, dtype=np.float64)
    z = state[1:]

    z_next, R_all, Gamma, theta_emp, sdf_factor = _portfolio_next_shock_arrays(
        model=model,
        z=z,
        shock_idx=shock_idx,
    )

    Rp = R_all @ w
    base = (Rp / Gamma) * float(rs)

    m_emp = base + theta_emp
    m_unemp = base + model.theta_low

    next_emp = np.column_stack([m_emp, z_next])
    next_unemp = np.column_stack([m_unemp, z_next])

    c_emp = np.asarray(_consumption(policy, next_emp), dtype=np.float64)
    c_unemp = np.asarray(_consumption(policy, next_unemp), dtype=np.float64)

    c_emp = np.maximum(c_emp, 1e-12)
    c_unemp = np.maximum(c_unemp, 1e-12)

    mu_emp = model.marginal_utility(c_emp)
    mu_unemp = model.marginal_utility(c_unemp)

    p_unemp = float(model.unemp_prob)
    mu_mix = (1.0 - p_unemp) * mu_emp + p_unemp * mu_unemp

    sdf_mu = sdf_factor * mu_mix

    return np.mean(sdf_mu[:, None] * R_all, axis=0)


def _portfolio_shock_indices(model, max_shock_nodes=None):
    """Select RQMC nodes for portfolio diagnostics.

    Full RQMC integration can be expensive for RL policies. By default, use at
    most `model.analysis_rqmc_nodes` if present, otherwise 1024. Use a prefix
    of the Sobol sequence so power-of-two sample sizes retain their nested
    low-discrepancy balance.
    """
    M = int(model._M_rqmc)

    if max_shock_nodes is None:
        max_shock_nodes = int(getattr(model, "analysis_rqmc_nodes", min(M, 1024)))

    max_shock_nodes = int(max_shock_nodes)

    if max_shock_nodes >= M:
        return np.arange(M)

    return np.arange(max_shock_nodes)


def _portfolio_next_shock_arrays(model, z, shock_idx):
    """Return true-distribution next predictors, returns, and income shocks.

    Uses the model's RQMC nodes under the true distribution, not the tilted RL
    simulation distribution.
    """
    idx = np.asarray(shock_idx, dtype=np.int64)

    eps = model._eps_nodes[idx]
    eta = model._eta_nodes[idx]

    z = np.asarray(z, dtype=np.float64)

    z_next = model.rho_z[None, :] * z[None, :] + model.sigma_z[None, :] * eps

    R_risky = np.exp(
        (model.kappa_R + model.xi_R * z)[None, :]
        + model.sigma_R[None, :] * eta
    )

    R_safe = np.full((len(idx), 1), model.R, dtype=np.float64)
    R_all = np.concatenate([R_safe, R_risky], axis=1)

    Gamma = model.Gamma_next[idx]
    theta_emp = model._theta_vals[idx]
    sdf_factor = model.SDF_factor[idx]

    return z_next, R_all, Gamma, theta_emp, sdf_factor

# =============================================================================
# Portfolio-choice Bellman evaluation
# =============================================================================


def _portfolio_next_states_for_value(model, state, rs, w, shock_idx):
    """Next states for portfolio Bellman evaluation at one current state."""
    z = np.asarray(state[1:], dtype=np.float64)

    z_next, R_all, Gamma, theta_emp, _ = _portfolio_next_shock_arrays(
        model=model,
        z=z,
        shock_idx=shock_idx,
    )

    Rp = R_all @ w
    base = (Rp / Gamma) * float(rs)

    m_emp = base + theta_emp
    m_unemp = base + model.theta_low

    next_emp = np.column_stack([m_emp, z_next])
    next_unemp = np.column_stack([m_unemp, z_next])

    return next_emp, next_unemp
