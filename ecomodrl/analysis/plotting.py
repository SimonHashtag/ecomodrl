# -----------------------------------------------------------------------------
# Figures for post-training analysis.
#
# State convention:
#   state = (m, x1, ..., xd)
#
# Policy-grid convention:
#   V.shape = (n_m, n_x1, ..., n_xd)
#
# For d >= 1, plots vary the first exogenous state over low/middle/high slices
# and hold all remaining exogenous states at their grid medians.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ecomodrl.algorithms.training import ALGO_REGISTRY
from ecomodrl.analysis.efficiency_metrics import (
    last_value_carried_forward, 
    normalized_regret,
)


DEFAULT_ALGORITHMS = tuple(ALGO_REGISTRY.keys())

ALGO_COLORS = {
    "ppo": "tab:red",
    "ddpg": "tab:blue",
    "td3": "tab:orange",
    "sac": "tab:green",
}

SLICE_COLORS = ["tab:blue", "tab:purple", "tab:brown"]

MODEL_LABELS = {
    "CassKoopmans": "Cass--Koopmans",
    "BrockMirman": "Brock--Mirman",
    "BufferStock": "Buffer Stock",
    "PortfolioChoice": "Portfolio Choice",
}

EGM_COLOR = "0.45"

PORTFOLIO_LEVEL_CMAP = "viridis"
PORTFOLIO_DIFF_CMAP = "RdBu_r"


def ensure_parent(path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def median_iqr(stack, axis=0):
    stack = np.asarray(stack, dtype=np.float64)
    med = np.nanmedian(stack, axis=axis)
    q1 = np.nanpercentile(stack, 25, axis=axis)
    q3 = np.nanpercentile(stack, 75, axis=axis)
    return med, q1, q3


def positive_m_grid(model):
    return np.asarray(model.m_grid)[np.asarray(model.m_grid) > 0.0]


def is_deterministic_one_state(model) -> bool:
    return model.weights is None and model.exogenous_dim == 0


def exogenous_plot_slices(model):
    """Return exogenous states for plotting.

    If there are no exogenous states, returns `[None]`.

    If there is one or more exogenous state, vary the first exogenous variable
    over low/mid/high and hold remaining exogenous variables fixed at their
    median grid point.
    """
    if model.exogenous_dim == 0:
        return [None], [""]

    grids = model.exogenous_grids

    first = grids[0]
    first_values = [
        first[0],
        first[len(first) // 2],
        first[-1],
    ]

    fixed = []
    for grid in grids[1:]:
        fixed.append(grid[len(grid) // 2])

    slices = []
    labels = []

    for value, label in zip(first_values, ["low", "middle", "high"]):
        x = np.asarray([value, *fixed], dtype=np.float64)
        slices.append(x)
        labels.append(fr"$x_1$ {label}")

    return slices, labels


def states_for_m_slice(m_grid, x_slice):
    if x_slice is None:
        return m_grid[:, None]

    x = np.broadcast_to(np.asarray(x_slice), (len(m_grid), len(x_slice)))
    return np.column_stack([m_grid, x])


def value_slice_indices(model):
    """Index tuple for selecting a line over m from V(m, x1, ..., xd)."""
    if model.exogenous_dim == 0:
        return [()], [""]

    grids = model.exogenous_grids

    first_indices = [0, len(grids[0]) // 2, len(grids[0]) - 1]
    fixed = tuple(len(g) // 2 for g in grids[1:])

    indices = []
    labels = []

    for idx, label in zip(first_indices, ["low", "middle", "high"]):
        indices.append((idx, *fixed))
        labels.append(fr"$x_1$ {label}")

    return indices, labels


def select_value_line(V, exog_index):
    """Select V(m, fixed exogenous state)."""
    if exog_index == ():
        return V

    return V[(slice(None), *exog_index)]


def cev_map(model, V_rl, V_egm):
    gamma = model.params.gamma

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if gamma == 1.0:
            return np.exp((V_egm - V_rl) * (1.0 - model.params.beta)) - 1.0

        return (V_egm / V_rl) ** (1.0 / (1.0 - gamma)) - 1.0


def subplot_grid(n):
    n = int(n)
    if n <= 1:
        return 1, 1
    if n <= 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    return int(np.ceil(n / 3)), 3


# =============================================================================
# Policy distribution
# =============================================================================

def plot_policy_distribution(
    model,
    egm_policy,
    rl_policies_by_algo,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
):
    """Plot consumption-share policies.

    Each panel shows:
        - EGM benchmark;
        - median RL policy;
        - IQR band across training seeds.
    """
    m = positive_m_grid(model)
    x_slices, slice_labels = exogenous_plot_slices(model)
    
    egm_shares = []
    for x in x_slices:
        states = states_for_m_slice(m, x)
        egm_shares.append(np.asarray(egm_policy.share(states), dtype=np.float64))

    rl_stacks = {}

    for algo in algorithms:
        policies = rl_policies_by_algo.get(algo, [])
        if not policies:
            continue

        stacks_for_slices = []

        for x in x_slices:
            states = states_for_m_slice(m, x)
            stack = np.stack(
                [np.asarray(policy.share(states), dtype=np.float64) for policy in policies],
                axis=0,
            )
            stacks_for_slices.append(stack)

        rl_stacks[algo] = stacks_for_slices

    # y-axis range
    candidates = [np.asarray(egm_shares).ravel()]
    for stacks in rl_stacks.values():
        for stack in stacks:
            candidates.append(stack[np.isfinite(stack)].ravel())

    finite = np.concatenate([x for x in candidates if x.size])
    y_min = max(0.0, float(np.nanpercentile(finite, 1)) if finite.size else 0.0)
    y_max = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
    y_max *= 1.05

    rows, cols = subplot_grid(len(algorithms))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, algo in zip(axes, algorithms):
        for i, egm_share in enumerate(egm_shares):
            label = "EGM" if model.exogenous_dim == 0 else f"EGM {slice_labels[i]}"
            color = EGM_COLOR if model.exogenous_dim == 0 else SLICE_COLORS[i]
            ax.plot(m, egm_share, color=color, linestyle="--", linewidth=1.8, alpha=0.8, label=label)

        stacks = rl_stacks.get(algo)

        if stacks is None:
            ax.text(0.5, 0.5, "no policies", ha="center", va="center", transform=ax.transAxes, color="0.5")
        else:
            for i, stack in enumerate(stacks):
                med, q1, q3 = median_iqr(stack, axis=0)
                color = ALGO_COLORS.get(algo, "tab:blue") if model.exogenous_dim == 0 else SLICE_COLORS[i]

                ax.fill_between(m, q1, q3, color=color, alpha=0.22)
                ax.plot(m, med, color=color, linewidth=1.8)

        ax.set_title(algo.upper())
        ax.set_xlabel("Cash-on-hand $m$")
        ax.set_ylabel("Consumption share $c/m$")
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(algorithms):]:
        ax.axis("off")

    fig.suptitle("Consumption share policy: median and IQR across training seeds")
    plt.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    

# =============================================================================
# Portfolio-share policy plots
# =============================================================================

def _risky_asset_label(asset_index: int) -> str:
    return fr"risky asset {asset_index}"


def portfolio_share_grid(model, policy, m_grid, x1_grid, asset_index=1):
    """Evaluate a policy's portfolio share of one risky asset over (m, x1).

    All exogenous predictors other than the first (x1) are held at their
    unconditional mean of zero. Returns an array of shape
    (len(m_grid), len(x1_grid)) where entry [i, j] is the portfolio weight on
    the requested risky asset at state (m_grid[i], x1_grid[j], 0, ..., 0).

    Asset order is [safe, risky_1, ..., risky_N], so asset_index=1 selects the
    first risky asset.
    """
    MM, XX = np.meshgrid(m_grid, x1_grid, indexing="ij")

    states = np.zeros((MM.size, model.state_dim), dtype=np.float64)
    states[:, 0] = MM.ravel()
    states[:, 1] = XX.ravel()

    controls = policy.controls(states)
    weights = np.asarray(controls["portfolio_weights"], dtype=np.float64)

    share = weights[..., asset_index]
    return share.reshape(MM.shape)


def median_rl_portfolio_share_grid(model, policies, m_grid, x1_grid, asset_index=1):
    """Median RL portfolio share across training seeds on the (m, x1) grid."""
    stacks = [
        portfolio_share_grid(model, policy, m_grid, x1_grid, asset_index)
        for policy in policies
    ]
    return np.nanmedian(np.stack(stacks, axis=0), axis=0)


def _portfolio_mesh(m_grid, x1_grid):
    """pcolormesh coordinate arrays with m on the x-axis and x1 on the y-axis."""
    return np.meshgrid(m_grid, x1_grid, indexing="xy")


def plot_portfolio_share_heatmaps(
    model,
    rl_policies_by_algo,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    asset_index=1,
):
    """Heatmaps of the median RL portfolio share of one risky asset.

    One panel per algorithm. The x-axis is cash-on-hand m, the y-axis is the
    predictor x_j (j=asset_index), and the color encodes the median (across training
    seeds) RL portfolio weight on the requested risky asset. All other
    predictors are held at their unconditional mean of zero.
    """
    m_grid = positive_m_grid(model)
    x_grid = np.asarray(model.exogenous_grids[asset_index-1], dtype=np.float64)

    share_grids = {}
    for algo in algorithms:
        policies = rl_policies_by_algo.get(algo, [])
        if not policies:
            continue
        share_grids[algo] = median_rl_portfolio_share_grid(
            model, policies, m_grid, x_grid, asset_index
        )

    if share_grids:
        pooled = np.concatenate([g.ravel() for g in share_grids.values()])
        pooled = pooled[np.isfinite(pooled)]
        vmin = float(np.nanpercentile(pooled, 1)) if pooled.size else 0.0
        vmax = float(np.nanpercentile(pooled, 99)) if pooled.size else 1.0
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    M, X1 = _portfolio_mesh(m_grid, x_grid)

    rows, cols = subplot_grid(len(algorithms))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.6 * cols, 3.8 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for ax, algo in zip(axes, algorithms):
        grid = share_grids.get(algo)

        if grid is None:
            ax.text(
                0.5, 0.5, "no policies", ha="center", va="center",
                transform=ax.transAxes, color="0.5",
            )
            ax.set_title(algo.upper())
            continue

        mesh = ax.pcolormesh(
            M, X1, grid.T,
            shading="auto",
            cmap=PORTFOLIO_LEVEL_CMAP,
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(algo.upper())
        ax.set_xlabel(r"Cash-on-hand $m$")
        ax.set_ylabel(r"Predictor $x_{" + str(asset_index) + r"}$")

    for ax in axes[len(algorithms):]:
        ax.axis("off")

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.9)
        cbar.set_label(
            fr"Median RL portfolio share on "
            fr"$w_{{{asset_index}}}$"
        )

    fig.suptitle(
        fr"RL portfolio share of {_risky_asset_label(asset_index)}: "
        fr"median across training seeds"
    )
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_portfolio_share_diff_heatmaps(
    model,
    egm_policy,
    rl_policies_by_algo,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    asset_index=1,
):
    """Heatmaps of median RL minus EGM portfolio share (percentage points).

    One panel per algorithm. Positive (red) values mean the RL policy holds
    more of the risky asset than the EGM benchmark; negative (blue) values mean
    it holds less. The color scale is symmetric around zero, so hue and
    saturation are directly comparable across panels.
    """
    m_grid = positive_m_grid(model)
    x_grid = np.asarray(model.exogenous_grids[asset_index-1], dtype=np.float64)

    egm_grid = portfolio_share_grid(model, egm_policy, m_grid, x_grid, asset_index)

    diff_grids = {}
    for algo in algorithms:
        policies = rl_policies_by_algo.get(algo, [])
        if not policies:
            continue
        rl_grid = median_rl_portfolio_share_grid(
            model, policies, m_grid, x_grid, asset_index
        )
        diff_grids[algo] = 100.0 * (rl_grid - egm_grid)

    if diff_grids:
        pooled = np.concatenate([g.ravel() for g in diff_grids.values()])
        pooled = pooled[np.isfinite(pooled)]
        vmax = float(np.nanpercentile(np.abs(pooled), 99)) if pooled.size else 1.0
    else:
        vmax = 1.0
    vmax = max(vmax, 1e-6)
    vmin = -vmax

    M, X1 = _portfolio_mesh(m_grid, x_grid)

    rows, cols = subplot_grid(len(algorithms))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.6 * cols, 3.8 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for ax, algo in zip(axes, algorithms):
        grid = diff_grids.get(algo)

        if grid is None:
            ax.text(
                0.5, 0.5, "no policies", ha="center", va="center",
                transform=ax.transAxes, color="0.5",
            )
            ax.set_title(algo.upper())
            continue

        mesh = ax.pcolormesh(
            M, X1, grid.T,
            shading="auto",
            cmap=PORTFOLIO_DIFF_CMAP,
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(algo.upper())
        ax.set_xlabel(r"Cash-on-hand $m$")
        ax.set_ylabel(r"Predictor $x_{" + str(asset_index) + r"}$")

    for ax in axes[len(algorithms):]:
        ax.axis("off")

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.9)
        cbar.set_label(
            fr"RL $-$ EGM portfolio share on w_{{{asset_index}}} [pp]"
        )

    fig.suptitle(
        fr"RL vs EGM portfolio share of {_risky_asset_label(asset_index)}: "
        fr"median deviation"
    )
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_portfolio_share_slices(
    model,
    egm_policy,
    rl_policies_by_algo,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    asset_index=1,
):
    """Optional complement to the heatmaps.

    Portfolio share of one risky asset as a function of m for low/middle/high
    x_j slices (j=asset_index), showing EGM (dashed) and the RL median with an IQR band across
    training seeds. Unlike the heatmaps, this figure exposes cross-seed
    dispersion, which is useful for judging whether the RL policy is reliable.
    """
    m = positive_m_grid(model)
    x_grid = np.asarray(model.exogenous_grids[asset_index - 1], dtype=np.float64)

    x_values = [x_grid[0], x_grid[len(x_grid) // 2], x_grid[-1]]
    slice_labels = ["low", "middle", "high"]

    def states_for(x):
        s = np.zeros((len(m), model.state_dim), dtype=np.float64)
        s[:, 0] = m
        s[:, asset_index] = x
        return s

    egm_lines = []
    for x in x_values:
        w = np.asarray(
            egm_policy.controls(states_for(x))["portfolio_weights"],
            dtype=np.float64,
        )
        egm_lines.append(w[..., asset_index])

    rows, cols = subplot_grid(len(algorithms))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.5 * cols, 3.5 * rows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, algo in zip(axes, algorithms):
        policies = rl_policies_by_algo.get(algo, [])

        for i, x in enumerate(x_values):
            color = SLICE_COLORS[i]
            ax.plot(
                m, egm_lines[i],
                color=color, linestyle="--", linewidth=1.6, alpha=0.8,
                label=fr"EGM $x_{{{asset_index}}}$ {slice_labels[i]}",
            )

            if policies:
                stack = np.stack(
                    [
                        np.asarray(
                            p.controls(states_for(x))["portfolio_weights"],
                            dtype=np.float64,
                        )[..., asset_index]
                        for p in policies
                    ],
                    axis=0,
                )
                med, q1, q3 = median_iqr(stack, axis=0)
                ax.fill_between(m, q1, q3, color=color, alpha=0.20)
                ax.plot(
                    m, med, color=color, linewidth=1.8,
                    label=fr"RL $x_{{{asset_index}}}$ {slice_labels[i]}",
                )

        if not policies:
            ax.text(
                0.5, 0.5, "no policies", ha="center", va="center",
                transform=ax.transAxes, color="0.5",
            )

        ax.set_title(algo.upper())
        ax.set_xlabel(r"Cash-on-hand $m$")
        ax.set_ylabel(fr"Portfolio share $w_{{{asset_index}}}$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    for ax in axes[len(algorithms):]:
        ax.axis("off")

    fig.suptitle(
        fr"Portfolio share of {_risky_asset_label(asset_index)}: "
        fr"EGM vs RL (median and IQR)"
    )
    plt.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


# =============================================================================
# CEV distribution
# =============================================================================

def _pooled_visited(visited_m, algorithms):
    """Pool visited cash-on-hand across algorithms into one array (or None)."""
    if not visited_m:
        return None

    arrays = []
    for algo in algorithms:
        v = visited_m.get(algo)
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            arrays.append(v)

    if not arrays:
        return None

    return np.concatenate(arrays)


def plot_cev_distribution(
    model,
    grid_axes,
    V_egm,
    rl_value_grids_by_algo,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    visited_m=None,
    support_quantiles=(0.1, 99.9),
):
    """Plot CEV relative to the EGM benchmark.
    We trim data-adaptively to the ergodic support: the plotted range is 
    restricted to the [support_quantiles] range of the pooled visited-state 
    distribution.
    """
    m_full = np.asarray(grid_axes[0], dtype=np.float64)

    # Data-adaptive ergodic-support trim
    pooled = _pooled_visited(visited_m, algorithms)
    if pooled is not None and pooled.size:
        lo, hi = np.nanpercentile(pooled, support_quantiles)
        mask = (m_full >= lo) & (m_full <= hi)

    # Safety fallback if we trimmed too aggressively
    if mask.sum() < 5:
        mask = np.ones(m_full.size, dtype=bool)

    m_grid = m_full[mask]

    exog_indices, slice_labels = value_slice_indices(model)
    deterministic = is_deterministic_one_state(model)

    n_algos = len(algorithms)
    fig, axes = plt.subplots(
        1,
        n_algos,
        figsize=(4.5 * n_algos, 3.8),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    positive_values_for_limits = []

    for ax, algo in zip(axes, algorithms):
        value_grids = rl_value_grids_by_algo.get(algo, [])

        legend_handles = []
        legend_labels = []

        if not value_grids:
            ax.text(
                0.5, 0.5, "no value grids", ha="center", va="center",
                transform=ax.transAxes, color="0.5",
            )
        else:
            cev_stack = np.stack(
                [cev_map(model, V_rl, V_egm) for V_rl in value_grids],
                axis=0,
            )

            for i, exog_idx in enumerate(exog_indices):
                line_stack = np.stack(
                    [select_value_line(cev_arr, exog_idx) for cev_arr in cev_stack],
                    axis=0,
                )
                line_stack = line_stack[:, mask] * 100.0

                med, q1, q3 = median_iqr(line_stack, axis=0)

                color = (
                    ALGO_COLORS.get(algo, "tab:blue")
                    if model.exogenous_dim == 0
                    else SLICE_COLORS[i]
                )
                label = None if model.exogenous_dim == 0 else slice_labels[i]

                ax.fill_between(m_grid, q1, q3, color=color, alpha=0.22, linewidth=0.0)
                line, = ax.plot(m_grid, med, color=color, linewidth=1.7, label=label)

                if label is not None:
                    legend_handles.append(line)
                    legend_labels.append(label)

                positive_values_for_limits.append(q1)
                positive_values_for_limits.append(q3)
                positive_values_for_limits.append(med)

        # State distribution (drawn behind the CEV curves).
        if not deterministic and visited_m is not None and algo in visited_m:
            visited = np.asarray(visited_m[algo], dtype=np.float64)
            visited = visited[np.isfinite(visited)]
            visited = visited[(visited >= m_grid[0]) & (visited <= m_grid[-1])]

            if visited.size:
                ax2 = ax.twinx()
                ax2.hist(
                    visited, bins=60, density=True,
                    color="0.5", alpha=0.38, edgecolor="0.35", linewidth=0.3,
                )
                ax2.set_yticks([])
                ax2.set_zorder(ax.get_zorder() - 1)
                ax.patch.set_visible(False)

        if legend_handles:
            ax.legend(legend_handles, legend_labels, fontsize=8, loc="best")

        ax.set_title(algo.upper())
        ax.set_xlabel("Cash-on-hand $m$")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")

    axes[0].set_ylabel(r"CEV relative to EGM, $\lambda_{\mathrm{CEV}}$ [%]")

    # Shared positive y-limits (robust percentiles).
    if positive_values_for_limits:
        vals = np.concatenate(
            [np.asarray(v, dtype=np.float64).ravel() for v in positive_values_for_limits]
        )
        vals = vals[np.isfinite(vals) & (vals > 0.0)]

        if vals.size:
            y_lo, y_hi = np.nanpercentile(vals, [1.0, 99.0])

            if y_lo <= 0.0 or not np.isfinite(y_lo):
                y_lo = np.nanmin(vals)
            if y_hi <= y_lo or not np.isfinite(y_hi):
                y_hi = np.nanmax(vals)

            y_lo = max(y_lo / 1.25, np.finfo(float).tiny)
            y_hi = y_hi * 1.25

            for ax in axes:
                ax.set_ylim(y_lo, y_hi)

    for ax in axes:
        ax.set_xlim(m_grid[0], m_grid[-1])

    fig.suptitle("Consumption-equivalent gain of EGM over RL")
    plt.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


# =============================================================================
# Euler-error distribution
# =============================================================================

def plot_euler_error_distribution(
    model,
    euler_pools_by_algo,
    euler_seedwise_by_algo,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
):
    deterministic = is_deterministic_one_state(model)

    if deterministic:
        fig, ax = plt.subplots(figsize=(9, 5))

        for algo in algorithms:
            arrays = euler_seedwise_by_algo.get(algo, [])
            if not arrays:
                continue

            min_len = min(len(a) for a in arrays)
            stack = np.stack([np.asarray(a[:min_len], dtype=np.float64) for a in arrays], axis=0)
            med, q1, q3 = median_iqr(stack, axis=0)

            x = model.log_grid(model.m_grid[0], model.m_grid[-1], min_len)
            color = ALGO_COLORS.get(algo, "tab:blue")

            ax.fill_between(x, q1, q3, color=color, alpha=0.22)
            ax.plot(x, med, color=color, linewidth=1.3, label=algo.upper())

        ax.set_xscale("log")
        ax.set_xlabel("Cash-on-hand $m$")
        ax.set_ylabel(r"$\log_{10}|1 - c^{\mathrm{implied}}/c|$")
        ax.set_title("Euler errors across deterministic test grid")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    else:
        rows = len(algorithms)
        fig, axes = plt.subplots(rows, 1, figsize=(9, 2.5 * rows), sharex=True)
        axes = np.atleast_1d(axes)

        all_values = []
        for algo in algorithms:
            for arr in euler_pools_by_algo.get(algo, []):
                x = np.asarray(arr, dtype=np.float64).ravel()
                x = x[np.isfinite(x)]
                if x.size:
                    all_values.append(x)

        if all_values:
            pooled_all = np.concatenate(all_values)
            lo, hi = np.nanpercentile(pooled_all, [0.5, 99.5])
            bins = np.linspace(lo, hi, 80)
        else:
            bins = 80

        for ax, algo in zip(axes, algorithms):
            arrays = euler_pools_by_algo.get(algo, [])

            if not arrays:
                ax.text(0.5, 0.5, "no Euler errors", ha="center", va="center", transform=ax.transAxes, color="0.5")
            else:
                pooled = np.concatenate([np.asarray(a).ravel() for a in arrays])
                pooled = pooled[np.isfinite(pooled)]

                weights = np.ones_like(pooled) / max(pooled.size, 1)
                ax.hist(
                    pooled,
                    bins=bins,
                    weights=weights,
                    color=ALGO_COLORS.get(algo, "tab:blue"),
                    edgecolor="black",
                    linewidth=0.25,
                )

                ax.set_ylabel(f"{algo.upper()}\n$n={pooled.size}$")

            ax.grid(alpha=0.3)

        axes[-1].set_xlabel(r"$\log_{10}|1 - c^{\mathrm{implied}}/c|$")
        fig.text(0.02, 0.5, "Relative frequency", rotation="vertical", va="center")
        fig.suptitle("Euler-error distributions across policies and test seeds")

    plt.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


# =============================================================================
# Convergence
# =============================================================================


def plot_training_convergence(
    histories_by_algo,
    egm_returns_by_algo,
    wallclock_seconds,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    grid_step=5.0,
    efficiency=None,
):
    """Plot convergence as log10 of the EGM-normalized regret over training time.

    If `efficiency` is passed, each algorithm's AULC is added to the legend and a 
    faint vertical marker is drawn at its median time-to-converged.
    """
    t_grid = np.arange(0.0, float(wallclock_seconds) + 1e-9, float(grid_step))

    fig, ax = plt.subplots(figsize=(9, 5))

    for algo in algorithms:
        histories = histories_by_algo.get(algo, [])
        egm_returns = egm_returns_by_algo.get(algo, [])

        if not histories or len(histories) != len(egm_returns):
            continue

        rl_returns = np.stack(
            [last_value_carried_forward(t, u, t_grid) for t, u in histories],
            axis=0,
        )

        reg_norm = normalized_regret(egm_returns, rl_returns)
        med, q1, q3 = median_iqr(reg_norm, axis=0)

        color = ALGO_COLORS.get(algo, "tab:blue")

        label = algo.upper()
        if efficiency is not None:
            aulc = efficiency.get(algo, {}).get("aulc", {})
            if aulc.get("n", 0):
                label = f"{algo.upper()} (AULC={aulc['iqm']:.4f})"

        ax.fill_between(t_grid, q1, q3, color=color, alpha=0.22)
        ax.plot(t_grid, med, color=color, linewidth=1.6, label=label)

        if efficiency is not None:
            conv = efficiency.get(algo, {}).get("time_to_converged", {})
            if conv.get("n", 0):
                ax.axvline(conv["median"], color=color, linestyle=":", linewidth=1.4, alpha=0.9)

    ax.set_xlabel("Wall-clock seconds")
    ax.set_ylabel(r"$\frac{U_{\mathrm{EGM}}-U_{\mathrm{RL}}}{|U_{\mathrm{EGM}}|}$")
    ax.set_yscale("log")
    ax.set_title("Convergence of RL policies relative to EGM")
    ax.legend(fontsize=9)

    plt.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)