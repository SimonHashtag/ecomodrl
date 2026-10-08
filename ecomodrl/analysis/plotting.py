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
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

from ecomodrl.algorithms.training import ALGO_REGISTRY
from ecomodrl.analysis.efficiency_metrics import (
    last_value_carried_forward,
    normalized_regret,
)
from ecomodrl.analysis.accuracy_metrics import iqm
from ecomodrl.configs.variants import VARIANTS, VARIANT_LABELS


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

# Combined mf-vs-pd figures: model-free layer in grey, post-decision in the
# algorithm colour; line styles distinguish the variants in single-axis plots.
MF_COLOR = "0.55"
VARIANT_LINESTYLES = {"pd": "-", "mf": "--"}

PORTFOLIO_LEVEL_CMAP = "viridis"
PORTFOLIO_DIFF_CMAP = "RdBu_r"
PORTFOLIO_UNIDENTIFIED_COLOR = "0.85"
DEFAULT_SAVING_SHARE_TOL = 1e-3
DEFAULT_MIN_IDENTIFIED_FRACTION = 0.75


def ensure_parent(path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def positive_m_grid(model):
    return np.asarray(model.m_grid)[np.asarray(model.m_grid) > 0.0]


def algorithm_label(algo: str, variant: str = "mf") -> str:
    """Display label for an algorithm in a single-variant figure."""
    label = str(algo).upper()
    return f"PD-{label}" if variant == "pd" else label


def exogenous_state_label(model, index: int = 1) -> str:
    """Math label for an exogenous state, using model-specific notation."""
    model_name = type(model).__name__
    if model_name == "BrockMirman":
        return r"$z_t$"
    if model_name == "PortfolioChoice":
        return fr"$z_{{{int(index)},t}}$"
    return fr"$x_{{{int(index)}}}$"


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
        labels.append(f"{exogenous_state_label(model, 1)} {label}")

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
        labels.append(f"{exogenous_state_label(model, 1)} {label}")

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


def iqm_and_iqr(values, axis=0):
    """Return the interquartile mean and the Q25--Q75 interval."""
    values = np.asarray(values, dtype=np.float64)
    center = iqm(values, axis=axis)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        q1, q3 = np.nanpercentile(values, [25, 75], axis=axis)
    return center, q1, q3


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
    variant="mf",
):
    """Plot consumption-share policies.

    Each panel shows:
        - EGM benchmark;
        - IQM RL policy;
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
    y_min = max(0.0, float(np.nanpercentile(finite, 5.0) - 0.05) if finite.size else 0.0)
    y_max = min(1.05, 0.05 + float(np.nanpercentile(finite, 95.0)) if finite.size else 1.0)

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
                iqm_values, q1, q3 = iqm_and_iqr(stack)
                color = ALGO_COLORS.get(algo, "tab:blue") if model.exogenous_dim == 0 else SLICE_COLORS[i]

                ax.fill_between(m, q1, q3, color=color, alpha=0.22)
                ax.plot(m, iqm_values, color=color, linewidth=1.8)

        ax.set_title(algorithm_label(algo, variant))
        ax.set_xlabel("Cash-on-hand $m$")
        ax.set_ylabel("Consumption share $c/m$")
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.3)

    for ax in axes[len(algorithms):]:
        ax.axis("off")
    if x_slices[0] is not None:
        handles = [Line2D([0], [0], color=SLICE_COLORS[i], label=label)
                for i, label in enumerate(slice_labels) if label]
        labels = [handle.get_label() for handle in handles]
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
        plt.tight_layout(rect=(0, 0.07, 1, 1))
    else:
        plt.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    

# =============================================================================
# Portfolio-share policy plots
# =============================================================================

def _risky_asset_label(asset_index: int) -> str:
    return fr"risky asset {asset_index}"


def _portfolio_share_and_identification(
    policy,
    states,
    *,
    asset_index=1,
    saving_share_tol=DEFAULT_SAVING_SHARE_TOL,
):
    """Return risky share, saving share, and portfolio identification mask."""
    saving_share_tol = float(saving_share_tol)
    states = np.asarray(states, dtype=np.float64)
    controls = policy.controls(states)
    weights = np.asarray(controls["portfolio_weights"], dtype=np.float64)
    consumption = np.asarray(controls["consumption"], dtype=np.float64)

    m = states[..., 0]
    saved_resources = np.maximum(m - consumption, 0.0)
    saving_share = saved_resources / np.maximum(m, np.finfo(float).tiny)
    risky_share = weights[..., asset_index]

    identified = (saving_share > saving_share_tol)
    return risky_share, saving_share, identified


def _portfolio_share_grid_with_identification(
    model,
    policy,
    m_grid,
    x1_grid,
    asset_index=1,
    *,
    saving_share_tol=DEFAULT_SAVING_SHARE_TOL,
):
    """Evaluate portfolio shares and identification over an (m, x1) grid."""
    MM, XX = np.meshgrid(m_grid, x1_grid, indexing="ij")

    states = np.zeros((MM.size, model.state_dim), dtype=np.float64)
    states[:, 0] = MM.ravel()
    states[:, asset_index] = XX.ravel()

    share, saving_share, identified = _portfolio_share_and_identification(
        policy,
        states,
        asset_index=asset_index,
        saving_share_tol=saving_share_tol,
    )
    shape = MM.shape
    return (
        np.asarray(share, dtype=np.float64).reshape(shape),
        np.asarray(saving_share, dtype=np.float64).reshape(shape),
        np.asarray(identified, dtype=bool).reshape(shape),
    )


def _rl_portfolio_share_stack(
    model,
    policies,
    m_grid,
    x1_grid,
    asset_index,
    *,
    saving_share_tol,
):
    """Stack RL portfolio shares and identification masks over training seeds."""
    shares = []
    identified = []
    for policy in policies:
        share, _, mask = _portfolio_share_grid_with_identification(
            model,
            policy,
            m_grid,
            x1_grid,
            asset_index,
            saving_share_tol=saving_share_tol,
        )
        shares.append(share)
        identified.append(mask)
    return np.stack(shares, axis=0), np.stack(identified, axis=0)


def iqm_rl_portfolio_share_grid(
    model,
    policies,
    m_grid,
    x1_grid,
    asset_index=1,
    *,
    saving_share_tol=DEFAULT_SAVING_SHARE_TOL,
    min_identified_fraction=DEFAULT_MIN_IDENTIFIED_FRACTION,
):
    """IQM RL portfolio share where sufficiently many seeds are identified."""
    min_identified_fraction = float(min_identified_fraction)

    shares, identified = _rl_portfolio_share_stack(
        model,
        policies,
        m_grid,
        x1_grid,
        asset_index,
        saving_share_tol=saving_share_tol,
    )
    identified_fraction = np.mean(identified, axis=0)
    masked_shares = np.where(identified, shares, np.nan)
    center = np.asarray(iqm(masked_shares, axis=0), dtype=np.float64)
    center[identified_fraction < min_identified_fraction] = np.nan
    return center


def _portfolio_mesh(m_grid, x1_grid):
    """pcolormesh coordinate arrays with m on the x-axis and x1 on the y-axis."""
    return np.meshgrid(m_grid, x1_grid, indexing="xy")


def _plot_state_hdr_contours(
    ax,
    states,
    m_grid,
    x1_grid,
    *,
    asset_index=1,
    masses=(0.50, 0.70, 0.90),
    bins=60,
    smoothing=3.0,
):
    """Draw smooth histogram-based Highest Density Region (HDR) contours for visited states.

    Density is estimated on a bounded, robustly extended histogram, so a
    contour can continue beyond the heatmap and be clipped at its boundary.
    Extreme observations outside that estimation domain retain their mass in
    the denominator but are treated as diffuse tail observations. Gaussian
    smoothing suppresses cell-level Monte Carlo noise without imposing an
    elliptical or Gaussian state distribution.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] <= asset_index:
        return []

    xy = states[:, [0, asset_index]]
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.shape[0] < 2:
        return []

    masses = tuple(float(mass) for mass in masses)

    n_bins = int(bins)
    if n_bins < 2:
        raise ValueError(f"HDR bins must be at least two; got {bins}.")

    max_mass = max(masses)
    # Four marginal tails are truncated in two dimensions. Allocate only half
    # of the probability not needed by the largest HDR to those tails, leaving
    # margin for smoothing and overlap between the marginal exclusions.
    tail_quantile = max((1.0 - max_mass) / 8.0, 1.0 / xy.shape[0])
    max_extra_bins = 10 * n_bins

    def extended_edges(data, plot_lo, plot_hi):
        plot_lo = float(plot_lo)
        plot_hi = float(plot_hi)
        step = (plot_hi - plot_lo) / n_bins

        robust_lo, robust_hi = np.quantile(
            data,
            [tail_quantile, 1.0 - tail_quantile],
        )
        target_lo = max(
            float(robust_lo),
            plot_lo - max_extra_bins * step,
        )
        target_hi = min(
            float(robust_hi),
            plot_hi + max_extra_bins * step,
        )
        left_bins = max(
            0,
            int(np.ceil((plot_lo - target_lo) / step)),
        )
        right_bins = max(
            0,
            int(np.ceil((target_hi - plot_hi) / step)),
        )
        return plot_lo + step * np.arange(
            -left_bins - 1,
            n_bins + right_bins + 2,
            dtype=np.float64,
        )

    m_edges = extended_edges(xy[:, 0], m_grid[0], m_grid[-1])
    x_edges = extended_edges(xy[:, 1], x1_grid[0], x1_grid[-1])
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(m_edges, x_edges))
    smoothed_counts = gaussian_filter(
        counts,
        sigma=float(smoothing),
        mode="constant",
    )
    if not np.any(smoothed_counts > 0.0):
        return []

    cell_area = np.diff(m_edges)[:, None] * np.diff(x_edges)[None, :]
    # Normalize by every finite observation, including explosive observations
    # outside the bounded density-estimation domain. This keeps HDR masses
    # unconditional instead of turning them into in-domain conditional masses.
    probability = smoothed_counts / float(xy.shape[0])
    density = probability / cell_area
    order = np.argsort(density.ravel())[::-1]
    cumulative_mass = np.cumsum(probability.ravel()[order])

    m_centers = 0.5 * (m_edges[:-1] + m_edges[1:])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    styles = ("-", "--", "-.")
    widths = (1.2, 1.4, 1.6)
    drawn = []

    for index, mass in enumerate(masses):
        if cumulative_mass[-1] < mass:
            continue
        cutoff_index = int(np.searchsorted(cumulative_mass, mass, side="left"))
        density_cutoff = float(density.ravel()[order[cutoff_index]])
        if not np.isfinite(density_cutoff) or density_cutoff <= 0.0:
            continue

        linestyle = styles[min(index, len(styles) - 1)]
        linewidth = widths[min(index, len(widths) - 1)]
        ax.contour(
            m_centers,
            x_centers,
            density.T,
            levels=[density_cutoff],
            colors="black",
            linestyles=linestyle,
            linewidths=linewidth,
            zorder=3,
        )
        drawn.append((mass, linestyle, linewidth))

    ax.set_xlim(float(m_grid[0]), float(m_grid[-1]))
    ax.set_ylim(float(x1_grid[0]), float(x1_grid[-1]))
    return drawn


def plot_portfolio_share_heatmaps(
    model,
    rl_policies_by_algo,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    asset_index=1,
    saving_share_tol=DEFAULT_SAVING_SHARE_TOL,
    min_identified_fraction=DEFAULT_MIN_IDENTIFIED_FRACTION,
    variant="mf",
):
    """Heatmaps of the IQM RL portfolio share of one risky asset.

    One panel per algorithm. The x-axis is cash-on-hand m, the y-axis is the
    predictor x_j (j=asset_index), and the color encodes the IQM (across training
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
        share_grids[algo] = iqm_rl_portfolio_share_grid(
            model,
            policies,
            m_grid,
            x_grid,
            asset_index,
            saving_share_tol=saving_share_tol,
            min_identified_fraction=min_identified_fraction,
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
    cmap = plt.get_cmap(PORTFOLIO_LEVEL_CMAP).copy()
    cmap.set_bad(PORTFOLIO_UNIDENTIFIED_COLOR)

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
            ax.set_title(algorithm_label(algo, variant))
            continue

        mesh = ax.pcolormesh(
            M, X1, np.ma.masked_invalid(grid.T),
            shading="auto",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            rasterized=True,
        )
        ax.set_title(algorithm_label(algo, variant))
        ax.set_xlabel(r"Cash-on-hand $m$")
        ax.set_ylabel(f"Predictor {exogenous_state_label(model, asset_index)}")

    for ax in axes[len(algorithms):]:
        ax.axis("off")

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.9)
        cbar.set_label(
            fr"IQM RL portfolio share on "
            fr"$w_{{{asset_index}}}$"
        )

    # fig.suptitle(
    #     fr"RL portfolio share of {_risky_asset_label(asset_index)}: "
    #     fr"IQM across training seeds"
    # )
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_portfolio_share_diff_heatmaps(
    model,
    egm_policy,
    rl_policies_by_algo,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    asset_index=1,
    states=None,
    hdr_masses=(0.50, 0.70, 0.90),
    hdr_bins=60,
    saving_share_tol=DEFAULT_SAVING_SHARE_TOL,
    min_identified_fraction=DEFAULT_MIN_IDENTIFIED_FRACTION,
    variant="mf",
):
    """Heatmaps of IQM RL minus EGM portfolio share (percentage points).

    One panel per algorithm. Positive (red) values mean the RL policy holds
    more of the risky asset than the EGM benchmark; negative (blue) values mean
    it holds less. The color scale is symmetric around zero, so hue and
    saturation are directly comparable across panels. When supplied, the same
    histogram-based EGM state-distribution HDR is overlaid on every panel.
    """
    m_grid = positive_m_grid(model)
    x_grid = np.asarray(model.exogenous_grids[asset_index-1], dtype=np.float64)

    min_identified_fraction = float(min_identified_fraction)

    egm_grid, _, egm_identified = _portfolio_share_grid_with_identification(
        model,
        egm_policy,
        m_grid,
        x_grid,
        asset_index,
        saving_share_tol=saving_share_tol,
    )

    diff_grids = {}
    for algo in algorithms:
        policies = rl_policies_by_algo.get(algo, [])
        if not policies:
            continue
        rl_shares, rl_identified = _rl_portfolio_share_stack(
            model,
            policies,
            m_grid,
            x_grid,
            asset_index,
            saving_share_tol=saving_share_tol,
        )
        valid = rl_identified & egm_identified[None, ...]
        diff_stack = np.where(
            valid,
            100.0 * (rl_shares - egm_grid[None, ...]),
            np.nan,
        )
        identified_fraction = np.mean(rl_identified, axis=0)
        diff_grid = np.asarray(iqm(diff_stack, axis=0), dtype=np.float64)
        display_mask = (
            egm_identified
            & (identified_fraction >= min_identified_fraction)
        )
        diff_grid[~display_mask] = np.nan
        diff_grids[algo] = diff_grid

    if diff_grids:
        pooled = np.concatenate([g.ravel() for g in diff_grids.values()])
        pooled = pooled[np.isfinite(pooled)]
        vmax = float(np.nanpercentile(np.abs(pooled), 99)) if pooled.size else 1.0
    else:
        vmax = 1.0
    vmax = max(vmax, 1e-6)
    vmin = -vmax

    M, X1 = _portfolio_mesh(m_grid, x_grid)
    cmap = plt.get_cmap(PORTFOLIO_DIFF_CMAP).copy()
    cmap.set_bad(PORTFOLIO_UNIDENTIFIED_COLOR)

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
            ax.set_title(algorithm_label(algo, variant))
            continue

        mesh = ax.pcolormesh(
            M, X1, np.ma.masked_invalid(grid.T),
            shading="auto",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            rasterized=True,
        )
        ax.set_title(algorithm_label(algo, variant))
        ax.set_xlabel(r"Cash-on-hand $m$")
        ax.set_ylabel(f"Predictor {exogenous_state_label(model, asset_index)}")

        algo_states = states.get(algo) if states else None
        if algo_states is not None and len(algo_states):
            drawn_hdrs = _plot_state_hdr_contours(
                ax,
                states[algo],
                m_grid,
                x_grid,
                asset_index=asset_index,
                masses=hdr_masses,
                bins=hdr_bins,
            )
            for _, linestyle, linewidth in drawn_hdrs:
                ax.plot(
                    [],
                    [],
                    color="black",
                    linestyle=linestyle,
                    linewidth=linewidth,
                )

    for ax in axes[len(algorithms):]:
        ax.axis("off")

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.9)
        cbar.set_label(
            fr"RL $-$ EGM portfolio share on $w_{{{asset_index}}}$ [pp]"
        )

    # fig.suptitle(
    #     fr"RL vs EGM portfolio share of {_risky_asset_label(asset_index)}: "
    #     fr"IQM deviation"
    # )
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_portfolio_share_slices(
    model,
    egm_policy,
    rl_policies_by_algo,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    asset_index=1,
    saving_share_tol=DEFAULT_SAVING_SHARE_TOL,
    min_identified_fraction=DEFAULT_MIN_IDENTIFIED_FRACTION,
    variant="mf",
):
    """Optional complement to the heatmaps.

    Portfolio share of one risky asset as a function of m for low/middle/high
    x_j slices (j=asset_index), showing EGM (dashed) and the RL IQM with an IQR band across
    training seeds. Unlike the heatmaps, this figure exposes cross-seed
    dispersion, which is useful for judging whether the RL policy is reliable.
    """
    m = positive_m_grid(model)
    x_grid = np.asarray(model.exogenous_grids[asset_index - 1], dtype=np.float64)

    x_values = [x_grid[0], x_grid[len(x_grid) // 2], x_grid[-1]]
    slice_labels = ["low", "middle", "high"]
    min_identified_fraction = float(min_identified_fraction)

    def states_for(x):
        s = np.zeros((len(m), model.state_dim), dtype=np.float64)
        s[:, 0] = m
        s[:, asset_index] = x
        return s

    egm_lines = []
    for x in x_values:
        share, _, identified = _portfolio_share_and_identification(
            egm_policy,
            states_for(x),
            asset_index=asset_index,
            saving_share_tol=saving_share_tol,
        )
        egm_lines.append(np.where(identified, share, np.nan))

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
                label=f"EGM {exogenous_state_label(model, asset_index)} {slice_labels[i]}",
            )

            if policies:
                slice_states = states_for(x)
                shares = []
                identified = []
                for policy in policies:
                    share, _, mask = _portfolio_share_and_identification(
                        policy,
                        slice_states,
                        asset_index=asset_index,
                        saving_share_tol=saving_share_tol,
                    )
                    shares.append(share)
                    identified.append(mask)

                share_stack = np.stack(shares, axis=0)
                identified_stack = np.stack(identified, axis=0)
                identified_fraction = np.mean(identified_stack, axis=0)
                masked_stack = np.where(
                    identified_stack,
                    share_stack,
                    np.nan,
                )
                iqm_values, q1, q3 = iqm_and_iqr(masked_stack)
                display_mask = identified_fraction >= min_identified_fraction
                iqm_values = np.where(display_mask, iqm_values, np.nan)
                q1 = np.where(display_mask, q1, np.nan)
                q3 = np.where(display_mask, q3, np.nan)
                ax.fill_between(m, q1, q3, color=color, alpha=0.20)
                ax.plot(
                    m, iqm_values, color=color, linewidth=1.8,
                    label=f"RL {exogenous_state_label(model, asset_index)} {slice_labels[i]}",
                )

        if not policies:
            ax.text(
                0.5, 0.5, "no policies", ha="center", va="center",
                transform=ax.transAxes, color="0.5",
            )

        ax.set_title(algorithm_label(algo, variant))
        ax.set_xlabel(r"Cash-on-hand $m$")
        ax.set_ylabel(fr"Portfolio share $w_{{{asset_index}}}$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    for ax in axes[len(algorithms):]:
        ax.axis("off")

    # fig.suptitle(
    #     fr"Portfolio share of {_risky_asset_label(asset_index)}: "
    #     fr"EGM vs RL (IQM and IQR)"
    # )
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
    support_quantiles=(0.01, 99.99),
    variant="mf",
):
    """Plot positive CEV relative to EGM benchmark over a wide visited-state support."""
    _plot_cev_distribution(
        model, grid_axes, V_egm, {variant: rl_value_grids_by_algo}, save_path,
        algorithms=algorithms, visited_by_variant={variant: visited_m or {}},
        support_quantiles=support_quantiles, variants=(variant,),
    )


def plot_cev_distribution_combined(
    model, grid_axes, V_egm, rl_value_grids_by_variant, save_path, *,
    algorithms=DEFAULT_ALGORITHMS, visited_by_variant=None,
    support_quantiles=(0.01, 99.99),
):
    """Stack MF and PD CEV panels with a common visited-state support and legend."""
    _plot_cev_distribution(
        model, grid_axes, V_egm, rl_value_grids_by_variant, save_path,
        algorithms=algorithms, visited_by_variant=visited_by_variant or {},
        support_quantiles=support_quantiles, variants=VARIANTS,
    )


def _plot_cev_distribution(
    model, grid_axes, V_egm, rl_value_grids_by_variant, save_path, *,
    algorithms, visited_by_variant, support_quantiles, variants,
):
    m_full = np.asarray(grid_axes[0], dtype=np.float64)

    # Data-adaptive ergodic-support trim
    pools = [_pooled_visited(visited_by_variant.get(variant, {}), algorithms) for variant in variants]
    pools = [pool for pool in pools if pool is not None and pool.size]
    pooled = np.concatenate(pools) if pools else None
    mask = np.ones(m_full.size, dtype=bool)
    if pooled is not None and pooled.size:
        lo, hi = np.nanpercentile(pooled, support_quantiles)
        mask = (m_full >= lo) & (m_full <= hi)

    # Safety fallback if we trimmed too aggressively
    if mask.sum() < 5:
        mask = np.ones(m_full.size, dtype=bool)

    m_grid = m_full[mask]

    exog_indices, slice_labels = value_slice_indices(model)

    n_algos = len(algorithms)
    fig, axes = plt.subplots(
        len(variants),
        n_algos,
        figsize=(4.5 * n_algos, 4.5 * len(variants)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    legend_handles = {}
    for ax, (variant, algo) in zip(axes, [(variant, algo) for variant in variants for algo in algorithms]):
        value_grids = rl_value_grids_by_variant.get(variant, {}).get(algo, [])
        visited_m = visited_by_variant.get(variant, {})

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
                line_stack = line_stack[:, mask] * 100.0    # convert to percentage points
                line_stack = np.where(line_stack > 0.0, line_stack, np.nan)
                iqm_values, q1, q3 = iqm_and_iqr(line_stack)

                color = (
                    ALGO_COLORS.get(algo, "tab:blue")
                    if model.exogenous_dim == 0
                    else SLICE_COLORS[i]
                )
                label = None if model.exogenous_dim == 0 else slice_labels[i]

                ax.fill_between(m_grid, q1, q3, color=color, alpha=0.22, linewidth=0.0)
                line, = ax.plot(m_grid, iqm_values, color=color, linewidth=1.7, label=label)

                if label is not None:
                    legend_handles.setdefault(label, line)

        # State distribution (drawn behind the CEV curves).
        if visited_m is not None and algo in visited_m:
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

        ax.set_title(algorithm_label(algo, variant), fontsize=18)
        ax.set_xlabel("Cash-on-hand $m$", fontsize=17)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(alpha=0.3, which="both")

    for row in range(len(variants)):
        axes[row * n_algos].set_ylabel(r"$\lambda\ \mathrm{in}\ \%$", fontsize=17)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.0)

    for ax in axes:
        ax.set_xlim(m_grid[0], m_grid[-1])

    #fig.suptitle("Consumption-equivalent gain of EGM over RL", fontsize="x-large")
    if len(legend_handles) > 1:
        fig.legend(list(legend_handles.values()), list(legend_handles),
                   loc="lower center", ncol=len(legend_handles), fontsize=12)
        plt.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        plt.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =============================================================================
# Shared curve / histogram statistics (per-variant and combined figures)
# =============================================================================

def _wallclock_grid(wallclock_seconds, grid_step):
    """Common evaluation-time grid for convergence curves."""
    return np.arange(0.0, float(wallclock_seconds) + 1e-9, float(grid_step))


def _regret_curve_stats(histories, egm_returns, t_grid):
    """IQM and IQR band (q1, q3) of the EGM-normalized regret over `t_grid` across seeds."""
    rl_returns = np.stack(
        [last_value_carried_forward(hist["t"], hist["u"], t_grid) for hist in histories],
        axis=0,
    )

    reg_norm = normalized_regret(egm_returns, rl_returns)
    return iqm_and_iqr(reg_norm)


def _set_regret_ylim(ax, iqr_values, linthresh: float, padding: float = 0.08) -> None:
    """Fit a symmetric-log axis to the finite IQR envelope with visual padding."""
    arrays = [np.ravel(np.asarray(values, dtype=np.float64)) for values in iqr_values]
    if not arrays:
        return

    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    threshold = max(float(linthresh), np.finfo(np.float64).tiny)
    transform = ax.yaxis.get_transform()
    transformed = np.asarray(transform.transform(values), dtype=np.float64)
    lower, upper = float(np.min(transformed)), float(np.max(transformed))
    pad = max(float(padding) * (upper - lower), 0.1 * threshold)

    inverse = transform.inverted()
    ymin, ymax = inverse.transform([lower - pad, upper + pad])
    ax.set_ylim(float(ymin), float(ymax))

def _deterministic_euler_curve(model, arrays):
    """Grid, IQM and IQR band (q1, q3) of Euler errors over the deterministic test grid."""
    min_len = min(len(a) for a in arrays)
    stack = np.stack([np.asarray(a[:min_len], dtype=np.float64) for a in arrays], axis=0)
    iqm_values, q1, q3 = iqm_and_iqr(stack)

    x = model.log_grid(model.m_grid[0], model.m_grid[-1], min_len)
    return x, iqm_values, q1, q3


def _euler_histogram_bins(euler_dicts, algorithms, n_bins=80, quantiles=(0.05, 99.95)):
    """Shared histogram bins from the pooled finite Euler errors of all dicts and algorithms."""
    all_values = []
    for euler_by_algo in euler_dicts:
        for algo in algorithms:
            for arr in euler_by_algo.get(algo, []):
                x = np.asarray(arr, dtype=np.float64).ravel()
                x = x[np.isfinite(x)]
                if x.size:
                    all_values.append(x)

    if all_values:
        pooled_all = np.concatenate(all_values)
        lo, hi = np.nanpercentile(pooled_all, list(quantiles))
        return np.linspace(lo, hi, n_bins)
    return n_bins


def _relative_frequency_hist(arrays):
    """Pooled finite values and relative-frequency weights for one histogram layer."""
    pooled = np.concatenate([np.asarray(a).ravel() for a in arrays])
    pooled = pooled[np.isfinite(pooled)]

    weights = np.ones_like(pooled) / max(pooled.size, 1)
    return pooled, weights


# =============================================================================
# Euler-error distribution
# =============================================================================

def plot_euler_error_distribution(
    model,
    euler_by_algo,
    save_path,
    deterministic=False,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    variant="mf",
):

    if deterministic:
        fig, ax = plt.subplots(figsize=(9, 5))

        for algo in algorithms:
            arrays = euler_by_algo.get(algo, [])
            if not arrays:
                continue

            x, iqm_values, q1, q3 = _deterministic_euler_curve(model, arrays)
            color = ALGO_COLORS.get(algo, "tab:blue")

            ax.fill_between(x, q1, q3, color=color, alpha=0.22)
            ax.plot(x, iqm_values, color=color, linewidth=1.3, label=algorithm_label(algo, variant))

        ax.set_xlabel("Cash-on-hand $m$", fontsize=14)
        ax.set_ylabel(r"$\log_{10}|1 - c^{\mathrm{implied}}/c|$", fontsize=14)
        #ax.set_title("Euler errors across deterministic test grid")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    else:
        rows = len(algorithms)
        fig, axes = plt.subplots(rows, 1, figsize=(9, 2.5 * rows), sharex=True)
        axes = np.atleast_1d(axes)

        bins = _euler_histogram_bins([euler_by_algo], algorithms)

        for ax, algo in zip(axes, algorithms):
            arrays = euler_by_algo.get(algo, [])

            if not arrays:
                ax.text(0.5, 0.5, "no Euler errors", ha="center", va="center", transform=ax.transAxes, color="0.5")
            else:
                pooled, weights = _relative_frequency_hist(arrays)
                ax.hist(
                    pooled,
                    bins=bins,
                    weights=weights,
                    color=ALGO_COLORS.get(algo, "tab:blue"),
                    edgecolor="black",
                    linewidth=0.25,
                )

                ax.set_ylabel(f"{algorithm_label(algo, variant)}\n$n={pooled.size}$", fontsize=14)

            ax.grid(alpha=0.3)

        axes[-1].set_xlabel(r"$\log_{10}|1 - c^{\mathrm{implied}}/c|$", fontsize=14)
        fig.text(0.02, 0.5, "Relative frequency", rotation="vertical", va="center", fontsize=14)
        #fig.suptitle("Euler-error distributions across policies and test seeds")

    plt.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def _combined_layer_color(variant, algo):
    return MF_COLOR if variant == "mf" else ALGO_COLORS.get(algo, "tab:blue")


def plot_euler_error_distribution_combined(
    model,
    euler_by_variant,
    save_path,
    deterministic=False,
    *,
    algorithms=DEFAULT_ALGORITHMS,
):
    """Overlay the model-free and post-decision Euler-error distributions.

    `euler_by_variant` maps "mf" / "pd" to `{algo: [arrays]}`; a variant may be
    absent or empty. Model-free is drawn first in grey, post-decision on top in
    the algorithm colour. Deterministic models get one panel per algorithm
    (IQM line + IQR band over cash-on-hand); stochastic models one histogram
    row per algorithm with bins shared across variants and algorithms.
    """
    # title = (
    #     f"Euler-error distributions: {VARIANT_LABELS['pd']} (colour) "
    #     f"vs {VARIANT_LABELS['mf']} (grey)"
    # )

    layout_rect = None
    if deterministic:
        rows, cols = subplot_grid(len(algorithms))
        fig, axes = plt.subplots(
            rows, cols, figsize=(4.5 * cols, 3.5 * rows), sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes).ravel()

        for ax, algo in zip(axes, algorithms):
            drawn = False
            for variant in VARIANTS:  # ("mf", "pd"): grey first, colour on top
                arrays = euler_by_variant.get(variant, {}).get(algo, [])
                if not arrays:
                    continue

                x, iqm_values, q1, q3 = _deterministic_euler_curve(model, arrays)
                color = _combined_layer_color(variant, algo)

                ax.fill_between(x, q1, q3, color=color, alpha=0.22)
                ax.plot(x, iqm_values, color=color, linewidth=1.3)
                drawn = True

            if not drawn:
                ax.text(0.5, 0.5, "no Euler errors", ha="center", va="center", transform=ax.transAxes, color="0.5")

            ax.set_title(algo.upper())
            ax.grid(alpha=0.3, which="both")

        for ax in axes[len(algorithms):]:
            ax.axis("off")

        fig.supxlabel("Cash-on-hand $m$", fontsize=14)
        fig.supylabel(r"$\log_{10}|1 - c^{\mathrm{implied}}/c|$", fontsize=14)
        #fig.suptitle(title)

    else:
        rows = len(algorithms)
        fig, axes = plt.subplots(rows, 1, figsize=(9, 2.5 * rows), sharex=True)
        axes = np.atleast_1d(axes)

        bins = _euler_histogram_bins(
            [euler_by_variant.get(variant, {}) for variant in VARIANTS], algorithms
        )

        for ax, algo in zip(axes, algorithms):
            drawn = False
            for zorder, variant in enumerate(VARIANTS, start=1):  # grey first, colour on top
                arrays = euler_by_variant.get(variant, {}).get(algo, [])
                if not arrays:
                    continue

                pooled, weights = _relative_frequency_hist(arrays)
                ax.hist(
                    pooled,
                    bins=bins,
                    weights=weights,
                    histtype="stepfilled",
                    color=_combined_layer_color(variant, algo),
                    alpha=0.7,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=zorder,
                )
                drawn = True

            if not drawn:
                ax.text(0.5, 0.5, "no Euler errors", ha="center", va="center", transform=ax.transAxes, color="0.5")

            ax.set_title(algo.upper(), loc="left", fontsize=14)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel(r"$\log_{10}|1 - c^{\mathrm{implied}}/c|$", fontsize=14)
        fig.text(0.035, 0.5, "Relative frequency", rotation="vertical", va="center", fontsize=14)
        layout_rect = (0.10, 0.0, 1.0, 1.0)
        #fig.suptitle(title)

    if layout_rect is None:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=layout_rect)
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
    linthresh=1e-4,
    variant="mf",
):
    """Plot convergence on a symmetric-log EGM-normalized-regret axis.

    If `efficiency` is passed, each algorithm's AULC is added to the legend.
    """
    t_grid = _wallclock_grid(wallclock_seconds, grid_step)

    fig, ax = plt.subplots(figsize=(9, 5))
    iqr_values = []

    for algo in algorithms:
        histories = histories_by_algo.get(algo, [])
        egm_returns = egm_returns_by_algo.get(algo, [])

        if not histories or len(histories) != len(egm_returns):
            continue

        iqm_values, q1, q3 = _regret_curve_stats(histories, egm_returns, t_grid)
        iqr_values.extend((q1, q3))

        color = ALGO_COLORS.get(algo, "tab:blue")

        label = algorithm_label(algo, variant)
        if efficiency is not None:
            aulc = efficiency.get(algo, {}).get("aulc", {})
            if aulc.get("n", 0):
                label = f"{algorithm_label(algo, variant)} (AULC={aulc['iqm']:.4f})"

        ax.fill_between(t_grid, q1, q3, color=color, alpha=0.10)
        ax.plot(t_grid, iqm_values, color=color, linewidth=1.6, label=label)

    ax.set_xlabel("Wall-clock seconds")
    ax.set_ylabel(r"Normalized regret", fontsize="large")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_yscale("symlog", linthresh=linthresh)
    _set_regret_ylim(ax, iqr_values, linthresh)
    #ax.set_title("Convergence of RL policies relative to EGM (symmetric log scale)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_training_convergence_combined(
    histories_by_variant,
    egm_returns_by_variant,
    wallclock_seconds,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    grid_step=5.0,
    linthresh=1e-4,
):
    """Overlay model-free (dashed) and post-decision (solid) convergence curves.

    Same statistics as `plot_training_convergence` (IQM line and IQR band of the
    EGM-normalized regret) for every algorithm and variant in one axis. Both
    inputs map "mf" / "pd" to `{algo: [...]}`; a variant may be absent or empty.
    """
    t_grid = _wallclock_grid(wallclock_seconds, grid_step)

    fig, ax = plt.subplots(figsize=(9, 5))
    iqr_values = []

    drawn_algos, drawn_variants = [], []

    for algo in algorithms:
        color = ALGO_COLORS.get(algo, "tab:blue")

        for variant in VARIANTS:  # ("mf", "pd"): dashed first, solid on top
            histories = histories_by_variant.get(variant, {}).get(algo, [])
            egm_returns = egm_returns_by_variant.get(variant, {}).get(algo, [])

            if not histories or len(histories) != len(egm_returns):
                continue

            iqm_values, q1, q3 = _regret_curve_stats(histories, egm_returns, t_grid)
            iqr_values.extend((q1, q3))

            ax.fill_between(t_grid, q1, q3, color=color, alpha=0.10)
            ax.plot(t_grid, iqm_values, color=color, linewidth=1.6, linestyle=VARIANT_LINESTYLES[variant])

            if algo not in drawn_algos:
                drawn_algos.append(algo)
            if variant not in drawn_variants:
                drawn_variants.append(variant)

    # Split legend: colours identify algorithms, line styles identify variants.
    handles = [Line2D([0], [0], color=ALGO_COLORS.get(a, "tab:blue"), linewidth=1.6) for a in drawn_algos]
    labels = [a.upper() for a in drawn_algos]
    for variant in ("pd", "mf"):
        if variant in drawn_variants:
            handles.append(Line2D([0], [0], color="black", linewidth=1.6, linestyle=VARIANT_LINESTYLES[variant]))
            labels.append(VARIANT_LABELS[variant])
    if handles:
        ax.legend(handles, labels, fontsize=9)

    ax.set_xlabel("Wall-clock seconds")
    ax.set_ylabel(r"Normalized regret", fontsize="large")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_yscale("symlog", linthresh=linthresh)
    _set_regret_ylim(ax, iqr_values, linthresh)
    # ax.set_title(
    #     f"Convergence of RL policies relative to EGM (symmetric log scale): {VARIANT_LABELS['pd']} (solid) "
    #     f"vs {VARIANT_LABELS['mf']} (dashed)"
    # )

    plt.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
