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


DEFAULT_ALGORITHMS = tuple(ALGO_REGISTRY.keys())

ALGO_COLORS = {
    "ppo": "tab:red",
    "ddpg": "tab:blue",
    "td3": "tab:orange",
    "sac": "tab:green",
}

MODEL_LABELS = {
    "CassKoopmans": "Cass--Koopmans",
    "BrockMirman": "Brock--Mirman",
    "BufferStock": "Buffer Stock",
    "PortfolioChoice": "Portfolio Choice",
}

EGM_COLOR = "0.45"


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

    slice_colors = ["tab:blue", "tab:purple", "tab:brown"]

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
            color = EGM_COLOR if model.exogenous_dim == 0 else slice_colors[i]
            ax.plot(m, egm_share, color=color, linestyle="--", linewidth=1.8, alpha=0.8, label=label)

        stacks = rl_stacks.get(algo)

        if stacks is None:
            ax.text(0.5, 0.5, "no policies", ha="center", va="center", transform=ax.transAxes, color="0.5")
        else:
            for i, stack in enumerate(stacks):
                med, q1, q3 = median_iqr(stack, axis=0)
                color = ALGO_COLORS.get(algo, "tab:blue") if model.exogenous_dim == 0 else slice_colors[i]

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
# CEV distribution
# =============================================================================

def plot_cev_distribution(
    model,
    grid_axes,
    V_egm,
    rl_value_grids_by_algo,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    visited_m=None,
):
    """Plot CEV relative to the EGM benchmark.

    For models with exogenous states, the figure varies the first exogenous
    state over low/middle/high and holds remaining exogenous states fixed.
    """
    m = np.asarray(grid_axes[0])
    exog_indices, slice_labels = value_slice_indices(model)
    slice_colors = ["tab:blue", "tab:purple", "tab:brown"]

    rows, cols = subplot_grid(len(algorithms))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.6 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    deterministic = is_deterministic_one_state(model)

    for ax, algo in zip(axes, algorithms):
        Vs = rl_value_grids_by_algo.get(algo, [])

        if not Vs:
            ax.text(0.5, 0.5, "no value grids", ha="center", va="center", transform=ax.transAxes, color="0.5")
        else:
            lam_stack = np.stack([cev_map(model, V, V_egm) for V in Vs], axis=0)
            lam_stack = np.where(np.isfinite(lam_stack), lam_stack, np.nan)

            for i, exog_idx in enumerate(exog_indices):
                line_stack = np.stack(
                    [select_value_line(lam, exog_idx) for lam in lam_stack],
                    axis=0,
                ) * 100.0

                med, q1, q3 = median_iqr(line_stack, axis=0)

                color = ALGO_COLORS.get(algo, "tab:blue") if model.exogenous_dim == 0 else slice_colors[i]
                label = None if model.exogenous_dim == 0 else slice_labels[i]

                ax.fill_between(m, q1, q3, color=color, alpha=0.22)
                ax.plot(m, med, color=color, linewidth=1.7, label=label)

            if model.exogenous_dim > 0:
                ax.legend(fontsize=8)

        if not deterministic and visited_m is not None and algo in visited_m:
            x = np.asarray(visited_m[algo], dtype=np.float64)
            x = x[np.isfinite(x)]

            if x.size:
                ax2 = ax.twinx()
                ax2.hist(x, bins=60, density=True, color="0.6", alpha=0.35, edgecolor="none")
                ax2.set_yticks([])
                ax2.set_zorder(ax.get_zorder() - 1)
                ax.patch.set_visible(False)

        #ax.axhline(0.0, color="0.4", linestyle=":", linewidth=0.8)
        ax.set_title(algo.upper())
        ax.set_xlabel("Cash-on-hand $m$")
        ax.set_ylabel("CEV relative to EGM [%]")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

    for ax in axes[len(algorithms):]:
        ax.axis("off")

    fig.suptitle("Consumption-equivalent variation relative to EGM")
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

def last_value_carried_forward(t, y, t_grid):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if t.size == 0:
        return np.full_like(t_grid, np.nan, dtype=np.float64)

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    idx = np.searchsorted(t, t_grid, side="right") - 1
    return np.where(idx >= 0, y[np.clip(idx, 0, len(y) - 1)], np.nan)


def plot_training_convergence(
    histories_by_algo,
    egm_returns_by_algo,
    wallclock_seconds,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
    grid_step=5.0,
):
    """Plot convergence as log10 EGM regret over training time."""
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

        egm = np.asarray(egm_returns, dtype=np.float64)[:, None]
        regret = egm - rl_returns

        with np.errstate(invalid="ignore", divide="ignore"):
            log_regret = np.where(regret > 0.0, np.log10(regret), np.nan)

        med, q1, q3 = median_iqr(log_regret, axis=0)

        color = ALGO_COLORS.get(algo, "tab:blue")
        ax.fill_between(t_grid, q1, q3, color=color, alpha=0.22)
        ax.plot(t_grid, med, color=color, linewidth=1.6, label=algo.upper())

    ax.set_xlabel("Training-only wall-clock seconds")
    ax.set_ylabel(r"$\log_{10}(U_{\mathrm{EGM}} - U_{\mathrm{RL}})$")
    ax.set_title("Convergence of RL policies relative to EGM")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    plt.tight_layout()
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)