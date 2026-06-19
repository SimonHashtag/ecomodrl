# -----------------------------------------------------------------------------
# Plotting infrastructure:
#   1. plot_policy              -- 2x2 RL panels vs EGM benchmark (gray)
#   2. plot_cev                 -- 1x4 lambda(s) on the grid
#   3. plot_euler_errors_hist   -- 4x1 stacked histograms (CK: 1x1 line plot)
#   4. plot_convergence         -- 1x1 mean log10|EE| vs wall-clock seconds
#
# Common y-axis applies WITHIN one figure only (one model). Across models, no.
# -----------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt


ALGO_ORDER = ["ddpg", "td3", "sac", "ppo"]
ALGO_LABELS = {"ddpg": "DDPG", "td3": "TD3", "sac": "SAC", "ppo": "PPO"}
MODEL_LABELS = {
    "cass_koopmans": "Cass--Koopmans",
    "brock_mirman":  "Brock--Mirman",
    "buffer_stock":  "Buffer Stock",
}
ALGO_COLORS = {
    "ddpg": "tab:blue",
    "td3":  "tab:orange",
    "sac":  "tab:green",
    "ppo":  "tab:red",
}
EGM_COLOR = "0.55"  # gray


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Policy plot
# ---------------------------------------------------------------------------

def plot_policy(model, policies, save_path):
    """policies: dict like {'egm': EGMPolicy, 'ppo': RLPolicy or None, ...}.
    Always renders a 2x2 grid (DDPG, TD3, SAC, PPO) with EGM in gray
    background. Missing algos -> empty axes. For BM: three z-slices."""
    egm = policies["egm"]

    # Compute reference shares (skip m=0 to avoid 0/0 in c/m)
    if model.HAS_EXO:
        z_grid = model.xi_grid
        z_slices = [z_grid[0], z_grid[len(z_grid) // 2], z_grid[-1]]
        z_labels = ["z_min", "z_median", "z_max"]
        slice_colors = ["tab:blue", "tab:purple", "tab:red"]
        m_grid = model.m_grid[model.m_grid > 0]

        egm_shares = []
        for z in z_slices:
            state = np.stack([m_grid, np.full_like(m_grid, z)], axis=-1)
            egm_shares.append(egm.share(state))
    else:
        m_grid = model.m_grid[model.m_grid > 0]
        state = m_grid[..., None]
        egm_share = egm.share(state)

    # First pass: collect RL shares to determine common y-axis upper bound
    rl_shares = {}
    for algo in ALGO_ORDER:
        pol = policies.get(algo)
        if pol is None:
            continue
        if model.HAS_EXO:
            shares = []
            for z in z_slices:
                state = np.stack([m_grid, np.full_like(m_grid, z)], axis=-1)
                shares.append(pol.share(state))
            rl_shares[algo] = shares
        else:
            rl_shares[algo] = pol.share(m_grid[..., None])

    # Common y-axis upper bound (highest RL share, or EGM if no RL yet)
    candidates = []
    for v in rl_shares.values():
        candidates.append(np.nanmax(np.asarray(v)))
    if not candidates:
        ref = np.asarray(egm_shares) if model.HAS_EXO else egm_share
        candidates.append(np.nanmax(ref))
    y_max = float(max(candidates)) * 1.05
    y_min = 0.0

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, algo in zip(axes, ALGO_ORDER):
        if model.HAS_EXO:
            for i, (z, lab, col) in enumerate(zip(z_slices, z_labels, slice_colors)):
                ax.plot(m_grid, egm_shares[i], color=col, alpha=0.35, linewidth=2)
            if algo in rl_shares:
                for i, (lab, col) in enumerate(zip(z_labels, slice_colors)):
                    ax.plot(m_grid, rl_shares[algo][i], color=col,
                            linewidth=1.5, label=lab)
                ax.legend(fontsize=8, loc="best")
        else:
            ax.plot(m_grid, egm_share, color=EGM_COLOR, linewidth=2, label="EGM")
            if algo in rl_shares:
                ax.plot(m_grid, rl_shares[algo], color="tab:blue",
                        linewidth=1.5, label=ALGO_LABELS[algo])
            ax.legend(fontsize=8, loc="best")
        ax.set_title(ALGO_LABELS[algo])
        ax.set_xlabel("Cash-on-hand $m$")
        ax.set_ylabel("Consumption share $c/m$")
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.3)

    fig.suptitle("Consumption share policy: RL vs EGM benchmark")
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. CEV plot
# ---------------------------------------------------------------------------

def plot_cev(model, grid, cev_dict, save_path):
    """grid: tuple (m_grid,) for 1-D models, (m_grid, z_grid) for BM.
    cev_dict: {algo: lambda_array_on_grid or None}."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    if model.HAS_EXO:
        m_grid, z_grid = grid
        z_idx = [0, len(z_grid) // 2, len(z_grid) - 1]
        z_labels = ["z_min", "z_median", "z_max"]
        slice_colors = ["tab:blue", "tab:purple", "tab:red"]
    else:
        (m_grid,) = grid

    for ax, algo in zip(axes, ALGO_ORDER):
        lam = cev_dict.get(algo)
        if lam is None:
            ax.text(0.5, 0.5, "(no policy)", ha="center", va="center",
                    transform=ax.transAxes, color="0.6")
        else:
            if model.HAS_EXO:
                for i, lab, col in zip(z_idx, z_labels, slice_colors):
                    ax.plot(m_grid, lam[i] * 100, color=col, label=lab, linewidth=1.5)
                ax.legend(fontsize=8, loc="best")
            else:
                ax.plot(m_grid, lam * 100, color="tab:blue", linewidth=1.5)
            ax.axhline(0, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_title(ALGO_LABELS[algo])
        ax.set_xlabel("Cash-on-hand $m$")
        ax.set_ylabel(r"CEV $\lambda(m)$ [%]")
        ax.grid(alpha=0.3)

    fig.suptitle("Consumption-equivalent variation vs EGM benchmark")
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Euler errors histogram (RL only -- excludes EGM)
# ---------------------------------------------------------------------------

def plot_euler_errors_hist(model, ee_dict, save_path):
    """ee_dict: {algo: log10|EE| array or None}.
    CK: deterministic -> 1x1 with all RL algos overlaid as lines on test grid.
    Else: 4x1 stacked histograms."""
    deterministic = (model.__class__.__name__ == "CassKoopmans")
    if deterministic:
        fig, ax = plt.subplots(figsize=(9, 5))
        for algo in ALGO_ORDER:
            ee = ee_dict.get(algo)
            if ee is None:
                continue
            ee = np.asarray(ee).ravel()
            ax.plot(np.arange(ee.size), ee, color=ALGO_COLORS[algo],
                    label=ALGO_LABELS[algo], linewidth=1.0, alpha=0.8)
        ax.set_xlabel("Test point index (along m-grid)")
        ax.set_ylabel(r"$\log_{10}|1 - c^{\mathrm{implied}}/c|$")
        ax.set_title("Euler errors across test grid (Cass-Koopmans, deterministic)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        for ax, algo in zip(axes, ALGO_ORDER):
            ee = ee_dict.get(algo)
            if ee is None:
                ax.text(0.5, 0.5, "(no policy)", ha="center", va="center",
                        transform=ax.transAxes, color="0.6")
            else:
                ax.hist(ee, bins=80, color=ALGO_COLORS[algo],
                        edgecolor="black", linewidth=0.3)
            ax.set_ylabel(ALGO_LABELS[algo])
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel(r"$\log_{10}|1 - c^{\mathrm{implied}}/c|$")
        fig.suptitle("Distribution of Euler errors over simulated rollouts")
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Convergence plot (RL only)
# ---------------------------------------------------------------------------

def plot_convergence(model, history_dict, save_path):
    """history_dict: {algo: (wall_clock_seconds, V_grid) or None}.

    `V_grid` is the per-checkpoint mean discounted return across a fixed
    grid of starting states (the selection metric used during training).
    It is negative under CRRA with gamma > 1 and rises during training.
    Plotted raw on a linear y-axis. Non-finite entries are stored as NaN
    in history.npz and render as gaps in the line.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGO_ORDER:
        hist = history_dict.get(algo)
        if hist is None:
            continue
        t, v_grid = hist
        v_grid = np.asarray(v_grid, dtype=np.float64)
        with np.errstate(invalid='ignore'):
            v_plot = np.where(np.isfinite(v_grid), v_grid, np.nan)
        ax.plot(t, v_plot, color=ALGO_COLORS[algo], label=ALGO_LABELS[algo],
                linewidth=1.5)
    ax.set_xlabel("Training-only wall-clock seconds")
    ax.set_ylabel(r"Mean grid value $V_{\mathrm{grid}}$")
    ax.set_title("Training-time mean grid value")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ===========================================================================
# Distribution-aware plots (multi-seed protocol)
# ===========================================================================
#
# These functions consume lists of per-train-seed quantities (and, for Euler
# errors and CEV regret, lists of per-(train_seed, test_seed) cells) and
# render median +/- IQR distributions instead of point estimates.
# ---------------------------------------------------------------------------


def _median_iqr(stack, axis=0):
    """Return (median, q1, q3) along `axis`, ignoring NaNs."""
    med = np.nanmedian(stack, axis=axis)
    q1  = np.nanpercentile(stack, 25, axis=axis)
    q3  = np.nanpercentile(stack, 75, axis=axis)
    return med, q1, q3


# ---------------------------------------------------------------------------
# 1d. Policy plot -- per-seed bundle of curves + median, vs EGM
# ---------------------------------------------------------------------------

def plot_policy_dist(model, egm_policy, rl_policies, save_path):
    """rl_policies: {algo: [RLPolicy_seed1, ..., RLPolicy_seedN]}.
    Renders the 2x2 algo grid; per panel, an IQR (25-75) band over
    per-seed share curves, a solid median, and EGM dashed in gray.
    BM: per-z-slice with three colors.
    """
    m_grid = model.m_grid[model.m_grid > 0]

    if model.HAS_EXO:
        z_grid = model.xi_grid
        z_slices = [z_grid[0], z_grid[len(z_grid) // 2], z_grid[-1]]
        z_labels = ["z_min", "z_median", "z_max"]
        slice_colors = ["tab:blue", "tab:purple", "tab:red"]
        egm_shares = []
        for z in z_slices:
            state = np.stack([m_grid, np.full_like(m_grid, z)], axis=-1)
            egm_shares.append(egm_policy.share(state))
    else:
        state = m_grid[..., None]
        egm_share = egm_policy.share(state)

    rl_share_stacks = {}
    y_min_cand = []
    y_max_cand = []
    for algo in ALGO_ORDER:
        pols = rl_policies.get(algo, [])
        if not pols:
            continue
        if model.HAS_EXO:
            slices = []
            for z in z_slices:
                state = np.stack([m_grid, np.full_like(m_grid, z)], axis=-1)
                slices.append(np.stack([p.share(state) for p in pols], axis=0))
            rl_share_stacks[algo] = slices
            for s in slices:
                q1, q3 = np.nanpercentile(s, [25, 75], axis=0)
                y_min_cand.append(np.nanmin(q1))
                y_max_cand.append(np.nanmax(q3))
        else:
            stack = np.stack([p.share(state) for p in pols], axis=0)
            rl_share_stacks[algo] = stack
            q1, q3 = np.nanpercentile(stack, [25, 75], axis=0)
            y_min_cand.append(np.nanmin(q1))
            y_max_cand.append(np.nanmax(q3))

    ref = np.asarray(egm_shares) if model.HAS_EXO else egm_share
    y_min_cand.append(np.nanmin(ref))
    y_max_cand.append(np.nanmax(ref))
    lo = float(min(y_min_cand))
    hi = float(max(y_max_cand))
    pad = 0.05 * (hi - lo) if hi > lo else 0.05 * abs(hi)
    y_min = lo - pad
    y_max = hi + pad

    def _bands(ax, x, stack, color):
        q1, med, q3 = np.nanpercentile(stack, [25, 50, 75], axis=0)
        ax.fill_between(x, q1, q3, color=color, alpha=0.35, linewidth=0)
        ax.plot(x, med, color=color, linewidth=1.8)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, algo in zip(axes, ALGO_ORDER):
        if model.HAS_EXO:
            for i, (lab, col) in enumerate(zip(z_labels, slice_colors)):
                ax.plot(m_grid, egm_shares[i], color=col, alpha=0.55,
                        linewidth=1.8, linestyle="--", label=f"EGM {lab}")
            stacks = rl_share_stacks.get(algo)
            if stacks is not None:
                for i, (lab, col) in enumerate(zip(z_labels, slice_colors)):
                    _bands(ax, m_grid, stacks[i], col)
            ax.legend(fontsize=7, loc="best")
        else:
            ax.plot(m_grid, egm_share, color=EGM_COLOR, linewidth=1.8,
                    linestyle="--", label="EGM")
            stack = rl_share_stacks.get(algo)
            if stack is not None:
                _bands(ax, m_grid, stack, ALGO_COLORS[algo])
            ax.legend(fontsize=8, loc="best")
        ax.set_title(ALGO_LABELS[algo])
        ax.set_xlabel("Cash-on-hand $m$")
        ax.set_ylabel("Consumption share $c/m$")
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.3)
    fig.suptitle("Consumption share policy: median + IQR band vs EGM")
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2d. CEV map -- median + IQR band across train seeds
# ---------------------------------------------------------------------------

def _cev_map(model, V_rl, V_egm):
    gamma = model.GAMMA
    with np.errstate(invalid='ignore', divide='ignore'):
        if gamma == 1.0:
            return np.exp((V_egm - V_rl) * (1.0 - model.BETA)) - 1.0
        return (V_egm / V_rl) ** (1.0 / (1.0 - gamma)) - 1.0


def plot_cev_dist(model, grid_disp, V_egm_grid, rl_V_grids, save_path,
                  visited_pools=None):
    """rl_V_grids: {algo: [V_rl_arr_seed1, ..., V_rl_arr_seedN]}.
    For BM, V is shape (n_z, n_m); for 1-D models, shape (n_m,).
    Plots 1x4 panels with median lambda(m) + shaded IQR band across seeds
    on a log y-axis (negative CEV entries are masked to NaN).

    visited_pools: optional {algo: 1-D array of visited m values pooled
    across (train_seed, test_seed) cells} -- if provided, a grey density
    histogram of visited m is overlaid on a twin y-axis (skipped for the
    deterministic CK model).
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    if model.HAS_EXO:
        m_grid, z_grid = grid_disp
        z_idx = [0, len(z_grid) // 2, len(z_grid) - 1]
        z_labels = ["z_min", "z_median", "z_max"]
        slice_colors = ["tab:blue", "tab:purple", "tab:red"]
    else:
        (m_grid,) = grid_disp
    deterministic = (getattr(model, "weights", None) is None) and not model.HAS_EXO

    for ax, algo in zip(axes, ALGO_ORDER):
        Vs = rl_V_grids.get(algo, [])
        if not Vs:
            ax.text(0.5, 0.5, "(no policies)", ha="center", va="center",
                    transform=ax.transAxes, color="0.6")
        else:
            lam_stack = np.stack([_cev_map(model, V, V_egm_grid) for V in Vs],
                                 axis=0)
            with np.errstate(invalid='ignore'):
                lam_stack = np.where(lam_stack > 0, lam_stack, np.nan)
            if model.HAS_EXO:
                for zi, lab, col in zip(z_idx, z_labels, slice_colors):
                    sub = lam_stack[:, zi, :] * 100.0
                    med, q1, q3 = _median_iqr(sub, axis=0)
                    ax.fill_between(m_grid, q1, q3, color=col, alpha=0.18)
                    ax.plot(m_grid, med, color=col, linewidth=1.5, label=lab)
                ax.legend(fontsize=8, loc="best")
            else:
                sub = lam_stack * 100.0
                med, q1, q3 = _median_iqr(sub, axis=0)
                ax.fill_between(m_grid, q1, q3, color=ALGO_COLORS[algo],
                                alpha=0.22)
                ax.plot(m_grid, med, color=ALGO_COLORS[algo], linewidth=1.8)

        if (not deterministic) and visited_pools is not None:
            vp = visited_pools.get(algo)
            if vp is not None and np.asarray(vp).size > 0:
                ax2 = ax.twinx()
                ax2.hist(np.asarray(vp).ravel(), bins=60, color="0.5",
                         alpha=0.5, density=True, edgecolor="none")
                ax2.set_yticks([])
                ax2.set_zorder(ax.get_zorder() - 1)
                ax.patch.set_visible(False)

        ax.set_title(ALGO_LABELS[algo])
        ax.set_xlabel("Cash-on-hand $m$")
        ax.set_ylabel(r"CEV $\lambda(m)$ [%]")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("CEV vs EGM benchmark: median +/- IQR across training seeds (log y)")
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3d. Euler errors -- pooled histograms / per-policy bands
# ---------------------------------------------------------------------------

def plot_euler_errors_hist_dist(model, ee_pools, ee_seedwise, save_path):
    """Stochastic models: ee_pools[algo] = list of log10|EE| arrays (one per
    (train_seed, test_seed) cell), pooled into a single histogram per algo
    (4x1 stack) with relative-frequency y-axis.
    CK (deterministic): ee_seedwise[algo] = list of log10|EE| arrays (one
    per train_seed). Plots median + IQR band across policies for each algo
    overlaid on a single axes against the log-spaced m-grid (x-axis).
    """
    deterministic = (model.__class__.__name__ == "CassKoopmans")
    if deterministic:
        fig, ax = plt.subplots(figsize=(9, 5))
        for algo in ALGO_ORDER:
            arrs = ee_seedwise.get(algo, [])
            if not arrs:
                continue
            min_len = min(a.size for a in arrs)
            stack = np.stack([a[:min_len] for a in arrs], axis=0)
            med, q1, q3 = _median_iqr(stack, axis=0)
            x = model._log_grid(model.m_grid[0], model.m_grid[-1], min_len)
            ax.fill_between(x, q1, q3, color=ALGO_COLORS[algo], alpha=0.22)
            ax.plot(x, med, color=ALGO_COLORS[algo], linewidth=1.2,
                    label=ALGO_LABELS[algo])
        ax.set_xscale("log")
        ax.set_xlabel("Cash-on-hand $m$")
        ax.set_ylabel(r"$\log_{10}|1 - c^{\mathrm{implied}}/c|$")
        ax.set_title("Euler errors (CK): median +/- IQR across policies")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    else:
        fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        # common bin range across algos
        all_vals = np.concatenate(
            [np.concatenate(ee_pools.get(a, []) or [np.array([])])
             for a in ALGO_ORDER]
        ) if any(ee_pools.get(a) for a in ALGO_ORDER) else np.array([])
        if all_vals.size > 0:
            lo = float(np.nanpercentile(all_vals, 0.5))
            hi = float(np.nanpercentile(all_vals, 99.5))
            bins = np.linspace(lo, hi, 80)
        else:
            bins = 80
        for ax, algo in zip(axes, ALGO_ORDER):
            arrs = ee_pools.get(algo, [])
            if not arrs:
                ax.text(0.5, 0.5, "(no policies)", ha="center", va="center",
                        transform=ax.transAxes, color="0.6")
            else:
                pooled = np.concatenate([np.asarray(a).ravel() for a in arrs])
                pooled = pooled[np.isfinite(pooled)]
                w = np.ones_like(pooled) / max(pooled.size, 1)
                ax.hist(pooled, bins=bins, weights=w,
                        color=ALGO_COLORS[algo],
                        edgecolor="black", linewidth=0.3)
                ax.set_ylabel(f"{ALGO_LABELS[algo]}\n(n={pooled.size})")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel(r"$\log_{10}|1 - c^{\mathrm{implied}}/c|$")
        fig.text(0.02, 0.5, "Relative frequency", rotation="vertical",
                 va="center", ha="center", fontsize=10)
        fig.suptitle("Euler errors: pooled across (policy x test-seed) cells per algo")
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4d. Convergence -- per-algo median + IQR band across training seeds
# ---------------------------------------------------------------------------

def _resample_lvcf(t, v, t_grid):
    """Last-value-carried-forward resample of (t, v) onto t_grid.
    Before the first sample, returns NaN; after the last sample, holds
    the last value."""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if t.size == 0:
        return np.full_like(t_grid, np.nan, dtype=np.float64)
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    # for each t_grid point, find rightmost t <= t_grid
    idx = np.searchsorted(t, t_grid, side="right") - 1
    out = np.where(idx >= 0, v[np.clip(idx, 0, len(v) - 1)], np.nan)
    return out


def plot_convergence_dist(histories, v_egms, wallclock, save_path,
                          grid_step=5.0):
    """histories: {algo: [(t_i, V_i), ...]}.
    v_egms:    {algo: [v_egm_i, ...]} aligned with histories so that the
               i-th history corresponds to the i-th V_EGM (both indexed
               by train_seed in insertion order).

    Resamples each (t, V) onto a common grid via last-value-carried-forward,
    converts to regret = V_EGM - V_RL (mask non-positive -> NaN), and
    plots the median + IQR band per algo on a log10 y-axis.
    """
    t_grid = np.arange(0.0, float(wallclock) + 1e-9, float(grid_step))
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGO_ORDER:
        runs = histories.get(algo, [])
        vegs = v_egms.get(algo, [])
        if not runs or len(runs) != len(vegs):
            continue
        resampled = np.stack(
            [_resample_lvcf(t, v, t_grid) for t, v in runs], axis=0
        )
        vegs_arr = np.asarray(vegs, dtype=np.float64)[:, None]
        regret = vegs_arr - resampled
        with np.errstate(invalid='ignore'):
            regret = np.where(regret > 0, regret, np.nan)
            log_regret = np.log10(regret)
        med, q1, q3 = _median_iqr(log_regret, axis=0)
        ax.fill_between(t_grid, q1, q3, color=ALGO_COLORS[algo], alpha=0.22)
        ax.plot(t_grid, med, color=ALGO_COLORS[algo], linewidth=1.6,
                label=ALGO_LABELS[algo])
    ax.set_xlabel("Training-only wall-clock seconds")
    ax.set_ylabel(r"$\log_{10}(V_{\mathrm{EGM}} - V_{\mathrm{RL}})$")
    ax.set_title("Convergence: median +/- IQR of log10 regret across training seeds")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX comparison tables
# ---------------------------------------------------------------------------

def _fmt_iqm_ci(point, lo, hi, fmt):
    # Stack the CI on a second line below the point estimate (needs the
    # ``makecell`` package). The CI is set in a smaller font.
    return (f"\\makecell{{${fmt % point}$ \\\\ "
            f"{{\\scriptsize $[{fmt % lo},\\ {fmt % hi}]$}}}}")


# Shared metric specification: (row label, summary key, best-direction, scale).
# All numbers are rounded to 4 digits after the decimal point. CEV is reported
# in percent (scale = 100).
_TABLE_ROWS = [
    (r"$V_{\mathrm{grid}}$", "v_rl",        "max", 1.0),
    ("CEV (\\%)",   "cev_regret",  "min", 100.0),
    ("EE mean",     "ee_mean",     "min", 1.0),
    ("Wasserstein", "wasserstein", "min", 1.0),
]
_NUM_FMT = "%.4f"


def make_latex_tables(model, summaries_by_algo, save_path):
    """Write a booktabs LaTeX fragment summarising the multi-seed test results
    for one model. Columns are algorithms (DDPG/TD3/SAC/PPO); rows are
    metrics. Each cell shows IQM with a 95% bootstrap CI as
    ``$X\\ [\\text{lo},\\ \\text{hi}]$``. The best entry per row is bolded
    (max for V_grid, min for the three regret-style metrics). All numbers are
    rounded to 4 digits after the decimal point and CEV is reported in percent.

    The output is a slide-friendly, float-free fragment (centered tabular
    wrapped in ``\\adjustbox`` so it scales to the text width, followed by a
    short caption line). It can be dropped directly inside a beamer frame.

    summaries_by_algo: {algo: dict from summary.json}.
    """
    rows = _TABLE_ROWS
    algos = ALGO_ORDER  # ddpg, td3, sac, ppo

    lines = []
    lines.append(r"\begin{minipage}{\linewidth}")
    lines.append(r"\centering")
    lines.append(r"\adjustbox{max width=\linewidth}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(algos) + r"}")
    lines.append(r"\toprule")
    header = "Metric & " + " & ".join(ALGO_LABELS[a] for a in algos) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for label, key, direction, scale in rows:
        iqms = []
        cells = []
        for a in algos:
            stats = summaries_by_algo.get(a, {}).get(key, {})
            if not stats or stats.get("n", 0) == 0:
                iqms.append(float("nan"))
                cells.append("--")
                continue
            iqm_v = stats["iqm"] * scale
            lo = stats["ci_lo"] * scale
            hi = stats["ci_hi"] * scale
            iqms.append(iqm_v)
            cells.append(_fmt_iqm_ci(iqm_v, lo, hi, _NUM_FMT))
        arr = np.asarray(iqms, dtype=np.float64)
        if np.any(np.isfinite(arr)):
            best = (np.nanargmax(arr) if direction == "max"
                    else np.nanargmin(arr))
            cells[best] = r"{\boldmath\bfseries " + cells[best] + "}"
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    model_name = getattr(model, "__class__", type(model)).__name__
    caption = (f"Multi-seed test results for {model_name}. Each cell shows "
               f"the IQM with a $95\\%$ bootstrap CI. Best per row in bold.")
    lines.append(r"\par\vspace{4pt}")
    lines.append(r"{\footnotesize " + caption + "}")
    lines.append(r"\end{minipage}")

    _ensure_dir(save_path)
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_combined_latex_table(summaries_by_model, save_path):
    """Write a single beamer-frame comparison table for all economic models at
    once, showing only the point estimates (IQM). Rows are grouped per model
    via ``\\multirow``; columns are algorithms (DDPG/TD3/SAC/PPO). The best
    point estimate per row is bolded. All numbers are rounded to 4 digits after
    the decimal point and CEV is reported in percent.

    summaries_by_model: list of ``(model_name, summaries_by_algo)`` pairs, where
    ``model_name`` is the internal key (e.g. ``"cass_koopmans"``) and
    ``summaries_by_algo`` is ``{algo: dict from summary.json}``.
    """
    algos = ALGO_ORDER  # ddpg, td3, sac, ppo
    rows = _TABLE_ROWS
    # Metric labels for the combined (point-estimate-only) table.
    metric_labels = {
        "v_rl":        r"$v_{\text{RL}}$",
        "cev_regret":  r"CEV (\%)",
        "ee_mean":     r"EE mean",
        "wasserstein": r"Wasserstein",
    }

    lines = []
    lines.append(r"\begin{frame}[t]{Results}")
    lines.append(r"  \centering\footnotesize")
    lines.append(r"  \renewcommand{\arraystretch}{1.15}")
    lines.append(r"  \begin{tabular}{ll" + "r" * len(algos) + r"}")
    lines.append(r"    \toprule")
    header = (r"    \textbf{Model} & \textbf{Metric} & "
              + " & ".join(r"\textbf{" + ALGO_LABELS[a] + "}" for a in algos)
              + r" \\")
    lines.append(header)
    lines.append(r"    \midrule")

    n_models = len(summaries_by_model)
    for mi, (model_name, summaries_by_algo) in enumerate(summaries_by_model):
        model_label = MODEL_LABELS.get(model_name, model_name)
        nrows = len(rows)
        for ri, (_, key, direction, scale) in enumerate(rows):
            iqms = []
            cells = []
            for a in algos:
                stats = summaries_by_algo.get(a, {}).get(key, {})
                if not stats or stats.get("n", 0) == 0:
                    iqms.append(float("nan"))
                    cells.append("--")
                    continue
                iqm_v = stats["iqm"] * scale
                iqms.append(iqm_v)
                cells.append("$" + (_NUM_FMT % iqm_v) + "$")
            arr = np.asarray(iqms, dtype=np.float64)
            if np.any(np.isfinite(arr)):
                best = (np.nanargmax(arr) if direction == "max"
                        else np.nanargmin(arr))
                if cells[best] != "--":
                    cells[best] = r"$\mathbf{" + (_NUM_FMT % arr[best]) + "}$"

            if ri == 0:
                first = (r"    \multirow{" + str(nrows) + r"}{*}{\textbf{"
                         + model_label + "}}")
            else:
                first = r"   "
            lines.append(first + " & " + metric_labels[key] + " & "
                         + " & ".join(cells) + r" \\")
        if mi < n_models - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{frame}")

    _ensure_dir(save_path)
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")
