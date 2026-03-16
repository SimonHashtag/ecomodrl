"""
Brock-Mirman comparison analysis: computes all metrics and generates plots.

This script:
1. Loads raw per-seed data from each algorithm's `raw_metrics.json`
2. Computes benchmark solution
3. Computes all derived metrics (CEV, Wasserstein distance, bootstrap CIs)
4. Generates comparison plots (with z-slices for 2D state space)
"""
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import wasserstein_distance

from ecomodrl.benchmarks.brockmirman.problemclass import PlanningProblem, PlanningSolvers
from ecomodrl.utils.metrics.accuracy import cev_crra
from ecomodrl.utils.metrics.performance import get_bootstrapped_iqm_ci, interquartile_mean
from ecomodrl.utils.params_loader import BrockMirmanParams
from ecomodrl.utils.latex_formatting import generate_latex_table

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ALGOS = ["PPO", "DDPG", "SAC", "TD3"]
TOML_PATH = Path("ecomodrl/experiments/model_params/brockmirman.toml")
RUNRESULTS_PATH = Path(os.environ.get("RUNRESULTS_PATH", "./runs/BrockMirman"))

# Bootstrap settings
N_BOOT = 1000
ALPHA = 0.05
BOOTSTRAP_SEED = 12345

# Evaluation episodes (must match n_eval_episodes used in BM RunConfig during training)
# TODO: Load N_EVAL_EPISODES & TRAINING_SEEDS from each algo's train_cfgs.json if available, to ensure consistency with training config
N_EVAL_EPISODES = 5

# Analysis seeds (distinct from training seeds 200-209)
TRAINING_SEEDS = list(range(200, 210))
ANALYSIS_SEED = 300  # For benchmark evaluation simulations
CEV_SAMPLING_SEED = 400  # For CEV visited-states sampling

# Plotting style
plt.style.use("seaborn-v0_8-colorblind")
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.serif": ["cmr10"],
    "axes.formatter.use_mathtext": True,
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BMComparison")


# ----------------------------------------------------------------------------
# Data Structures
# ----------------------------------------------------------------------------

@dataclass
class Benchmark:
    """Shared benchmark data computed once."""
    model: PlanningProblem
    k_grid: np.ndarray
    z_grid: np.ndarray
    policy: Callable              # RegularGridInterpolator on (k, z)
    V: Callable                   # RegularGridInterpolator on (k, z)
    euler_errors_log: np.ndarray  # Flattened log10|error|
    utility: float
    returns_eval: np.ndarray      # Return on evaluation grid (n_seeds, n_eval_episodes)


@dataclass  
class AlgoResults:
    """Processed results for one algorithm."""
    # Metadata
    seeds: list                    # List of seed numbers
    k_grid: np.ndarray
    z_grid: np.ndarray
    
    # Raw per-seed data (3D arrays: n_seeds x n_k x n_z)
    return_curves: np.ndarray      # (n_seeds, n_steps)
    compute_times: np.ndarray      # (n_seeds,) - wallclock seconds
    policies: np.ndarray           # (n_seeds, n_k, n_z)
    values: np.ndarray             # (n_seeds, n_k, n_z) or empty
    euler_errors: list             # List of 1D arrays per seed
    states_k: list                 # List of 1D arrays (capital states visited)
    utilities: np.ndarray          # (n_seeds,)
    value_from_policy: np.ndarray  # (n_seeds, n_k, n_z)
    
    # Derived metrics (computed with benchmark)
    cev_grid: np.ndarray           # (n_seeds, n_k, n_z)
    cev_relevant: np.ndarray       # (n_seeds,) - CEV on visited states
    wasserstein_dists: np.ndarray  # (n_seeds,)


# ----------------------------------------------------------------------------
# Data Loading & Processing
# ----------------------------------------------------------------------------

def compute_benchmark(params, state_min: float, state_max: float) -> Benchmark:
    """Compute EGM benchmark solution."""
    model = PlanningProblem(
        gamma=params.gamma,
        beta=params.beta,
        delta=params.delta,
        alpha=params.alpha,
        rho=params.rho,
        sigma=params.sigma,
    )
    solvers = PlanningSolvers(k_points=400, z_points=21, conv=1e-5)
    policy_bm = solvers.EGM(model)
    
    k_grid = model.make_k_grid(points=solvers.k_points)
    z_grid = model.make_z_grid(points=solvers.z_points)
    V_bm = model.V_from_a(k_grid, z_grid, policy_bm, conv=1e-3)
    
    # Benchmark Euler errors and utility
    euler_bm, _, U_bm = model.get_euler_errors(k_grid, z_grid, policy_bm, episodes=100, seed=ANALYSIS_SEED)
    euler_bm_log = np.log10(np.abs(euler_bm))
    
    # Benchmark return on evaluation grid, matching RL eval structure:
    # - Per-env independent seeding (DummyVecEnv.seed(s) gives env_i seed s+i)
    k_eval = np.linspace(state_min, state_max, 20)
    per_seed_returns = np.array([
        model.get_return_on_eval_grid(
            k_eval, policy_bm, T=100, seed=s, n_episodes=N_EVAL_EPISODES
        )
        for s in TRAINING_SEEDS
    ])
    
    return Benchmark(
        model=model,
        k_grid=k_grid,
        z_grid=z_grid,
        policy=policy_bm,
        V=V_bm,
        euler_errors_log=euler_bm_log.ravel(),
        utility=U_bm,
        returns_eval=per_seed_returns,
    )


def load_raw_metrics(algo: str) -> dict | None:
    """Load raw_metrics.json for a given algorithm."""
    json_path = RUNRESULTS_PATH / algo / "raw_metrics.json"
    if not json_path.exists():
        logger.warning(f"Missing raw_metrics.json for {algo}: {json_path}")
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_algo_results(raw: dict, benchmark: Benchmark, params) -> AlgoResults:
    """Process raw metrics and compute derived metrics using shared benchmark."""
    k_grid = np.array(raw["k_grid"])
    z_grid = np.array(raw["z_grid"])
    V_bm = benchmark.V
    euler_bm_log = benchmark.euler_errors_log
    
    # Convert raw data to numpy
    return_curves = np.array(raw["return_curves"])
    compute_times = np.array(raw.get("compute_times", [0.0] * len(raw["utilities"])))
    policies = np.array(raw["policies"])           # (n_seeds, n_k, n_z)
    values = np.array(raw["values"]) if raw["values"] else np.array([])
    euler_errors = [np.array(e) for e in raw["euler_errors"]]
    states_k = [np.array(s) for s in raw["states_k"]]
    utilities = np.array(raw["utilities"])
    value_from_policy = np.array(raw["value_from_policy"])  # (n_seeds, n_k, n_z)
    
    n_seeds = len(utilities)
    n_k, n_z = len(k_grid), len(z_grid)
    
    # Precompute mesh for grid evaluations
    K, Z = np.meshgrid(k_grid, z_grid, indexing="ij")
    pts_kz = np.column_stack([K.ravel(), Z.ravel()])
    
    # Compute CEV on grid for each seed
    cev_grid = np.zeros((n_seeds, n_k, n_z))
    cev_relevant = np.zeros(n_seeds)
    wasserstein_dists = np.zeros(n_seeds)
    
    # Determine beta for CEV calculation (only needed if gamma=1)
    beta_for_cev = params.beta if params.gamma == 1.0 else None
    
    for i in range(n_seeds):
        # Create interpolator from stored value function
        V_rl_interp = RegularGridInterpolator(
            (k_grid, z_grid),
            value_from_policy[i],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        
        # CEV on grid
        cev_flat = cev_crra(pts_kz, V_bm, V_rl_interp, gamma=params.gamma, beta=beta_for_cev)
        cev_grid[i] = cev_flat.reshape(n_k, n_z)
        
        # CEV on visited states (pair with z=1)
        k_vis = np.asarray(states_k[i]).ravel()
        z_vis = np.ones_like(k_vis)
        states_vis = np.column_stack([k_vis, z_vis])
        
        # Sample up to 1000 points for relevance approximation
        rng = np.random.default_rng(seed=CEV_SAMPLING_SEED + i)
        n_sample = min(1000, states_vis.shape[0])
        idx_sample = rng.choice(states_vis.shape[0], size=n_sample, replace=False)
        states_sample = states_vis[idx_sample, :]
        
        cev_relevant[i] = np.mean(
            cev_crra(states_sample, V_bm, V_rl_interp, gamma=params.gamma, beta=beta_for_cev)
        )
        
        # Wasserstein distance of Euler error distributions
        wasserstein_dists[i] = wasserstein_distance(euler_bm_log, euler_errors[i])
    
    return AlgoResults(
        seeds=raw.get("seeds", list(range(n_seeds))),
        k_grid=k_grid,
        z_grid=z_grid,
        return_curves=return_curves,
        compute_times=compute_times,
        policies=policies,
        values=values,
        euler_errors=euler_errors,
        states_k=states_k,
        utilities=utilities,
        value_from_policy=value_from_policy,
        cev_grid=cev_grid,
        cev_relevant=cev_relevant,
        wasserstein_dists=wasserstein_dists,
    )


def compute_summary_stats(results: AlgoResults, benchmark: Benchmark) -> dict:
    """Compute bootstrap CIs for all metrics."""
    def pack_ci(name, iqm, lo, hi):
        return {
            f"iqm_{name}": float(iqm[0]) if hasattr(iqm, '__len__') else float(iqm),
            f"lo_{name}": float(lo[0]) if hasattr(lo, '__len__') else float(lo),
            f"hi_{name}": float(hi[0]) if hasattr(hi, '__len__') else float(hi),
        }
    
    # Utility
    iqm_U, lo_U, hi_U = get_bootstrapped_iqm_ci(
        results.utilities.reshape(-1, 1), n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED
    )
    
    # CEV (grid average across all k and z)
    cev_grid_mean = results.cev_grid.reshape(results.cev_grid.shape[0], -1).mean(axis=1)
    iqm_cev, lo_cev, hi_cev = get_bootstrapped_iqm_ci(
        cev_grid_mean.reshape(-1, 1), n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED
    )
    
    # CEV (visited states)
    iqm_cev_rel, lo_cev_rel, hi_cev_rel = get_bootstrapped_iqm_ci(
        results.cev_relevant.reshape(-1, 1), n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED
    )
    
    # Max Euler error
    max_euler = np.array([np.max(e) for e in results.euler_errors]).reshape(-1, 1)
    iqm_euler, lo_euler, hi_euler = get_bootstrapped_iqm_ci(
        max_euler, n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED
    )
    
    # Wasserstein distance
    iqm_wd, lo_wd, hi_wd = get_bootstrapped_iqm_ci(
        results.wasserstein_dists.reshape(-1, 1), n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED
    )
    
    # Compute time (seconds)
    iqm_time, lo_time, hi_time = get_bootstrapped_iqm_ci(
        results.compute_times.reshape(-1, 1), n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED
    )
    
    return {
        "U_bm": float(benchmark.utility),
        "returns_bm": benchmark.returns_eval.tolist(),
        "max_abs_euler_error_bm": float(np.max(benchmark.euler_errors_log)),
        **pack_ci("U_rl", iqm_U, lo_U, hi_U),
        **pack_ci("cev", iqm_cev, lo_cev, hi_cev),
        **pack_ci("cev_relevant", iqm_cev_rel, lo_cev_rel, hi_cev_rel),
        **pack_ci("max_abs_euler_error_rl", iqm_euler, lo_euler, hi_euler),
        **pack_ci("wasserstein_dist", iqm_wd, lo_wd, hi_wd),
        **pack_ci("compute_time", iqm_time, lo_time, hi_time),
    }


# ----------------------------------------------------------------------------
# Plotting Functions
# ----------------------------------------------------------------------------

def get_z_slice_indices(z_grid: np.ndarray) -> tuple[list[int], list[str]]:
    """Get indices for min, median (z≈1), and max z values."""
    idx_min = 0
    idx_max = len(z_grid) - 1
    # Find index closest to z=1 (stationary mean)
    idx_med = int(np.argmin(np.abs(z_grid - 1.0)))
    
    indices = [idx_min, idx_med, idx_max]
    labels = [f"z={z_grid[idx_min]:.3f} (min)", 
              f"z={z_grid[idx_med]:.3f} (median)", 
              f"z={z_grid[idx_max]:.3f} (max)"]
    return indices, labels


def plot_learning_curves(
    algo_results: dict[str, AlgoResults], 
    benchmark: Benchmark,
    global_steps: np.ndarray,
    out_dir: Path,
):
    """Learning curves as regret (G_EGM - G_RL) with log scale."""
    _, ax = plt.subplots(figsize=(8, 5))

    for algo in ALGOS:
        if algo not in algo_results:
            continue
        
        results = algo_results[algo]
        curves = results.return_curves
        
        if len(curves) == 0:
            continue

        # Compute regret: (benchmark - RL return)
        regret_curves = benchmark.returns_eval.reshape(-1, 1) - curves
        regret_iqm, regret_lower, regret_upper = get_bootstrapped_iqm_ci(regret_curves, n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED)

        line = ax.plot(global_steps, regret_iqm, label=algo)
        ax.fill_between(global_steps, regret_lower, regret_upper, color=line[0].get_color(), alpha=0.2)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    ax.set_xlabel("Wallclock time (seconds)", fontsize=10)
    ax.set_ylabel(r"Regret $(G_{EGM} - G_{RL})$", fontsize=10)
    ax.set_title("Learning Curves (Brock-Mirman)")
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "1_learning_curves.pdf", format="pdf")
    plt.close()
    logger.info("Saved: 1_learning_curves.pdf")


def plot_policy_functions(
    algo_results: dict[str, AlgoResults],
    benchmark: Benchmark,
    out_dir: Path,
):
    """2x2 grid of policy functions with z-slices shown as different colors."""
    k_grid = benchmark.k_grid
    z_grid = benchmark.z_grid
    z_indices, z_labels = get_z_slice_indices(z_grid)
    
    # Precompute mesh for benchmark evaluation
    K, Z = np.meshgrid(k_grid, z_grid, indexing="ij")
    pts_kz = np.column_stack([K.ravel(), Z.ravel()])
    bm_policy_grid = benchmark.policy(pts_kz).reshape(len(k_grid), len(z_grid))
    
    # Colors for z-slices
    z_colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # Determine global y-limits
    y_min, y_max = bm_policy_grid.min(), bm_policy_grid.max()
    for algo in ALGOS:
        if algo in algo_results:
            policies = algo_results[algo].policies
            if len(policies) > 0:
                y_min = min(y_min, policies.min())
                y_max = max(y_max, policies.max())

    handles, labels = [], []

    for i, algo in enumerate(ALGOS):
        ax = axes[i]
        
        if algo not in algo_results:
            ax.set_title(f"{algo} (No Data)")
            ax.set_xlabel("Capital k")
            ax.set_ylabel("Consumption share a")
            continue

        results = algo_results[algo]
        policies = results.policies  # (n_seeds, n_k, n_z)
        
        if len(policies) == 0:
            ax.set_title(f"{algo} (No Data)")
            ax.set_xlabel("Capital k")
            ax.set_ylabel("Consumption share a")
            continue

        # Plot each z-slice with different colors
        for j, (z_idx, z_label, color) in enumerate(zip(z_indices, z_labels, z_colors)):
            # Extract policy shares for this z-slice
            a_slice = policies[:, :, z_idx]  # (n_seeds, n_k)
            
            iqm, lo, hi = get_bootstrapped_iqm_ci(a_slice, n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED)
            
            line, = ax.plot(k_grid, iqm, color=color, label=f"RL {z_label}")
            ax.fill_between(k_grid, lo, hi, alpha=0.2, color=color)
            ax.plot(k_grid, bm_policy_grid[:, z_idx], "--", color=color, alpha=0.6, 
                   label=f"EGM {z_label}" if i == 0 else None)
        
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

        ax.set_title(algo)
        ax.set_xlabel("Capital k")
        ax.set_ylabel("Consumption share a")

    # Unified y-limits
    pad = (y_max - y_min) * 0.05
    for ax in axes:
        ax.set_ylim(y_min - pad, y_max + pad)

    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(out_dir / "2_policy_functions.pdf", format="pdf")
    plt.close()
    logger.info("Saved: 2_policy_functions.pdf")


def plot_euler_errors(algo_results: dict[str, AlgoResults], out_dir: Path):
    """4x1 vertical stack of Euler error histograms."""
    _, axes = plt.subplots(len(ALGOS), 1, figsize=(8, 10))
    if len(ALGOS) == 1:
        axes = [axes]

    # Compute global x-limits across all algorithms
    all_errors = []
    for algo in ALGOS:
        if algo in algo_results:
            for e in algo_results[algo].euler_errors:
                all_errors.extend(e)
    
    if all_errors:
        all_errors = np.array(all_errors)
        x_min = np.percentile(all_errors, 0.1)
        x_max = all_errors.max()
    else:
        x_min, x_max = -5, 0

    for i, algo in enumerate(ALGOS):
        ax = axes[i]
        
        if algo in algo_results and algo_results[algo].euler_errors:
            euler_flat = np.concatenate(algo_results[algo].euler_errors)
            ax.hist(euler_flat, bins=50, density=True, alpha=0.8, edgecolor='white', linewidth=0.5)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(algo, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)

        ax.set_xlim(x_min - 0.5, x_max + 0.5)

    axes[-1].set_xlabel(r"$\log_{10} | \text{Euler Error} |$", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_dir / "3_euler_errors.pdf", format="pdf")
    plt.close()
    logger.info("Saved: 3_euler_errors.pdf")


def plot_cev(
    algo_results: dict[str, AlgoResults],
    benchmark: Benchmark,
    out_dir: Path,
):
    """1x4 CEV plots with z-slices shown as different colors and state density overlay."""
    k_grid = benchmark.k_grid
    z_grid = benchmark.z_grid
    z_indices, z_labels = get_z_slice_indices(z_grid)
    
    # Colors for z-slices
    z_colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    # Pre-compute stats for global y-limits
    plot_data = {}
    y_min, y_max = np.inf, -np.inf
    
    for algo in ALGOS:
        if algo in algo_results:
            cev_vals = algo_results[algo].cev_grid
            if len(cev_vals) > 0:
                algo_data = {}
                for z_idx in z_indices:
                    cev_slice = cev_vals[:, :, z_idx]
                    iqm, lo, hi = get_bootstrapped_iqm_ci(cev_slice, n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED)
                    y_min = min(y_min, np.nanmin(lo[lo > 0]) if np.any(lo > 0) else y_min)
                    y_max = max(y_max, np.nanmax(hi))
                    algo_data[z_idx] = (iqm, lo, hi)
                plot_data[algo] = algo_data

    if y_min == np.inf or y_min <= 0:
        y_min = 1e-4
    if y_max == -np.inf:
        y_max = 0.1
    else:
        y_max = y_max * 1.1

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    handles, labels = [], []

    for i, algo in enumerate(ALGOS):
        ax = axes[i]

        if algo not in plot_data:
            ax.set_title(f"{algo} (No Data)")
            ax.set_xlabel("Capital k")
            continue

        # Plot each z-slice with different colors
        for j, (z_idx, z_label, color) in enumerate(zip(z_indices, z_labels, z_colors)):
            iqm, lo, hi = plot_data[algo][z_idx]
            ax.plot(k_grid, iqm, color=color, label=z_label, linewidth=2, zorder=3)
            ax.fill_between(k_grid, lo, hi, alpha=0.2, color=color, zorder=2)

        ax.set_title(algo)
        ax.set_xlabel("Capital k")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
        ax.set_ylim(y_min, y_max)

        # State density overlay
        states_k = algo_results[algo].states_k
        if states_k:
            states_flat = np.concatenate(states_k)
            ax2 = ax.twinx()
            ax2.hist(states_flat, bins=50, density=True, alpha=0.2, color='gray', 
                    edgecolor=None, label="Visited States")
            ax2.set_yticks([])
            ax2.set_ylabel("")

            if not handles:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                handles = h1 + h2
                labels = l1 + l2
        elif not handles:
            handles, labels = ax.get_legend_handles_labels()

        if i == 0:
            ax.set_ylabel("Consumption Equivalent Variation (CEV)", fontsize=10)

    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(out_dir / "4_cev_analysis.pdf", format="pdf")
    plt.close()
    logger.info("Saved: 4_cev_analysis.pdf")


def generate_comparison_table(
    summary_stats: dict[str, dict],
    out_dir: Path,
):
    """Generate LaTeX comparison table from summary stats."""
    if not summary_stats:
        logger.error("No data available for table generation.")
        return

    table_path = out_dir / "comparison_table.tex"
    generate_latex_table(summary_stats, ALGOS, table_path)
    logger.info(f"Saved: {table_path}")


def save_full_metrics(
    algo_results: dict[str, AlgoResults],
    summary_stats: dict[str, dict],
    benchmark: Benchmark,
    out_dir: Path,
):
    """Save full metrics JSON with all computed data for reproducibility."""
    metrics = {
        "benchmark": {
            "returns_eval": benchmark.returns_eval.tolist(),
            "utility": float(benchmark.utility),
            "max_abs_euler_error_log": float(np.max(benchmark.euler_errors_log)),
            "k_grid": benchmark.k_grid.tolist(),
            "z_grid": benchmark.z_grid.tolist(),
        },
        "bootstrap_config": {
            "n_boot": N_BOOT,
            "alpha": ALPHA,
            "seed": BOOTSTRAP_SEED,
        },
        "algorithms": {},
    }
    
    for algo, results in algo_results.items():
        metrics["algorithms"][algo] = {
            "n_seeds": len(results.seeds),
            "seeds": results.seeds,
            "summary": summary_stats.get(algo, {}),
        }
    
    out_path = out_dir / "comparison_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved: {out_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run_comparison():
    """Main entry point for comparison analysis.
    
    Workflow:
    1. Load parameters and compute benchmark
    2. Load raw metrics for all algorithms
    3. Process each algorithm's metrics against the shared benchmark
    4. Compute summary statistics with bootstrap CIs
    5. Generate plots and LaTeX table
    """
    logger.info(f"Loading results from: {RUNRESULTS_PATH}")

    # 1. Load parameters
    params = BrockMirmanParams.load_params(TOML_PATH)
    
    # Compute state bounds from economic model
    model = PlanningProblem(
        gamma=params.gamma,
        beta=params.beta,
        delta=params.delta,
        alpha=params.alpha,
        rho=params.rho,
        sigma=params.sigma,
    )
    k_grid = model.make_k_grid()
    state_min, state_max = k_grid[0], k_grid[-1]
    
    # Compute benchmark
    benchmark = compute_benchmark(params, state_min, state_max)

    # 2 & 3. Load raw metrics and process for all algorithms
    algo_results: dict[str, AlgoResults] = {}
    global_steps = None
    
    for algo in ALGOS:
        raw = load_raw_metrics(algo)
        if raw is not None:
            results = process_algo_results(raw, benchmark, params)
            algo_results[algo] = results
            logger.info(f"Processed {algo}: {len(results.seeds)} seeds")
            
            # Infer global_steps from return_curves shape (same across all algos)
            if global_steps is None and len(results.return_curves) > 0:
                n_steps = results.return_curves.shape[1]
                # max_processed_samples=1_000_000, eval_every=10_000
                global_steps = np.linspace(0, n_steps-1, n_steps)

    if not algo_results:
        logger.error("No algorithm results found. Run training first.")
        return
    
    if global_steps is None:
        logger.error("Could not infer global_steps from return curves.")
        return

    # 4. Compute summary statistics with bootstrap CIs
    summary_stats = {}
    for algo, results in algo_results.items():
        summary_stats[algo] = compute_summary_stats(results, benchmark)

    # Create output directory
    out_dir = RUNRESULTS_PATH / "Comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5. Generate plots
    plot_learning_curves(algo_results, benchmark, global_steps, out_dir)
    plot_policy_functions(algo_results, benchmark, out_dir)
    plot_euler_errors(algo_results, out_dir)
    plot_cev(algo_results, benchmark, out_dir)
    
    # Generate LaTeX table
    generate_comparison_table(summary_stats, out_dir)
    
    # Save full metrics for reproducibility
    save_full_metrics(algo_results, summary_stats, benchmark, out_dir)

    logger.info(f"Comparison complete. Results in: {out_dir}")


if __name__ == "__main__":
    run_comparison()
