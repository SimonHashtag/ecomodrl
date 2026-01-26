import json
import logging
import os
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch as th
from scipy.interpolate import CubicSpline
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from ecomodrl.algorithms.training import ALGO_REGISTRY
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem, PlanningSolvers
from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.envs.setup_envs import build_envs
from ecomodrl.utils.metrics.accuracy import cev_crra
from ecomodrl.utils.metrics.performance import get_bootstrapped_iqm_ci
from ecomodrl.utils.params_loader import CassKoopmansParams
from ecomodrl.utils.tb_reader import tb_read_scalar

# ----------------------------------------------------------------------------
# Global Configuration
# ----------------------------------------------------------------------------
ALGOS = ["PPO", "DDPG", "SAC", "TD3"]
TOML_PATH = Path("ecomodrl/experiments/model_params/casskoopmans.toml")
RUNRESULTS_PATH = Path(os.environ.get("RUNRESULTS_PATH", "./runs/CassKoopmans"))

# Set plotting style globally
plt.style.use("seaborn-v0_8-colorblind")

# Configure LaTeX rendering
plt.rcParams.update({
    # Font styles
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.serif": ["cmr10"],
    "axes.formatter.use_mathtext": True,
    # Font sizes
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CombinedAnalysis")

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def compute_benchmark(params):
    """Computes the EGM benchmark solution."""
    model = PlanningProblem(
        gamma=params.gamma,
        beta=params.beta,
        delta=params.delta,
        alpha=params.alpha,
        A=params.A
    )
    solvers = PlanningSolvers(grid_points=1000, conv=1e-5)
    policy_bm = solvers.EGM(model)
    grid = model.make_grid(solvers.grid_points)
    V_bm = model.V_from_a(grid, policy_bm, conv=1e-5)
    return model, grid, policy_bm, V_bm

def get_policy_value_functions(
    model_path: Path,
    vecnorm_path: Path,
    cfg: Any,
    env: Callable[..., Any],
    params: Any,
    grid: np.ndarray,
    algo_name: str,
):
    """Loads a trained model and returns Policy and Value functions as CubicSplines."""
    device = get_device("auto")
    ALGO = ALGO_REGISTRY[algo_name.lower()]
    model = ALGO.load(model_path, device=device)

    # Rebuild dummy env for normalization
    dummy_env, _ = build_envs(env, params=params, cfg=cfg, seed=1234)
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
    vecnorm.training = False

    obs = grid.reshape(-1, 1).astype(np.float32, copy=False)
    obs_norm = vecnorm.normalize_obs(obs)

    # Policy
    actions, _ = model.predict(obs_norm, deterministic=True)
    a_rl = np.asarray(actions).squeeze()
    policy_spline = CubicSpline(obs.squeeze(), a_rl, bc_type="natural", extrapolate=True)

    # Value
    value_spline = None
    # Check for PPO-style value prediction
    if hasattr(getattr(model, "policy", None), "predict_values"):
        with th.no_grad():
            V_norm = model.policy.predict_values(
                th.as_tensor(obs_norm, dtype=th.float32, device=device)
            ).cpu().numpy().squeeze()
        ret_rms = getattr(vecnorm, "ret_rms", None)
        eps = float(getattr(vecnorm, "epsilon", 1e-8))
        scale = float(np.sqrt(getattr(ret_rms, "var", 1.0) + eps)) if ret_rms else 1.0
        V_nn = V_norm * scale
        value_spline = CubicSpline(obs.squeeze(), V_nn, bc_type="natural", extrapolate=True)
    else:
        # Off-policy (SAC, TD3, DDPG)
        policy = getattr(model, "policy", None)
        if policy is not None and hasattr(policy, "critic"):
            with th.no_grad():
                obs_t = th.as_tensor(obs_norm, dtype=th.float32, device=device)
                act_t = th.as_tensor(actions, dtype=th.float32, device=device).view(obs_t.shape[0], -1)
                q_out = policy.critic(obs_t, act_t)
                if isinstance(q_out, (list, tuple)):
                    qs = [q.flatten() for q in q_out]
                    q_min = th.min(th.stack(qs, dim=0), dim=0).values
                else:
                    q_min = q_out.flatten()
                Q_norm = q_min.cpu().numpy()
            ret_rms = getattr(vecnorm, "ret_rms", None)
            eps = float(getattr(vecnorm, "epsilon", 1e-8))
            if getattr(vecnorm, "norm_reward", False) and ret_rms is not None:
                scale = float(np.sqrt(ret_rms.var + eps))
                Q = Q_norm * scale
            else:
                Q = Q_norm
            value_spline = CubicSpline(obs.squeeze(), Q, bc_type="natural", extrapolate=True)

    return policy_spline, value_spline


# ----------------------------------------------------------------------------
# Plotting Functions
# ----------------------------------------------------------------------------

def plot_learning_curves(results, steps, benchmark_ret, out_dir):

    _, ax = plt.subplots(figsize=(8, 5))
    
    # Absolute Relative Regret transformation
    transform = lambda x: np.abs(1 - x / benchmark_ret)

    for algo, data in results.items():
        curves = data["return_curves"] # Shape: (n_seeds, n_steps)
        if len(curves) == 0: continue
        
        # Compute stats on the raw return curves first
        iqm, ci_lower, ci_upper = get_bootstrapped_iqm_ci(curves)
        
        # Apply transformation to the aggregated stats
        iqm_trans = transform(iqm)
        lo_trans = transform(ci_lower)
        hi_trans = transform(ci_upper)
        
        line = ax.plot(steps, iqm_trans, label=algo)
        ax.fill_between(steps, lo_trans, hi_trans, color=line[0].get_color(), alpha=0.2)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    ax.set_xlabel("Processed Samples")
    ax.set_ylabel(r"Relative Regret $|1 - \frac{G_{RL}}{G_{EGM}}|$", fontsize=10)
    
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "1_learning_curves.pdf", format="pdf")
    plt.close()
    

def plot_policy_functions(results, grid, policy_bm, out_dir):
    """(2x2) Policy functions with CI and unified legend."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Determine global limits for uniform axes
    y_min, y_max = np.inf, -np.inf
    bm_vals = policy_bm(grid)
    y_min, y_max = min(y_min, bm_vals.min()), max(y_max, bm_vals.max())
    
    for data in results.values():
        if len(data["policies"]) > 0:
            y_min = min(y_min, data["policies"].min())
            y_max = max(y_max, data["policies"].max())
    
    # Storage for legend handles
    handles, labels = [], []

    # Plot
    for i, algo in enumerate(ALGOS):
        ax = axes[i]
        if algo not in results or len(results[algo]["policies"]) == 0:
            ax.set_title(f"{algo} (No Data)")
            ax.set_xlabel("Capital k")
            ax.set_ylabel("Consumption c")
            continue

        data = results[algo]["policies"]
        iqm, lo, hi = get_bootstrapped_iqm_ci(data)
        
        ax.plot(grid, iqm, label="RL IQM")
        ax.fill_between(grid, lo, hi, alpha=0.3, label="95% CI")
        ax.plot(grid, bm_vals, "k--", label="EGM", alpha=0.6)
        
        # Capture handles/labels from a populated axis
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

        ax.set_title(algo)
        
        ax.set_xlabel("Capital k")
        ax.set_ylabel("Consumption c")
        # Individual legend removed
                
    # Set unified limits
    pad = (y_max - y_min) * 0.05
    for ax in axes:
        ax.set_ylim(y_min - pad, y_max + pad)
    
    # Unified Legend
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_dir / "2_policy_functions.pdf", format="pdf")
    plt.close()


def plot_value_functions(results, grid, V_bm, out_dir):
    """(2x2) Value functions with CI and unified legend."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Determine global limits for uniform axes
    y_min, y_max = np.inf, -np.inf
    bm_vals = V_bm(grid)
    y_min, y_max = min(y_min, bm_vals.min()), max(y_max, bm_vals.max())

    for data in results.values():
        if len(data["values"]) > 0:
            y_min = min(y_min, data["values"].min())
            y_max = max(y_max, data["values"].max())

    # Storage for legend handles
    handles, labels = [], []

    # Plot
    for i, algo in enumerate(ALGOS):
        ax = axes[i]
        if algo not in results or len(results[algo]["values"]) == 0:
            ax.set_title(f"{algo} (No Data)")
            ax.set_xlabel("Capital k")
            ax.set_ylabel("Value V(k)")
            continue

        data = results[algo]["values"]
        iqm, lo, hi = get_bootstrapped_iqm_ci(data)
        
        ax.plot(grid, iqm, label="RL IQM")
        ax.fill_between(grid, lo, hi, alpha=0.3, label="95% CI")
        ax.plot(grid, bm_vals, "k--", label="EGM", alpha=0.6)
        
        # Capture handles/labels from a populated axis
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

        ax.set_title(algo)
        
        ax.set_xlabel("Capital k")
        ax.set_ylabel("Value V(k)")

    # Set unified limits
    pad = (y_max - y_min) * 0.05
    for ax in axes:
        ax.set_ylim(y_min - pad, y_max + pad)

    # Unified Legend
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_dir / "3_value_functions.pdf", format="pdf")
    plt.close()


def plot_euler_errors(results, out_dir):
    """(4x1) Vertical stack of Euler Error histograms with individual x-ticks."""
    _, axes = plt.subplots(len(ALGOS), 1, figsize=(8, 10))
    if len(ALGOS) == 1: axes = [axes]
    
    # Global X limits: Use 1st percentile to remove the smallest 0.1% of outliers
    all_errors = np.concatenate([data["euler_errors"] for data in results.values() if len(data["euler_errors"]) > 0])
    x_min = np.percentile(all_errors, 0.1) if len(all_errors) > 0 else -5
    x_max = all_errors.max() if len(all_errors) > 0 else 0
    
    for i, algo in enumerate(ALGOS):
        ax = axes[i]
        if algo in results and len(results[algo]["euler_errors"]) > 0:
            data = results[algo]["euler_errors"]
            ax.hist(data, bins=50, density=True, alpha=0.8, edgecolor='white', linewidth=0.5)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(algo, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
        
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
    
    axes[-1].set_xlabel(r"$\log_{10} | \text{Euler Error} |$", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(out_dir / "4_euler_errors.pdf", format="pdf")
    plt.close()


def plot_cev(results, grid, out_dir):
    """(1x4) CEV plots with State Density Histogram overlay and unified legend."""
    
    # 1. Pre-compute stats to determine global y-limits
    plot_data = {}
    y_min, y_max = np.inf, -np.inf
    for algo in ALGOS:
        if algo in results and len(results[algo]["cev"]) > 0:
            cev_vals = results[algo]["cev"]
            iqm, lo, hi = get_bootstrapped_iqm_ci(cev_vals)
            y_min = min(y_min, lo.min())
            y_max = max(y_max, hi.max())
            plot_data[algo] = (iqm, lo, hi)

    if y_min == np.inf: y_min, y_max = 0.001, 0.1  # Fallback if no data
    else:
        y_max = y_max * 1.1

    # 2. Setup Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    
    handles, labels = [], []

    for i, algo in enumerate(ALGOS):
        ax = axes[i]
        
        if algo not in plot_data:
            ax.set_title(f"{algo} (No Data)")
            ax.set_xlabel("Capital k")
            continue

        # Retrieve pre-computed stats
        iqm, lo, hi = plot_data[algo]
        data = results[algo] 
        
        # Plot CEV
        ax.plot(grid, iqm, label="CEV", linewidth=2, zorder=3)
        ax.fill_between(grid, lo, hi, alpha=0.3, label="95% CI", zorder=2)
        
        ax.set_title(algo)
        ax.set_xlabel("Capital k")
        ax.set_yscale("log") 
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.set_ylim(y_min, y_max)
        
        # Plot State Density
        if "states" in data and len(data["states"]) > 0:
            ax2 = ax.twinx()
            ax2.hist(data["states"], bins=50, density=True, alpha=0.2, color='gray', edgecolor=None, label="Visited States")
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

    # Unified Legend
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(out_dir / "5_cev_analysis.pdf", format="pdf")
    plt.close()

# ----------------------------------------------------------------------------
# Main Analysis Logic
# ----------------------------------------------------------------------------

def run_combined_analysis():
    # 1. Setup Model and Benchmark
    params = CassKoopmansParams.load_params(TOML_PATH)
    planning_prob, grid, policy_bm, V_bm = compute_benchmark(params)
    
    _, _, return_bm = planning_prob.get_euler_errors(
        np.linspace(grid[0], grid[-1], 20), policy_bm, episodes=100
    )

    mock_cfg = SimpleNamespace(
        max_episode_steps=30,
        max_eval_episode_steps=100,
        n_envs=1,
        state_min=grid[0],
        state_max=grid[-1]
    )

    results = {}
    # TODO: HARDCODED. MAKE DYNAMIC LATER!
    global_steps = np.linspace(0, 1_000_000, 101)

    # 2. Data Collection Loop
    for algo in ALGOS:
        algo_path = RUNRESULTS_PATH / algo
        json_path = algo_path / "train_cfgs.json"
        
        if not json_path.exists():
            logger.warning(f"Skipping {algo}: No train_cfgs.json found.")
            continue
            
        logger.info(f"Processing algorithm: {algo}")
        with open(json_path, "r") as f:
            train_cfgs = json.load(f)

        algo_data = {
            "return_curves": [],
            "policies": [],
            "values": [],
            "euler_errors": [],
            "cev": [],
            "states": []
        }

        for entry in train_cfgs:
            seed = entry["seed"]
            # A. Learning Curve
            tb_path = algo_path / f"seed_{seed}"
            try:
                _, returns = tb_read_scalar(tb_path, "eval/avg_discounted_return")
                _, steps_proc = tb_read_scalar(tb_path, "eval/x_processed_samples")
                interp_ret = np.interp(global_steps, steps_proc, returns)
                algo_data["return_curves"].append(interp_ret)
            except Exception as e:
                logger.warning(f"Could not read TB for seed {seed}: {e}")

            # B. Load Model & Compute Metrics
            try:
                pol_spline, val_spline = get_policy_value_functions(
                    Path(entry["best_model_path"]),
                    Path(entry["vecnorm_path"]),
                    mock_cfg,
                    CassKoopmansEnv,
                    params,
                    grid,
                    algo
                )
                
                algo_data["policies"].append(pol_spline(grid))
                algo_data["values"].append(val_spline(grid))

                euler_rl, states, _ = planning_prob.get_euler_errors(grid, pol_spline, episodes=100)
                algo_data["euler_errors"].append(np.log10(np.abs(euler_rl)))
                algo_data["states"].append(states) 

                V_rl_rec = planning_prob.V_from_a(grid, pol_spline, conv=1e-9)
                cev_val = cev_crra(grid, V_bm, V_rl_rec, gamma=params.gamma)
                algo_data["cev"].append(cev_val)

            except Exception as e:
                logger.error(f"  Failed model analysis for seed {seed}: {e}")

        # Stack lists into numpy arrays
        for k in algo_data:
            if algo_data[k]:
                # Euler errors and States flattened (1D array of all points)
                if k in ["euler_errors", "states"]:
                    algo_data[k] = np.concatenate(algo_data[k])
                else:
                    algo_data[k] = np.vstack(algo_data[k])
        
        results[algo] = algo_data

    # 3. Plotting
    out_dir = RUNRESULTS_PATH / "COMPARISON_PLOTS"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(results, global_steps, return_bm, out_dir)
    plot_policy_functions(results, grid, policy_bm, out_dir)
    plot_value_functions(results, grid, V_bm, out_dir)
    plot_euler_errors(results, out_dir)
    plot_cev(results, grid, out_dir)

    logger.info("Analysis complete. Check 'COMPARISON_PLOTS' folder.")


if __name__ == "__main__":
    # Required for PyTorch/CUDA compatibility to prevent deadlocks in subprocesses
    mp.set_start_method("spawn", force=True)
    # Execute the combined analysis
    run_combined_analysis()