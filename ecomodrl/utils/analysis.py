import json
import logging
from pathlib import Path
from typing import Callable, Union, Mapping, Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import get_device

from ecomodrl.algorithms.training import ALGO_REGISTRY
from ecomodrl.envs.setup_envs import build_envs
from ecomodrl.envs.casskoopmans import c_from_next_k
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem
from ecomodrl.utils.metrics.accuracy import wev, cev_crra
from ecomodrl.utils.metrics.performance import interquartile_range, interquartile_mean
from ecomodrl.utils.tb_reader import tb_read_scalar

def get_policy_value_functions(
    model_path: Path, 
    vecnorm_path: Path, 
    cfg: Union[Mapping[str, Any], Any],
    env: Callable[..., Any],
    params: Union[Mapping[str, Any], Any],
    grid,
    algo_name: str):
    """ 
    Given a trained model and VecNormalize, return policy and value functions
    as CubicSpline interpolators on the specified grid.
    """
    device = get_device("auto")
    ALGO = ALGO_REGISTRY[algo_name.lower()]
    model: BaseAlgorithm = ALGO.load(model_path, device=device)  # Restores best policy
    # create a dummy VecNormalize with same stats for obs normalization
    dummy_env, _ = build_envs(
        env,
        params=params, 
        cfg=cfg,
        seed=1234,
        )
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
    vecnorm.training = False

    obs = grid.reshape(-1, 1).astype(np.float32, copy=False)
    obs_norm = vecnorm.normalize_obs(obs)
    
    # Get policy function for normalized observations
    actions, _ = model.predict(obs_norm, deterministic=True)
    # TODO: Make env-agnostic
    c_rl = c_from_next_k(obs.squeeze(), np.asarray(actions).squeeze(), params.alpha, params.delta)
    policy_rl = CubicSpline(obs.squeeze(), c_rl, bc_type="natural", extrapolate=True)

    # Get value function for normalized observations
    value_rl = None
    if hasattr(getattr(model, "policy", None), "predict_values"): # PPO
        with th.no_grad():
            V_norm = model.policy.predict_values(th.as_tensor(obs_norm, dtype=th.float32, device=device)).cpu().numpy().squeeze()
        ret_rms = getattr(vecnorm, "ret_rms", None)
        eps = float(getattr(vecnorm, "epsilon", 1e-8))
        scale = float(np.sqrt(getattr(ret_rms, "var", 1.0) + eps)) if ret_rms is not None else 1.0
        V_nn = V_norm * scale
        value_rl = CubicSpline(obs.squeeze(), V_nn, bc_type="natural", extrapolate=True)
    else:
        # Off-policy critics: V(s) ≈ min_i Q_i(s, π(s))
        policy = getattr(model, "policy", None)
        if policy is not None and hasattr(policy, "critic"):
            with th.no_grad():
                obs_t = th.as_tensor(obs_norm, dtype=th.float32, device=device)
                act_t = th.as_tensor(actions, dtype=th.float32, device=device).view(obs_t.shape[0], -1)
                q_out = policy.critic(obs_t, act_t)
                if isinstance(q_out, (list, tuple)): # SAC: Twin critics
                    qs = [q.flatten() for q in q_out]
                    q_min = th.min(th.stack(qs, dim=0), dim=0).values
                else: # DDPG/TD3
                    q_min = q_out.flatten()
                V_q = q_min.cpu().numpy()
            value_rl = CubicSpline(obs.squeeze(), V_q, bc_type="natural", extrapolate=True)
            
    return policy_rl, value_rl

# TODO: Refactor for other envs!
def compute_benchmark(
    params: Union[Mapping[str, Any], Any], 
    k_min: float, 
    k_max: float
    ):
    p = PlanningProblem(σ=params.gamma, β=params.beta, δ=params.delta, α=params.alpha, A=params.A)
    grid_bm, policy_bm = p.egm(grid_points=200, lower_limit=k_min, upper_limit=k_max, conv=1e-4, verbose=False)
    V_bm = p.get_vfrom_copt(grid_bm, policy_bm, conv=1e-3)
    return p, grid_bm, policy_bm, V_bm

# TODO: Bias-corrected and accelerated bootstrap CI? 
def bootstrap_mean_ci(
    Y: np.ndarray, *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 123
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-sided bootstrap over seeds with replacement; reproducible via 'seed'.
    Args:
        Y: array of shape (n_seeds, n_points).
    Returns:
        mean, lower, upper arrays of shape (n_points,).
    """
    rng = np.random.default_rng(seed)
    n_seeds, _ = Y.shape
    idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    samples = Y[idx, :]
    boot_means = samples.mean(axis=1)
    mean = Y.mean(axis=0)
    lower = np.percentile(boot_means, 100 * (alpha / 2), axis=0)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return mean, lower, upper

def analyze_all_seeds(
    cfg: Union[Mapping[str, Any], Any],
    algo_cfg: Union[Mapping[str, Any], Any], 
    env: Callable[..., Any],
    params: Union[Mapping[str, Any], Any],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    bootstrap_seed: int = 123
    ):
    """
    Aggregate analysis and plotting across seeds:
      1) Policy + Value plots with mean and 95% bootstrap CI (CI only for NN value).
      2) Euler error histogram (pooled) + per-seed KDE overlays; EGM histogram as reference.
      3) WEV/CEV plots with mean and 95% bootstrap CI; pooled state histogram on twin axes.
    Saves: policy_value_agg.png, euler_errors_agg.png, wev_cev_agg.png in cfg.out_dir.
    """
    plt.style.use("seaborn-v0_8-colorblind")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # common grid for RL & Benchmark evals
    grid = np.linspace(cfg.state_min, cfg.state_max, 200, dtype=np.float32)
    
    # Grid for average return plot
    global_steps = np.linspace(0, cfg.total_timesteps, num=100)
    
    # Benchmark once
    p, _, policy_bm, V_bm = compute_benchmark(params, cfg.state_min, cfg.state_max)
    euler_bm, _, U_bm = p.get_euler_errors(grid, policy_bm, n_eps=100)
    euler_bm = np.log10(np.abs(euler_bm))
    
    # storage across seeds
    return_curves = []
    rl_policy_vals = []       
    rl_value_nn_vals = []     
    rl_value_from_policy = [] 
    euler_logabs_list = []    
    pooled_states = []        
    Us_rl = []               
    WEV_mat = []
    CEV_mat = []
    WEV_rel_mat = []
    CEV_rel_mat = []
    seed_ids = list(cfg.seeds)
    
    train_cfgs_path = out_dir / "train_cfgs.json"
    with open(train_cfgs_path, "r") as f:
        train_cfgs = json.load(f)   # list of dicts

    # Policy/value/euler/wev/cev per seed
    for seed_entry in train_cfgs:
        best_model_path = Path(seed_entry["best_model_path"])
        vecnorm_path = Path(seed_entry["vecnorm_path"])
        
        # average evaluation return/learning curves
        s = seed_entry["seed"]
        tb_path = out_dir / f"seed_{s}"
        steps, returns = tb_read_scalar(tb_path, "eval/avg_discounted_return")
        
        # RL policy + value (NN spline)
        policy_rl, value_rl = get_policy_value_functions(
            best_model_path, vecnorm_path, 
            cfg, env, params, 
            grid, algo_cfg.name
        )

        # Value recomputed from policy (for plotting mean only)
        V_rl = p.get_vfrom_copt(grid, policy_rl, conv=1e-3)

        # Euler errors (RL), states, and U metrics
        euler_rl, states, U_rl = p.get_euler_errors(grid, policy_rl, n_eps=100)
        
        wev_seed = wev(grid, V_bm, V_rl)
        if wev_seed is not None:
            wev_on_grid = wev_seed
            avg_wev_relevant = wev(states, V_bm, V_rl).mean()
            WEV_mat.append(wev_on_grid)
            WEV_rel_mat.append(avg_wev_relevant)
        cev_on_grid = cev_crra(grid, V_bm, V_rl, σ=params.gamma)
        
        avg_cev_relevant = cev_crra(states, V_bm, V_rl, σ=params.gamma).mean()
        
        # stash
        return_curves.append(np.interp(global_steps, steps, returns))
        rl_policy_vals.append(policy_rl(grid))
        rl_value_nn_vals.append(value_rl(grid))
        rl_value_from_policy.append(V_rl(grid))
        euler_logabs_list.append(np.log10(np.abs(euler_rl)))
        pooled_states.append(states)
        Us_rl.append(U_rl)
        CEV_mat.append(cev_on_grid)
        CEV_rel_mat.append(avg_cev_relevant)

    # stack for bootstrap shapes [n_seeds, n_grid]
    return_curves = np.vstack(return_curves)
    rl_policy_vals = np.vstack(rl_policy_vals)
    rl_value_nn_vals = np.vstack(rl_value_nn_vals)
    rl_value_from_policy = np.vstack(rl_value_from_policy)
    WEV_mat = np.vstack(WEV_mat)  
    CEV_mat = np.vstack(CEV_mat) 
    
    # ----------- Returns ---------------------------
    # Average Returns for RL policies on k0-grid + CI
    mean_Us, lo_Us, hi_Us = bootstrap_mean_ci(
        np.vstack(Us_rl), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    
    # Average eval return over training run with IQR
    iqm_return = interquartile_mean(return_curves)
    q25_return, q75_return = interquartile_range(return_curves) #np.percentile(return_curves, [25, 75], axis=0, method="interpolated_inverted_cdf")
    # Benchmark return: Grid differs for evaluation during training (only 20 grid points!)
    _, _, return_bm = p.get_euler_errors( 
        np.linspace(cfg.state_min, cfg.state_max, 20), 
        policy_bm, 
        n_eps=100)
    
    fig = plt.figure(figsize=(9,4.5))
    plt.fill_between(global_steps, q25_return, q75_return, alpha=0.3, label="IQR")
    plt.plot(global_steps, iqm_return, label="Interquartile Mean")
    plt.axhline(return_bm, linestyle="-.", color="darkorange", label="Benchmark")
    plt.xlabel("Timesteps"); plt.ylabel("Avg discounted episode return (true scale)")
    plt.title("PPO on Cass–Koopmans (evaluation)")
    plt.legend(); plt.tight_layout()
    fig.savefig(cfg.out_dir / "learning_curve_agg.png", dpi=150)
    plt.close(fig)
    
    # ---------- 1) Policy + Value aggregate ----------
    mean_pol, lo_pol, hi_pol = bootstrap_mean_ci(
        rl_policy_vals, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    mean_Vnn, lo_Vnn, hi_Vnn = bootstrap_mean_ci(
        rl_value_nn_vals, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed + 1
    )
    mean_Vfrom = rl_value_from_policy.mean(axis=0)  # mean only (no CI)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    # Policy
    ax[0].plot(grid, mean_pol, label="RL Policy (mean)")
    ax[0].fill_between(grid, lo_pol, hi_pol, alpha=0.25, label="95% CI (bootstrap)")
    ax[0].plot(grid, policy_bm(grid), linestyle="--", label="Benchmark Policy (EGM)")
    ax[0].set_title("Policy Function")
    ax[0].set_xlabel("Capital k")
    ax[0].set_ylabel("Consumption")
    ax[0].legend()

    # Value
    ax[1].plot(grid, mean_Vnn, label="RL Value (NN mean)")
    ax[1].fill_between(grid, lo_Vnn, hi_Vnn, alpha=0.25, label="95% CI (bootstrap)")
    ax[1].plot(grid, V_bm(grid), linestyle="--", label="Benchmark Value (EGM)")
    ax[1].plot(grid, mean_Vfrom, label="RL Value (from policy mean)")
    ax[1].set_title("Value Function V(k)")
    ax[1].set_xlabel("Capital k")
    ax[1].set_ylabel("Value")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(out_dir / "policy_value_agg.png", dpi=150)
    plt.close(fig)

    # ---------- 2) Euler histogram (pooled) + per-seed KDEs ----------
    # pool RL errors and EGM errors across seeds
    mean_max_euler, lo_max_euler, hi_max_euler = bootstrap_mean_ci(
        np.max(euler_logabs_list, axis=1).reshape(-1, 1), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    
    rl_all = np.concatenate(euler_logabs_list, axis=0)

    fig = plt.figure(figsize=(7.2, 4.5))
    # pooled histograms
    plt.hist(rl_all, bins=50, density=True, alpha=0.7, label="RL policy (pooled)")
    plt.axvline(euler_bm.mean(), color="gray", linestyle="--", label="Mean EGM",)

    # per-seed KDE overlays (subtle)
    for i, arr in enumerate(euler_logabs_list):
        try:
            kde = gaussian_kde(arr)
            xs = np.linspace(
                min(np.min(rl_all), np.min(euler_bm)),
                max(np.max(rl_all), np.max(euler_bm)), 
                256
                )
            plt.plot(xs, kde(xs), alpha=0.3, linewidth=1.0)
        except Exception as exc:
            # KDE can fail if variance ~0; skip quietly
            logging.warning("Skipping KDE for seed %s due to: %s", seed_ids[i], exc)
            pass

    plt.title("Log10 Euler Equation Errors")
    plt.xlabel("log10(|error|)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "euler_errors_agg.png", dpi=150)
    plt.close(fig)

    # ---------- 3) WEV / CEV with CIs; pooled state hist ----------
    # Average wev on grid + CI
    mean_wev, lo_wev, hi_wev = bootstrap_mean_ci(
        WEV_mat.mean(axis=1).reshape(-1, 1), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    # Average wev for relevant states only + CI
    mean_rel_wev, lo_rel_wev, hi_rel_wev = bootstrap_mean_ci(
        np.vstack(WEV_rel_mat), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    # Full wev on grid + CI
    mean_WEV, lo_WEV, hi_WEV = bootstrap_mean_ci(
        WEV_mat, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed + 2
    )
    # Average cev on grid + CI
    mean_cev, lo_cev, hi_cev = bootstrap_mean_ci(
        CEV_mat.mean(axis=1).reshape(-1, 1), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    # Average cev for relevant states only + CI
    mean_rel_cev, lo_rel_cev, hi_rel_cev = bootstrap_mean_ci(
        np.vstack(CEV_rel_mat), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    # Full wev on grid + CI
    mean_CEV, lo_CEV, hi_CEV = bootstrap_mean_ci(
        CEV_mat, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed + 3
    )

    # pooled states across seeds
    states_all = np.concatenate(pooled_states, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # WEV
    ax[0].plot(grid, mean_WEV, label="WEV (mean)")
    ax[0].fill_between(grid, lo_WEV, hi_WEV, alpha=0.25, label="95% CI (bootstrap)")
    ax[0].set_ylim(bottom=0.0)
    ax[0].set_xlabel("Capital k")
    ax[0].set_ylabel("WEV")
    ax[0].set_title("Wealth-Equivalent Variation")
    ax0p = ax[0].twinx()
    ax0p.hist(states_all, bins=30, density=True, alpha=0.3, color="gray")
    ax0p.set_ylabel("State density")

    # CEV
    ax[1].plot(grid, mean_CEV, label="CEV (mean)")
    ax[1].fill_between(grid, lo_CEV, hi_CEV, alpha=0.25, label="95% CI (bootstrap)")
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_xlabel("Capital k")
    ax[1].set_ylabel("CEV")
    ax[1].set_title("Consumption-Equivalent Variation")
    ax1p = ax[1].twinx()
    ax1p.hist(states_all, bins=30, density=True, alpha=0.3, color="gray")
    ax1p.set_ylabel("State density")

    for a in ax:
        a.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "wev_cev_agg.png", dpi=150)
    plt.close(fig)
    
    
    metrics = {
        "U_bm": float(U_bm),
        "mean_U_rl": float(mean_Us[0]),
        "lo_U_rl": float(lo_Us[0]),
        "hi_U_rl": float(hi_Us[0]),
        "mean_wev": float(mean_wev[0]),
        "lo_wev": float(lo_wev[0]), 
        "hi_wev": float(hi_wev[0]),
        "mean_wev_relevant": float(mean_rel_wev[0]),
        "lo_wev_relevant": float(lo_rel_wev[0]),
        "hi_wev_relevant": float(hi_rel_wev[0]),
        "mean_cev": float(mean_cev[0]),
        "lo_cev": float(lo_cev[0]),
        "hi_cev": float(hi_cev[0]),
        "mean_cev_relevant": float(mean_rel_cev[0]),
        "lo_cev_relevant": float(lo_rel_cev[0]),
        "hi_cev_relevant": float(hi_rel_cev[0]),
        "max_abs_euler_error_bm": float(np.max(euler_bm)),
        "mean_max_abs_euler_error_rl": float(mean_max_euler[0]),
        "lo_max_abs_euler_error_rl": float(lo_max_euler[0]),
        "hi_max_abs_euler_error_rl": float(hi_max_euler[0]),
    }
    with (out_dir / "metrics_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)