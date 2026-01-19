import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from ecomodrl.algorithms.training import ALGO_REGISTRY
from ecomodrl.benchmarks.bufferstock.problemclass import PlanningProblem, PlanningSolvers
from ecomodrl.envs.setup_envs import build_envs
from ecomodrl.utils.metrics.accuracy import cev_crra, wev
from ecomodrl.utils.metrics.performance import interquartile_mean, interquartile_range
from ecomodrl.utils.tb_reader import tb_read_scalar


def bootstrap_mean_ci(
    Y: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-sided bootstrap confidence interval for mean across seeds.
    """
    # Initialize RNG and setup
    rng = np.random.default_rng(seed)
    n_seeds, _ = Y.shape

    # Draw bootstrap samples with replacement
    idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    samples = Y[idx, :]

    # Compute bootstrap means
    boot_means = samples.mean(axis=1)

    # Compute mean and percentile-based confidence interval
    mean = Y.mean(axis=0)
    lower = np.percentile(boot_means, 100 * (alpha / 2), axis=0)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)

    return mean, lower, upper


def compute_benchmark(params: Union[Mapping[str, Any], Any]):

    # Set up planning problem
    model = PlanningProblem(gamma=params.gamma,
                            beta=params.beta,
                            r=params.r,
                            g=params.g,
                            sigma_tran=params.sigma_tran,
                            sigma_perm=params.sigma_perm,
                            unemp_prob=params.unemp_prob,
                            theta_low=params.theta_low,
                            gh_order=params.gh_order)

    # Instantiate solvers class
    solvers = PlanningSolvers(grid_points=100, conv=1e-5)
    
    # Solve model using EGM
    policy_bm = solvers.EGM(model)
    
    # Derive the policy-implied value function
    grid = model.make_grid(solvers.grid_points)
    V_bm = model.V_from_a(grid, policy_bm, conv=1e-3)

    # Evaluation grid
    euler_grid = model.make_grid(solvers.grid_points, span=(1.5, 15.))

    # Return model class, grid, evaluation grid, optimal policy and corresponding value function
    return model, grid, euler_grid, policy_bm, V_bm

def get_policy_value_functions(
    model_path: Path,
    vecnorm_path: Path,
    cfg: Union[Mapping[str, Any], Any],
    env: Callable[..., Any],
    params: Union[Mapping[str, Any], Any],
    grid: np.ndarray,
    algo_name: str,
):
    """
    Given a trained model and VecNormalize, return policy and value functions
    as CubicSpline interpolators on the specified grid.
    """
    # Setup
    device = get_device("auto")
    ALGO = ALGO_REGISTRY[algo_name.lower()]

    # Load trained policy
    model: BaseAlgorithm = ALGO.load(model_path, device=device)

    # Create dummy VecNormalize with same stats for observation normalization
    dummy_env, _ = build_envs(
        env,
        params=params,
        cfg=cfg,
        seed=1234,
    )
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
    vecnorm.training = False

    # Prepare normalized observation grid
    obs = grid.reshape(-1, 1).astype(np.float32, copy=False)
    obs_norm = vecnorm.normalize_obs(obs)

    # Policy function
    actions, _ = model.predict(obs_norm, deterministic=True)
    a_rl = np.asarray(actions).squeeze()
    policy_rl = CubicSpline(obs.squeeze(), a_rl, bc_type="natural", extrapolate=True)

    # Value function
    value_rl = None

    # PPO-like algorithms: predict_values() available
    if hasattr(getattr(model, "policy", None), "predict_values"):
        with th.no_grad():
            V_norm = (
                model.policy.predict_values(
                    th.as_tensor(obs_norm, dtype=th.float32, device=device)
                ).cpu().numpy().squeeze()
            )

        ret_rms = getattr(vecnorm, "ret_rms", None)
        eps = float(getattr(vecnorm, "epsilon", 1e-8))
        scale = float(np.sqrt(getattr(ret_rms, "var", 1.0) + eps)) if ret_rms else 1.0

        V_nn = V_norm * scale
        value_rl = CubicSpline(obs.squeeze(), V_nn, bc_type="natural", extrapolate=True)

        # Off-policy algorithms (e.g., SAC, TD3, DDPG)
    else:
        policy = getattr(model, "policy", None)
        if policy is not None and hasattr(policy, "critic"):
            with th.no_grad():
                obs_t = th.as_tensor(obs_norm, dtype=th.float32, device=device)
                act_t = th.as_tensor(actions, dtype=th.float32, device=device).view(
                    obs_t.shape[0], -1
                )

                q_out = policy.critic(obs_t, act_t)
                if isinstance(q_out, (list, tuple)):  # SAC: twin critics
                    qs = [q.flatten() for q in q_out]
                    q_min = th.min(th.stack(qs, dim=0), dim=0).values
                else:  # DDPG / TD3
                    q_min = q_out.flatten()

                Q_norm = q_min.cpu().numpy()

            # --- UNNORMALIZE HERE (if reward normalization was used) ---
            ret_rms = getattr(vecnorm, "ret_rms", None)
            eps = float(getattr(vecnorm, "epsilon", 1e-8))
            if getattr(vecnorm, "norm_reward", False) and ret_rms is not None:
                scale = float(np.sqrt(ret_rms.var + eps))
                Q = Q_norm * scale
            else:
                Q = Q_norm

            value_rl = CubicSpline(obs.squeeze(), Q, bc_type="natural", extrapolate=True)

    return policy_rl, value_rl


def analyze_all_seeds(
    cfg: Union[Mapping[str, Any], Any],
    algo_cfg: Union[Mapping[str, Any], Any],
    env: Callable[..., Any],
    params: Union[Mapping[str, Any], Any],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    bootstrap_seed: int = 123,
):
    """
    Aggregate and visualize performance across training seeds.

    Generates:
      1) Learning curve plot (with IQR and benchmark line).
      2) Policy and value function plots with bootstrap confidence bands.
      3) Euler error histogram with per-seed KDE overlays.
      4) WEV and CEV plots with bootstrap CIs and state density overlays.

    Saves all figures and summary metrics JSON to cfg.out_dir.
    """
    # Setup and output directory
    plt.style.use("seaborn-v0_8-colorblind")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_steps = np.linspace(0, cfg.max_processed_samples, retstep=cfg.eval_every)[0]

    # Benchmark reference solution
    p, grid, euler_grid, policy_bm, V_bm = compute_benchmark(params)
    # TODO: Seeding! (Consistent with RL eval seed)
    euler_bm, _, U_bm = p.get_euler_errors(euler_grid, policy_bm, episodes=100)

    # --- Initialize containers for aggregation across seeds ---
    # Learning performance
    return_curves = []
    # Policy and value functions
    rl_policy_vals = []
    rl_value_nn_vals = []
    rl_value_from_policy = []
    # Euler errors and state visitation
    euler_logabs_list = []
    pooled_states = []
    Us_rl = []
    # Welfare metrics
    WEV_mat = []
    WEV_rel_mat = []
    CEV_mat = []
    CEV_rel_mat = []
    # Seed identifiers (for labeling/logging)
    seed_ids = list(cfg.seeds)

    # Load per-seed training metadata
    with (out_dir / "train_cfgs.json").open("r", encoding="utf-8") as f:
        train_cfgs = json.load(f)

    # Per-seed analysis: policy, value, errors, and welfare metrics
    for seed_entry in train_cfgs:
        best_model_path = Path(seed_entry["best_model_path"])
        vecnorm_path = Path(seed_entry["vecnorm_path"])
        s = seed_entry["seed"]
        tb_path = out_dir / f"seed_{s}"

        # --- Learning curve (evaluation performance) ---
        _, returns = tb_read_scalar(tb_path, "eval/avg_discounted_return")
        _, steps_proc = tb_read_scalar(tb_path, "eval/x_processed_samples")
        # Interpolate to global step grid for consistent aggregation
        return_curves.append(np.interp(global_steps, steps_proc, returns))

        # --- RL policy and value functions ---
        policy_rl, value_rl = get_policy_value_functions(
            best_model_path,
            vecnorm_path,
            cfg,
            env,
            params,
            grid,
            algo_cfg.name,
        )
        # Store policy and value predictions across grid
        rl_policy_vals.append(policy_rl(grid))
        rl_value_nn_vals.append(value_rl(euler_grid))

        # Value recomputed from policy via environment recursion
        V_rl = p.V_from_a(grid, policy_rl, conv=1e-9)
        rl_value_from_policy.append(V_rl(euler_grid))

        # --- Euler errors and episode utilities ---
        # Simulate policy to measure equilibrium errors and returns
        # TODO: Seeding!
        euler_rl, states, U_rl = p.get_euler_errors(euler_grid, policy_rl, episodes=10)
        euler_logabs_list.append(euler_rl)
        pooled_states.append(states)
        Us_rl.append(U_rl)

        # --- WEV and CEV metrics ---
        # Compute welfare differences vs. benchmark on full grid
        wev_seed = wev(grid, V_bm, V_rl, p.cash_on_hand)
        if wev_seed is not None:
            WEV_mat.append(wev_seed)
            # Average WEV over visited states (policy-relevant region)
            avg_wev_rel = wev(states, V_bm, V_rl, p.cash_on_hand).mean()
            WEV_rel_mat.append(avg_wev_rel)

        # Compute analogous CEV metrics (grid + visited states)
        cev_on_grid = cev_crra(grid, V_bm, V_rl, gamma=params.gamma)
        CEV_mat.append(cev_on_grid)
        avg_cev_rel = cev_crra(states, V_bm, V_rl, gamma=params.gamma).mean()
        CEV_rel_mat.append(avg_cev_rel)

    # --- Stack results across seeds for downstream analysis ---
    # Learning performance
    return_curves = np.vstack(return_curves)
    # Policy and value functions
    rl_policy_vals = np.vstack(rl_policy_vals)
    rl_value_nn_vals = np.vstack(rl_value_nn_vals)
    rl_value_from_policy = np.vstack(rl_value_from_policy)
    # Welfare metrics (grid-based)
    WEV_mat = np.vstack(WEV_mat)
    CEV_mat = np.vstack(CEV_mat)

    # ---------------------------------------------------------------
    # Learning curve aggregation
    # ---------------------------------------------------------------
    mean_Us, lo_Us, hi_Us = bootstrap_mean_ci(
        np.vstack(Us_rl), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    iqm_return = interquartile_mean(return_curves)
    q25_return, q75_return = interquartile_range(return_curves)

    # Benchmark evaluation return (different grid during training)
    # TODO: Seeding!
    _, _, return_bm = p.get_euler_errors(
        np.linspace(cfg.state_min, cfg.state_max, 20),
        policy_bm,
        episodes=cfg.max_eval_episode_steps,
    )

    fig = plt.figure(figsize=(9, 4.5))
    plt.fill_between(global_steps, q25_return, q75_return, alpha=0.3, label="IQR")
    plt.plot(global_steps, iqm_return, label="Interquartile Mean")
    plt.axhline(return_bm, linestyle="-.", color="darkorange", label="Benchmark")
    plt.xlabel("Processed Samples")
    plt.ylabel("Avg discounted episode return (true scale)")
    plt.title("Learning Curve (PPO Buffer Stock)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "learning_curve_agg.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Policy and value functions
    # ---------------------------------------------------------------
    mean_pol, lo_pol, hi_pol = bootstrap_mean_ci(
        rl_policy_vals, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    mean_Vnn, lo_Vnn, hi_Vnn = bootstrap_mean_ci(
        rl_value_nn_vals, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed + 1
    )
    mean_Vfrom = rl_value_from_policy.mean(axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    # Policy
    ax[0].plot(grid, mean_pol, label="RL Policy (mean)")
    ax[0].fill_between(grid, lo_pol, hi_pol, alpha=0.25, label="95% CI (bootstrap)")
    ax[0].plot(grid, policy_bm(grid), linestyle="--", label="Benchmark (EGM)")
    ax[0].set_title("Policy Function")
    ax[0].set_xlabel("Capital k")
    ax[0].set_ylabel("Consumption")
    ax[0].legend()

    # Value
    ax[1].plot(euler_grid, mean_Vnn, label="RL Value (NN mean)")
    ax[1].fill_between(euler_grid, lo_Vnn, hi_Vnn, alpha=0.25, label="95% CI (bootstrap)")
    ax[1].plot(euler_grid, V_bm(euler_grid), linestyle="--", label="Benchmark (EGM)")
    ax[1].plot(grid, mean_Vfrom, label="RL Value (from policy mean)")
    ax[1].set_title("Value Function V(k)")
    ax[1].set_xlabel("Capital k")
    ax[1].set_ylabel("Value")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(out_dir / "policy_value_agg.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Euler error distribution
    # ---------------------------------------------------------------
    mean_max_euler, lo_max_euler, hi_max_euler = bootstrap_mean_ci(
        np.nanmax(euler_logabs_list, axis=1).reshape(-1, 1),
        n_boot=n_boot,
        alpha=alpha,
        seed=bootstrap_seed,
    )
    rl_all = np.concatenate(euler_logabs_list, axis=0)

    fig = plt.figure(figsize=(7.2, 4.5))
    plt.hist(rl_all, bins=50, density=True, alpha=0.7, label="RL policy (pooled)")
    plt.axvline(np.nanmean(euler_bm), color="gray", linestyle="--", label="Mean EGM")

    # Per-seed KDE overlays (can fail for nearly constant samples)
    for i, arr in enumerate(euler_logabs_list):
        try:
            kde = gaussian_kde(arr[~np.isnan(arr)])
            xs = np.linspace(
                min(np.nanmin(rl_all), np.nanmin(euler_bm)),
                max(np.nanmax(rl_all), np.nanmax(euler_bm)),
                256,
            )
            plt.plot(xs, kde(xs), alpha=0.3, linewidth=1.0)
        except Exception as exc:
            logging.warning("Skipping KDE for seed %s due to: %s", seed_ids[i], exc)

    plt.title("Log10 Euler Equation Errors")
    plt.xlabel("log10(|error|)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "euler_errors_agg.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # WEV and CEV metrics
    # ---------------------------------------------------------------
    mean_wev, lo_wev, hi_wev = bootstrap_mean_ci(
        WEV_mat.mean(axis=1).reshape(-1, 1),
        n_boot=n_boot,
        alpha=alpha,
        seed=bootstrap_seed,
    )
    mean_rel_wev, lo_rel_wev, hi_rel_wev = bootstrap_mean_ci(
        np.vstack(WEV_rel_mat), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    mean_WEV, lo_WEV, hi_WEV = bootstrap_mean_ci(
        WEV_mat, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed + 2
    )
    mean_cev, lo_cev, hi_cev = bootstrap_mean_ci(
        CEV_mat.mean(axis=1).reshape(-1, 1),
        n_boot=n_boot,
        alpha=alpha,
        seed=bootstrap_seed,
    )
    mean_rel_cev, lo_rel_cev, hi_rel_cev = bootstrap_mean_ci(
        np.vstack(CEV_rel_mat), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    mean_CEV, lo_CEV, hi_CEV = bootstrap_mean_ci(
        CEV_mat, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed + 3
    )

    states_all = np.concatenate(pooled_states, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # WEV
    ax[0].plot(grid, mean_WEV, label="WEV (mean)")
    ax[0].fill_between(grid, lo_WEV, hi_WEV, alpha=0.25, label="95% CI (bootstrap)")
    ax[0].axhline(0, color="red", linestyle="--", label="Zero Line")
    ax[0].set_xlabel("Capital k")
    ax[0].set_ylabel("WEV")
    ax[0].set_title("Wealth-Equivalent Variation")
    ax0p = ax[0].twinx()
    ax0p.hist(states_all, bins=30, density=True, alpha=0.3, color="gray")
    ax0p.set_ylabel("State density")

    # CEV
    ax[1].plot(grid, mean_CEV, label="CEV (mean)")
    ax[1].fill_between(grid, lo_CEV, hi_CEV, alpha=0.25, label="95% CI (bootstrap)")
    ax[1].axhline(0, color="red", linestyle="--", label="Zero Line")
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

    # ---------------------------------------------------------------
    # Save metrics summary
    # ---------------------------------------------------------------

    # Helper to unpack (mean, lo, hi) triplets into flat metric entries
    pack_ci = lambda name, mean, lo, hi: {
        f"mean_{name}": float(mean[0]),
        f"lo_{name}": float(lo[0]),
        f"hi_{name}": float(hi[0]),
    }

    # Helper to unpack (mean, lo, hi) triplets into flat metric entries
    metrics = {
        "U_bm": float(U_bm),
        "max_abs_euler_error_bm": float(np.nanmax(euler_bm)),
        **pack_ci("U_rl", mean_Us, lo_Us, hi_Us),
        **pack_ci("wev", mean_wev, lo_wev, hi_wev),
        **pack_ci("wev_relevant", mean_rel_wev, lo_rel_wev, hi_rel_wev),
        **pack_ci("cev", mean_cev, lo_cev, hi_cev),
        **pack_ci("cev_relevant", mean_rel_cev, lo_rel_cev, hi_rel_cev),
        **pack_ci("max_abs_euler_error_rl", mean_max_euler, lo_max_euler, hi_max_euler),
    }

    # Write aggregated metrics summary to JSON file
    with (out_dir / "metrics_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)