import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from ecomodrl.algorithms.training import ALGO_REGISTRY
from ecomodrl.benchmarks.brockmirman.problemclass import PlanningProblem, PlanningSolvers
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
    """
    Solve the Brock-Mirman planning problem via EGM to obtain benchmark policy/value.
    """

    # Set up planning problem
    model = PlanningProblem(
        gamma=params.gamma,
        beta=params.beta,
        delta=params.delta,
        alpha=params.alpha,
        rho=params.rho,
        sigma=params.sigma,
    )

    # Instantiate solvers class
    solvers = PlanningSolvers(k_points=400, z_points=21, conv=1e-5)

    # Solve model using EGM
    policy_bm = solvers.EGM(model)
    
    # Derive the policy-implied value function
    k_grid = model.make_k_grid(points=solvers.k_points)
    z_grid = model.make_z_grid(points=solvers.z_points)
    V_bm = model.V_from_a(k_grid, z_grid, policy_bm, conv=1e-3)

    # Return model class, optimal policy and corresponding value function
    return model, k_grid, z_grid, policy_bm, V_bm


def get_policy_value_functions(
    model_path: Path,
    vecnorm_path: Path,
    cfg: Union[Mapping[str, Any], Any],
    env: Callable[..., Any],
    params: Union[Mapping[str, Any], Any],
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    algo_name: str,
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    """
    Given a trained model and VecNormalize, return policy and value functions
    as RegularGridInterpolator objects on the (k, z) grid.
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
    K, Z = np.meshgrid(k_grid, z_grid, indexing="ij")
    # Env observation order is (z, k)
    state_grid = np.column_stack([Z.ravel(), K.ravel()])
    obs = state_grid.astype(np.float32, copy=False)
    obs_norm = vecnorm.normalize_obs(obs)

    # Policy function
    actions, _ = model.predict(obs_norm, deterministic=True)
    a_rl = np.asarray(actions, dtype=float).reshape(len(k_grid), len(z_grid))
    policy_rl = RegularGridInterpolator(
        (k_grid, z_grid),
        a_rl,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

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
        V = V_norm * scale
        value_rl = RegularGridInterpolator(
            (k_grid, z_grid),
            V.reshape(len(k_grid), len(z_grid)),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    # Off-policy algorithms (e.g., SAC, TD3, DDPG)
    else:
        with th.no_grad():
            policy_net = getattr(model, "policy", None)
            if policy_net is None or not hasattr(policy_net, "critic"):
                raise AttributeError("Algorithm does not expose a value function estimator.")

            obs_t = th.as_tensor(obs_norm, dtype=th.float32, device=device)
            act_t = th.as_tensor(actions, dtype=th.float32, device=device)
            q_out = policy_net.critic(obs_t, act_t)

            if isinstance(q_out, (list, tuple)): # SAC: twin critics
                qs = [q.flatten() for q in q_out]
                q_min = th.min(th.stack(qs, dim=0), dim=0).values
            else: # DDPG / TD3
                q_min = q_out.flatten()

            V_q = q_min.cpu().numpy()

        value_rl = RegularGridInterpolator(
            (k_grid, z_grid),
            V_q.reshape(len(k_grid), len(z_grid)),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

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
    # Grid for average return plot
    global_steps = np.linspace(0, cfg.max_processed_samples, retstep=cfg.eval_every)[0]

    # Benchmark once
    p, k_grid, z_grid, policy_bm, V_bm = compute_benchmark(params)
    euler_bm, _, U_bm = p.get_euler_errors(k_grid, z_grid, policy_bm, episodes=100)
    euler_bm = np.log10(np.abs(euler_bm))

    # Precompute mesh/grid helpers
    K, Z = np.meshgrid(k_grid, z_grid, indexing="ij")
    pts_kz = np.column_stack([K.ravel(), Z.ravel()])

    # --- Initialize containers for aggregation across seeds ---
    # Learning performance
    return_curves = []
    # Policy and value functions
    rl_policy_vals = []
    rl_value_nn_vals = []
    rl_value_from_policy = []
    # Euler errors and state visitation
    euler_logabs_list = []
    pooled_states_k = []
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

    # Determine beta for CEV calculation
    beta_for_cev = params.beta if getattr(params, "gamma", 2.0) == 1.0 else None

    # Policy/value/euler/wev/cev per seed
    for seed_entry in train_cfgs:
        best_model_path = Path(seed_entry["best_model_path"])
        vecnorm_path = Path(seed_entry["vecnorm_path"])
        s = seed_entry["seed"]

        tb_path = out_dir / f"seed_{s}"
        _, returns = tb_read_scalar(tb_path, "eval/avg_discounted_return")
        _, steps_proc = tb_read_scalar(tb_path, "eval/x_processed_samples")
        return_curves.append(np.interp(global_steps, steps_proc, returns))

        # RL policy/value functions on (k,z)
        policy_rl, value_rl = get_policy_value_functions(
            best_model_path,
            vecnorm_path,
            cfg,
            env,
            params,
            k_grid,
            z_grid,
            algo_cfg.name,
        )

        # Policy: convert share to consumption on grid
        a_grid = policy_rl(pts_kz).reshape(len(k_grid), len(z_grid))
        resources = p.cash_on_hand(K, Z)
        c_grid = a_grid * resources

        # Value from network
        V_nn_grid = value_rl(pts_kz).reshape(len(k_grid), len(z_grid))

        # Value recomputed from policy via Bellman iterations
        V_rl_interp = p.V_from_a(k_grid, z_grid, policy_rl, conv=1e-3)
        V_from_pol_grid = V_rl_interp(pts_kz).reshape(len(k_grid), len(z_grid))

        # Euler errors and visited capital states
        euler_rl, k_states_rl, U_rl = p.get_euler_errors(k_grid, z_grid, policy_rl, episodes=100, seed=s)
        euler_logabs_list.append(np.log10(np.abs(euler_rl)))
        pooled_states_k.append(k_states_rl)
        Us_rl.append(U_rl)

        wev_grid = wev(pts_kz, V_bm, V_rl_interp, p.cash_on_hand).reshape(len(k_grid), len(z_grid))
        CEV_grid = cev_crra(
            pts_kz,
            V_bm,
            V_rl_interp,
            gamma=params.gamma,
            beta=beta_for_cev,
        ).reshape(len(k_grid), len(z_grid))

        WEV_mat.append(wev_grid)
        CEV_mat.append(CEV_grid)

        # Approximate relevant WEV/CEV by pairing visited k with z=1.0 (stationary mean)
        k_vis = np.asarray(k_states_rl).ravel()
        z_vis = np.ones_like(k_vis)
        states_vis = np.column_stack([k_vis, z_vis])
        
        # Sample 1000 points from visited states for relevance approximation
        rng = np.random.default_rng(seed=s)
        idx_sample = rng.choice(states_vis.shape[0], size=1000, replace=False)
        states_vis = states_vis[idx_sample, :]
        
        wev_rel = np.mean(wev(states_vis, V_bm, V_rl_interp, p.cash_on_hand))
        cev_rel = np.mean(
            cev_crra(states_vis, V_bm, V_rl_interp, gamma=params.gamma,
                     beta=beta_for_cev)
        )
        WEV_rel_mat.append(float(wev_rel))
        CEV_rel_mat.append(float(cev_rel))

        # Stash grids for aggregation
        rl_policy_vals.append(c_grid)
        if V_nn_grid is not None:
            rl_value_nn_vals.append(V_nn_grid)
        rl_value_from_policy.append(V_from_pol_grid)

    # Stack arrays across seeds
    return_curves = np.vstack(return_curves)
    rl_policy_vals = np.stack(rl_policy_vals)
    rl_value_from_policy = np.stack(rl_value_from_policy)
    rl_value_nn_vals = np.stack(rl_value_nn_vals) if len(rl_value_nn_vals) > 0 else None
    WEV_mat = np.stack(WEV_mat)
    CEV_mat = np.stack(CEV_mat)
    euler_logabs_arr = np.vstack(euler_logabs_list)
    pooled_states_k_arr = np.hstack(pooled_states_k)
    Us_rl = np.asarray(Us_rl).reshape(-1, 1)
    WEV_rel_mat_arr = np.asarray(WEV_rel_mat).reshape(-1, 1)
    CEV_rel_mat_arr = np.asarray(CEV_rel_mat).reshape(-1, 1)

    # ---------------------------------------------------------------
    # Learning curve aggregation
    # ---------------------------------------------------------------
    mean_Us, lo_Us, hi_Us = bootstrap_mean_ci(
        Us_rl, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )
    iqm_return = interquartile_mean(return_curves)
    q25_return, q75_return = interquartile_range(return_curves)

    # Benchmark evaluation return: start from z=1 across eval grid of k
    k_eval = np.linspace(cfg.state_min, cfg.state_max, 20)
    z_eval = np.array([1.0])
    _, _, return_bm = p.get_euler_errors(k_eval, z_eval, policy_bm, episodes=100)

    fig = plt.figure(figsize=(9, 4.5))
    plt.fill_between(global_steps, q25_return, q75_return, alpha=0.3, label="IQR")
    plt.plot(global_steps, iqm_return, label="Interquartile Mean")
    plt.axhline(return_bm, linestyle="-.", color="darkorange", label="Benchmark")
    plt.xlabel("Processed Samples")
    plt.ylabel("Avg discounted episode return (true scale)")
    plt.title("Learning Curve (PPO on Brock–Mirman)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "learning_curve_agg.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Policy and value function slices at z = {min, median, max}
    # ---------------------------------------------------------------
    mean_pol_flat, lo_pol_flat, hi_pol_flat = bootstrap_mean_ci(
        rl_policy_vals.reshape(rl_policy_vals.shape[0], -1),
        n_boot=n_boot,
        alpha=alpha,
        seed=bootstrap_seed,
    )
    mean_pol = mean_pol_flat.reshape(len(k_grid), len(z_grid))
    lo_pol = lo_pol_flat.reshape(len(k_grid), len(z_grid))
    hi_pol = hi_pol_flat.reshape(len(k_grid), len(z_grid))

    if rl_value_nn_vals is not None:
        mean_Vnn_flat, lo_Vnn_flat, hi_Vnn_flat = bootstrap_mean_ci(
            rl_value_nn_vals.reshape(rl_value_nn_vals.shape[0], -1),
            n_boot=n_boot,
            alpha=alpha,
            seed=bootstrap_seed + 1,
        )
        mean_Vnn = mean_Vnn_flat.reshape(len(k_grid), len(z_grid))
        lo_Vnn = lo_Vnn_flat.reshape(len(k_grid), len(z_grid))
        hi_Vnn = hi_Vnn_flat.reshape(len(k_grid), len(z_grid))
    else:
        mean_Vnn = lo_Vnn = hi_Vnn = None

    mean_Vfrom = rl_value_from_policy.mean(axis=0)
    V_bm_grid = V_bm(pts_kz).reshape(len(k_grid), len(z_grid))

    # Indices for z slices
    z_indices = [0, len(z_grid) // 2, len(z_grid) - 1]
    ncols = len(z_indices)
    fig, axes = plt.subplots(2, ncols, figsize=(3.8 * ncols, 6.2), sharex=True)
    cmap = plt.cm.get_cmap("viridis", ncols)

    for j, idx in enumerate(z_indices):
        color = cmap(j)
        z_val = z_grid[idx]

        # Policy: consumption c(k,z)
        ax_pol = axes[0, j]
        ax_pol.plot(k_grid, mean_pol[:, idx], color=color, linewidth=2.0, label="RL mean")
        ax_pol.fill_between(k_grid, lo_pol[:, idx], hi_pol[:, idx], color=color, alpha=0.2)
        # Benchmark consumption from benchmark shares
        a_bm_slice = policy_bm(np.column_stack([k_grid, np.full_like(k_grid, z_val)]))
        c_bm_slice = np.clip(a_bm_slice, 1e-8, 1.0 - 1e-8) * np.maximum(p.cash_on_hand(k_grid, z_val), 1e-8)
        ax_pol.plot(k_grid, c_bm_slice, linestyle="--", color="black", alpha=0.9, label="EGM")
        ax_pol.set_title(f"z = {z_val:.3f}")
        if j == 0:
            ax_pol.set_ylabel("Consumption")
        ax_pol.legend(fontsize=8)

        # Value function
        ax_val = axes[1, j]
        if mean_Vnn is not None:
            ax_val.plot(k_grid, mean_Vnn[:, idx], color=color, linewidth=2.0, label="RL mean (NN)")
            ax_val.fill_between(k_grid, lo_Vnn[:, idx], hi_Vnn[:, idx], color=color, alpha=0.2)
            ax_val.plot(k_grid, mean_Vfrom[:, idx], linestyle=":", color=color, alpha=0.9, label="RL value (from policy)")
        else:
            ax_val.plot(k_grid, mean_Vfrom[:, idx], linestyle=":", color=color, linewidth=2.0, label="RL value (from policy)")
        ax_val.plot(k_grid, V_bm_grid[:, idx], linestyle="--", color="black", alpha=0.9, label="EGM")
        if j == 0:
            ax_val.set_ylabel("Value")
        ax_val.set_xlabel("Capital k")
        ax_val.legend(fontsize=8)

    axes[0, 0].set_title("Policy slices (per z)")
    axes[1, 0].set_title("Value slices (per z)")

    plt.tight_layout()
    fig.savefig(out_dir / "policy_value_agg.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # WEV and CEV plots with bootstrap CIs; pooled state histogram
    # ---------------------------------------------------------------
    mean_WEV_flat, lo_WEV_flat, hi_WEV_flat = bootstrap_mean_ci(
        WEV_mat.reshape(WEV_mat.shape[0], -1),
        n_boot=n_boot,
        alpha=alpha,
        seed=bootstrap_seed + 2,
    )
    mean_WEV = mean_WEV_flat.reshape(len(k_grid), len(z_grid))
    lo_WEV = lo_WEV_flat.reshape(len(k_grid), len(z_grid))
    hi_WEV = hi_WEV_flat.reshape(len(k_grid), len(z_grid))

    mean_CEV_flat, lo_CEV_flat, hi_CEV_flat = bootstrap_mean_ci(
        CEV_mat.reshape(CEV_mat.shape[0], -1),
        n_boot=n_boot,
        alpha=alpha,
        seed=bootstrap_seed + 3,
    )
    mean_CEV = mean_CEV_flat.reshape(len(k_grid), len(z_grid))
    lo_CEV = lo_CEV_flat.reshape(len(k_grid), len(z_grid))
    hi_CEV = hi_CEV_flat.reshape(len(k_grid), len(z_grid))

    # Scalars (averages over grid and relevance approximation)
    mean_WEV_scalar, lo_WEV_scalar, hi_WEV_scalar = bootstrap_mean_ci(
        WEV_mat.reshape(WEV_mat.shape[0], -1).mean(axis=1).reshape(-1, 1),
        n_boot=n_boot,
        alpha=alpha,
        seed=bootstrap_seed,
    )
    mean_rel_wev, lo_rel_wev, hi_rel_wev = bootstrap_mean_ci(
        WEV_rel_mat_arr,
        n_boot=n_boot,
        alpha=alpha,
        seed=bootstrap_seed,
    )
    mean_CEV_scalar, lo_CEV_scalar, hi_CEV_scalar = bootstrap_mean_ci(
        CEV_mat.reshape(CEV_mat.shape[0], -1).mean(axis=1).reshape(-1, 1),
        n_boot=n_boot,
        alpha=alpha,
        seed=bootstrap_seed,
    )
    mean_rel_cev, lo_rel_cev, hi_rel_cev = bootstrap_mean_ci(
        CEV_rel_mat_arr,
        n_boot=n_boot,
        alpha=alpha,
        seed=bootstrap_seed,
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].axhline(0.0, linestyle="--", color="black", linewidth=0.8, alpha=0.6, label="Zero")
    ax[1].axhline(0.0, linestyle="--", color="black", linewidth=0.8, alpha=0.6, label="Zero")
    for idx in z_indices:
        z_val = z_grid[idx]
        line_w, = ax[0].plot(k_grid, mean_WEV[:, idx], label=f"RL mean (z={z_val:.3f})")
        ax[0].fill_between(k_grid, lo_WEV[:, idx], hi_WEV[:, idx], color=line_w.get_color(), alpha=0.2)
        line_c, = ax[1].plot(k_grid, mean_CEV[:, idx], label=f"RL mean (z={z_val:.3f})")
        ax[1].fill_between(k_grid, lo_CEV[:, idx], hi_CEV[:, idx], color=line_c.get_color(), alpha=0.2)

    ax[0].set_ylim(bottom=min(0.0, ax[0].get_ylim()[0]))
    ax[0].set_xlabel("Capital k")
    ax[0].set_ylabel("WEV")
    ax[0].set_title("Wealth-Equivalent Variation")
    ax0p = ax[0].twinx()
    ax0p.hist(pooled_states_k_arr, bins=30, density=True, alpha=0.3, color="gray")
    ax0p.set_ylabel("State density (k)")

    ax[1].set_ylim(bottom=min(0.0, ax[1].get_ylim()[0]))
    ax[1].set_xlabel("Capital k")
    ax[1].set_ylabel("CEV")
    ax[1].set_title("Consumption-Equivalent Variation")
    ax1p = ax[1].twinx()
    ax1p.hist(pooled_states_k_arr, bins=30, density=True, alpha=0.3, color="gray")
    ax1p.set_ylabel("State density (k)")

    for axis in ax:
        axis.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "wev_cev_agg.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Euler error histogram (pooled) + per-seed KDEs
    # ---------------------------------------------------------------
    mean_max_euler, lo_max_euler, hi_max_euler = bootstrap_mean_ci(
        np.max(euler_logabs_arr, axis=1).reshape(-1, 1),
        n_boot=n_boot,
        alpha=alpha,
        seed=bootstrap_seed,
    )

    rl_all = euler_logabs_arr.ravel()
    bm_all = euler_bm.ravel()
    bins = np.linspace(
        min(rl_all.min(), bm_all.min()),
        max(rl_all.max(), bm_all.max()),
        60,
    )

    fig = plt.figure(figsize=(7.2, 4.5))
    plt.hist(rl_all, bins=bins, density=True, alpha=0.7, label="RL policy (pooled)")
    plt.hist(bm_all, bins=bins, density=True, alpha=0.5, label="EGM benchmark")

    for i, arr in enumerate(euler_logabs_list):
        try:
            kde = gaussian_kde(arr)
            xs = np.linspace(bins[0], bins[-1], 256)
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
    # Metrics summary
    # ---------------------------------------------------------------
    pack_ci = lambda name, mean, lo, hi: {
        f"mean_{name}": float(mean[0]),
        f"lo_{name}": float(lo[0]),
        f"hi_{name}": float(hi[0]),
    }

    metrics = {
        "U_bm": float(U_bm),
        "max_abs_euler_error_bm": float(np.max(bm_all)),
        **pack_ci("U_rl", mean_Us, lo_Us, hi_Us),
        **pack_ci("wev", mean_WEV_scalar, lo_WEV_scalar, hi_WEV_scalar),
        **pack_ci("wev_relevant", mean_rel_wev, lo_rel_wev, hi_rel_wev),
        **pack_ci("cev", mean_CEV_scalar, lo_CEV_scalar, hi_CEV_scalar),
        **pack_ci("cev_relevant", mean_rel_cev, lo_rel_cev, hi_rel_cev),
        **pack_ci("max_abs_euler_error_rl", mean_max_euler, lo_max_euler, hi_max_euler),
    }

    with (out_dir / "metrics_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
