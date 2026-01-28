"""
Cass-Koopmans post-training analysis: computes and saves metrics for later comparison.

This module provides `analyze_all_seeds()` which aggregates results across training seeds
and saves metrics to JSON. No plots are generated here - use `casskoopmans_compare.py` 
for visualization.
"""
import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Union

import numpy as np
import torch as th
from scipy.interpolate import CubicSpline
from scipy.stats import wasserstein_distance
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from ecomodrl.algorithms.training import ALGO_REGISTRY
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem, PlanningSolvers
from ecomodrl.envs.setup_envs import build_envs
from ecomodrl.utils.metrics.accuracy import cev_crra
from ecomodrl.utils.metrics.performance import get_bootstrapped_iqm_ci
from ecomodrl.utils.tb_reader import tb_read_scalar

logger = logging.getLogger(__name__)


def compute_benchmark(params: Union[Mapping[str, Any], Any]):
    """Compute EGM benchmark solution for the Cass-Koopmans model."""
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
    cfg: Union[Mapping[str, Any], Any],
    env: Callable[..., Any],
    params: Union[Mapping[str, Any], Any],
    grid: np.ndarray,
    algo_name: str,
):
    """
    Load trained model and extract policy/value functions as CubicSpline interpolators.
    """
    device = get_device("auto")
    ALGO = ALGO_REGISTRY[algo_name.lower()]
    model: BaseAlgorithm = ALGO.load(model_path, device=device)

    # Rebuild VecNormalize with saved stats
    dummy_env, _ = build_envs(env, params=params, cfg=cfg, seed=1234)
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
    vecnorm.training = False

    # Prepare normalized observation grid (log-transformed)
    obs = grid.reshape(-1, 1).astype(np.float32, copy=False)
    obs_norm = vecnorm.normalize_obs(np.log(obs))

    # Extract policy function
    actions, _ = model.predict(obs_norm, deterministic=True)
    a_rl = np.asarray(actions).squeeze()
    policy_rl = CubicSpline(obs.squeeze(), a_rl, bc_type="natural", extrapolate=True)

    # Extract value function
    value_rl = None

    # PPO-like algorithms: predict_values() available
    if hasattr(getattr(model, "policy", None), "predict_values"):
        with th.no_grad():
            V_norm = (
                model.policy.predict_values(
                    th.as_tensor(obs_norm, dtype=th.float32, device=device)
                ).cpu().numpy().squeeze()
            )
        # Unnormalize if rewards were normalized during training
        if getattr(vecnorm, "norm_reward", False):
            ret_rms = getattr(vecnorm, "ret_rms", None)
            eps = float(getattr(vecnorm, "epsilon", 1e-8))
            scale = float(np.sqrt(getattr(ret_rms, "var", 1.0) + eps)) if ret_rms else 1.0
            V_nn = V_norm * scale
        else:
            V_nn = V_norm
        value_rl = CubicSpline(obs.squeeze(), V_nn, bc_type="natural", extrapolate=True)

    # Off-policy algorithms (SAC, TD3, DDPG): use critic Q-values
    else:
        policy = getattr(model, "policy", None)
        if policy is not None and hasattr(policy, "critic"):
            with th.no_grad():
                obs_t = th.as_tensor(obs_norm, dtype=th.float32, device=device)
                act_t = th.as_tensor(actions, dtype=th.float32, device=device).view(obs_t.shape[0], -1)
                q_out = policy.critic(obs_t, act_t)
                if isinstance(q_out, (list, tuple)):  # SAC: twin critics
                    qs = [q.flatten() for q in q_out]
                    q_min = th.min(th.stack(qs, dim=0), dim=0).values
                else:
                    q_min = q_out.flatten()
                Q_norm = q_min.cpu().numpy()

            # Unnormalize Q-values
            if getattr(vecnorm, "norm_reward", False):
                ret_rms = getattr(vecnorm, "ret_rms", None)
                eps = float(getattr(vecnorm, "epsilon", 1e-8))
                scale = float(np.sqrt(getattr(ret_rms, "var", 1.0) + eps)) if ret_rms else 1.0
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
    Aggregate metrics across training seeds and save to JSON.
    
    This function computes all metrics needed for comparison plots:
    - Learning curves (return over training steps)
    - Policy and value function evaluations on grid
    - Euler equation errors
    - Consumption-Equivalent Variation (CEV)
    - Wasserstein distance of error distributions
    
    Saves `metrics_analysis.json` to cfg.out_dir with aggregated statistics.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute benchmark solution
    planning_prob, grid, policy_bm, V_bm = compute_benchmark(params)
    euler_bm, _, U_bm = planning_prob.get_euler_errors(grid, policy_bm, episodes=100)
    euler_bm_log = np.log10(np.abs(euler_bm))

    # Benchmark return on training evaluation grid
    _, _, return_bm = planning_prob.get_euler_errors(
        np.linspace(cfg.state_min, cfg.state_max, 20),
        policy_bm,
        episodes=100,
    )

    # Global step grid for learning curve interpolation
    global_steps = np.linspace(0, cfg.max_processed_samples, int(cfg.max_processed_samples / cfg.eval_every) + 1)

    # --- Initialize containers ---
    return_curves = []
    rl_policy_vals = []
    rl_value_nn_vals = []
    euler_logabs_list = []
    pooled_states = []
    Us_rl = []
    CEV_mat = []
    CEV_rel_mat = []

    # Load per-seed training metadata
    with (out_dir / "train_cfgs.json").open("r", encoding="utf-8") as f:
        train_cfgs = json.load(f)

    # --- Per-seed analysis ---
    for seed_entry in train_cfgs:
        best_model_path = Path(seed_entry["best_model_path"])
        vecnorm_path = Path(seed_entry["vecnorm_path"])
        seed = seed_entry["seed"]
        tb_path = out_dir / f"seed_{seed}"

        # Learning curve from TensorBoard
        try:
            _, returns = tb_read_scalar(tb_path, "eval/avg_discounted_return")
            _, steps_proc = tb_read_scalar(tb_path, "eval/x_processed_samples")
            return_curves.append(np.interp(global_steps, steps_proc, returns))
        except Exception as e:
            logger.warning(f"Could not read TensorBoard for seed {seed}: {e}")
            continue

        # Load trained policy and value functions
        try:
            policy_rl, value_rl = get_policy_value_functions(
                best_model_path, vecnorm_path, cfg, env, params, grid, algo_cfg.name
            )
        except Exception as e:
            logger.error(f"Failed to load model for seed {seed}: {e}")
            continue

        # Policy and value on grid
        rl_policy_vals.append(policy_rl(grid))
        if value_rl is not None:
            rl_value_nn_vals.append(value_rl(grid))

        # Euler errors and episode utilities
        euler_rl, states, U_rl = planning_prob.get_euler_errors(grid, policy_rl, episodes=100)
        euler_logabs_list.append(np.log10(np.abs(euler_rl)))
        pooled_states.append(states)
        Us_rl.append(U_rl)

        # CEV metrics
        V_rl_rec = planning_prob.V_from_a(grid, policy_rl, conv=1e-9)
        cev_on_grid = cev_crra(grid, V_bm, V_rl_rec, gamma=params.gamma)
        CEV_mat.append(cev_on_grid)
        avg_cev_rel = cev_crra(states, V_bm, V_rl_rec, gamma=params.gamma).mean()
        CEV_rel_mat.append(avg_cev_rel)

    # --- Stack results ---
    if not return_curves:
        logger.error("No valid seeds found for analysis.")
        return

    return_curves = np.vstack(return_curves)
    rl_policy_vals = np.vstack(rl_policy_vals)
    rl_value_nn_vals = np.vstack(rl_value_nn_vals) if rl_value_nn_vals else np.array([])
    CEV_mat = np.vstack(CEV_mat)

    # --- Compute aggregated statistics with bootstrap CIs ---
    
    # Learning performance (utility)
    iqm_Us, lo_Us, hi_Us = get_bootstrapped_iqm_ci(
        np.array(Us_rl).reshape(-1, 1), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )

    # CEV (grid average per seed)
    iqm_cev, lo_cev, hi_cev = get_bootstrapped_iqm_ci(
        CEV_mat.mean(axis=1).reshape(-1, 1), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )

    # CEV (visited states)
    iqm_rel_cev, lo_rel_cev, hi_rel_cev = get_bootstrapped_iqm_ci(
        np.array(CEV_rel_mat).reshape(-1, 1), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )

    # Max Euler error
    max_euler_per_seed = np.max(euler_logabs_list, axis=1).reshape(-1, 1)
    iqm_max_euler, lo_max_euler, hi_max_euler = get_bootstrapped_iqm_ci(
        max_euler_per_seed, n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )

    # Wasserstein distance
    wd_list = [wasserstein_distance(euler_bm_log, euler_rl_seed) for euler_rl_seed in euler_logabs_list]
    iqm_wd, lo_wd, hi_wd = get_bootstrapped_iqm_ci(
        np.array(wd_list).reshape(-1, 1), n_boot=n_boot, alpha=alpha, seed=bootstrap_seed
    )

    # --- Save metrics JSON ---
    def pack_ci(name, iqm_val, lo, hi):
        return {
            f"iqm_{name}": float(iqm_val[0]) if hasattr(iqm_val, '__len__') else float(iqm_val),
            f"lo_{name}": float(lo[0]) if hasattr(lo, '__len__') else float(lo),
            f"hi_{name}": float(hi[0]) if hasattr(hi, '__len__') else float(hi),
        }

    metrics = {
        # Benchmark reference
        "U_bm": float(U_bm),
        "return_bm": float(return_bm),
        "max_abs_euler_error_bm": float(np.max(euler_bm_log)),
        # RL performance
        **pack_ci("U_rl", iqm_Us, lo_Us, hi_Us),
        **pack_ci("cev", iqm_cev, lo_cev, hi_cev),
        **pack_ci("cev_relevant", iqm_rel_cev, lo_rel_cev, hi_rel_cev),
        **pack_ci("max_abs_euler_error_rl", iqm_max_euler, lo_max_euler, hi_max_euler),
        **pack_ci("wasserstein_dist", iqm_wd, lo_wd, hi_wd),
        # Raw data for comparison plots (stored as lists for JSON serialization)
        "_grid": grid.tolist(),
        "_global_steps": global_steps.tolist(),
        "_return_curves": return_curves.tolist(),
        "_policies": rl_policy_vals.tolist(),
        "_values": rl_value_nn_vals.tolist() if len(rl_value_nn_vals) > 0 else [],
        "_euler_errors": [e.tolist() for e in euler_logabs_list],
        "_cev": CEV_mat.tolist(),
        "_states": [s.tolist() for s in pooled_states],
    }

    with (out_dir / "metrics_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Metrics saved to {out_dir / 'metrics_analysis.json'}")
