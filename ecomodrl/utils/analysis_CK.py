"""
Cass-Koopmans post-training analysis: extracts and saves raw per-seed metrics.

This module provides `analyze_all_seeds()` which extracts raw data from trained models
and saves to JSON. All aggregation (bootstrap CIs, benchmark comparison) is conducted
in a separate script `casskoopmans_compare.py` to avoid redundant computation across algorithms.

Output: `raw_metrics.json`.
"""
import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Union

import numpy as np
import torch as th
from scipy.interpolate import CubicSpline
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from ecomodrl.algorithms.training import ALGO_REGISTRY
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem
from ecomodrl.envs.setup_envs import build_envs
from ecomodrl.utils.tb_reader import tb_read_scalar

logger = logging.getLogger(__name__)


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
):
    """
    Extract raw per-seed metrics and save to JSON.
    
    This function extracts data needed for comparison plots but does NOT compute:
    - Benchmark solution (computed once in compare script)
    - Bootstrap CIs (computed in compare script)  
    - Derived metrics like CEV, Wasserstein distance (computed in compare script)
    
    Saves `raw_metrics.json` to cfg.out_dir with per-seed arrays only.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create grid for policy evaluation (same grid used by benchmark solver)
    planning_prob = PlanningProblem(
        gamma=params.gamma,
        beta=params.beta,
        delta=params.delta,
        alpha=params.alpha,
        A=params.A
    )
    grid = planning_prob.make_grid(points=1000)
    
    # Global step grid for learning curve interpolation
    global_steps = np.linspace(
        0, cfg.max_wallclock_seconds if hasattr(cfg, "max_wallclock_seconds") else cfg.total_timesteps,
        int(cfg.max_wallclock_seconds) + 1 if hasattr(cfg, "max_wallclock_seconds") else int(cfg.total_timesteps / cfg.eval_every) + 1
    )

    # --- Initialize containers ---
    return_curves = []
    rl_policy_vals = []
    rl_value_nn_vals = []
    euler_logabs_list = []
    pooled_states = []
    Us_rl = []
    value_from_policy_list = []
    compute_times = []
    seeds_processed = []

    # Load per-seed training metadata
    with (out_dir / "train_cfgs.json").open("r", encoding="utf-8") as f:
        train_cfgs = json.load(f)

    # --- Per-seed extraction ---
    for seed_entry in train_cfgs:
        best_model_path = Path(seed_entry["best_model_path"])
        vecnorm_path = Path(seed_entry["vecnorm_path"])
        seed = seed_entry["seed"]
        tb_path = out_dir / f"seed_{seed}"

        # Learning curve from TensorBoard
        try:
            _, returns = tb_read_scalar(tb_path, "eval/avg_discounted_return")
            _, wallclock = tb_read_scalar(tb_path, "eval/x_wallclock_seconds")
            interp_returns = np.interp(global_steps, wallclock, returns)
            final_time = wallclock[-1] if len(wallclock) > 0 else 0.0
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

        seeds_processed.append(seed)
        return_curves.append(interp_returns.tolist())
        compute_times.append(float(final_time))

        # Policy and value on grid
        rl_policy_vals.append(policy_rl(grid).tolist())
        if value_rl is not None:
            rl_value_nn_vals.append(value_rl(grid).tolist())

        # Euler errors and episode utilities
        euler_rl, states, U_rl = planning_prob.get_euler_errors(grid, policy_rl, episodes=100)
        euler_logabs_list.append(np.log10(np.abs(euler_rl)).tolist())
        pooled_states.append(states.tolist())
        Us_rl.append(float(U_rl))

        # Value function from policy (for CEV computation in compare script)
        V_rl_rec = planning_prob.V_from_a(grid, policy_rl, conv=1e-9)
        value_from_policy_list.append(V_rl_rec(grid).tolist())

    # --- Validation ---
    if not return_curves:
        logger.error("No valid seeds found for analysis.")
        return

    # --- Save raw metrics JSON ---
    raw_metrics = {
        "seeds": seeds_processed,
        "return_curves": return_curves,
        "compute_times": compute_times,
        "policies": rl_policy_vals,
        "values": rl_value_nn_vals,
        "euler_errors": euler_logabs_list,
        "states": pooled_states,
        "utilities": Us_rl,
        "value_from_policy": value_from_policy_list,
    }

    with (out_dir / "raw_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(raw_metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Raw metrics saved to {out_dir / 'raw_metrics.json'} ({len(seeds_processed)} seeds)")
