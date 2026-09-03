import os
import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Optional, Callable, Union, Mapping, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch as th

from stable_baselines3 import PPO, TD3, DDPG, SAC
from stable_baselines3.common.utils import set_random_seed

from ecomodrl.analysis.accuracy_metrics import make_evaluation_sample
from ecomodrl.envs.setup_envs import build_envs
from ecomodrl.algorithms.learning_schedule import WallClockCosineLR
from ecomodrl.algorithms.stopper import TrainingClock, WallClockBudgetCallback
from ecomodrl.algorithms.evaluation import BestValueEvalCallback
from ecomodrl.algorithms.grad_norm_wrapper import add_grad_norm_rescaling
from ecomodrl.algorithms.action_noise import make_action_noise

# -------------------------------------------------------
# Build deep RL algorithm
# -------------------------------------------------------

ALGO_REGISTRY = {
    "ppo":  PPO,
    "td3":  TD3,
    "ddpg": DDPG,
    "sac":  SAC,
}

@dataclass
class AlgoConfig:
    name: str                    # "ppo" | "td3" | "ddpg" | "sac"
    policy: str = "MlpPolicy"
    algo_params: Optional[dict] = None
    policy_kwargs: Optional[dict] = None


def _set_lr_schedule(
    algo_params: Dict[str, Any],
    cfg: Any,
    clock: TrainingClock | None = None,
) -> None:
    initial_lr = float(algo_params["learning_rate"])
    drop = float(algo_params.pop("drop", 0.0))

    algo_params["learning_rate"] = WallClockCosineLR(
        clock=clock,
        lr0=initial_lr,
        drop=drop,
        wallclock_seconds=cfg.max_wallclock_seconds,
    )


def build_model(
    algo_cfg: AlgoConfig,
    env,
    seed: int,
    tensorboard_log: Path,
    verbose: int = 0,
):
    AlgoCls = ALGO_REGISTRY[algo_cfg.name.lower()]
    algo_params = dict(algo_cfg.algo_params or {})
    policy_kwargs = dict(algo_cfg.policy_kwargs or {})

    algo_params.setdefault("verbose", verbose)
    algo_params.setdefault("seed", seed)
    algo_params.setdefault("tensorboard_log", str(tensorboard_log))
    
    # off-policy SB3 algos are incomptible with built-in grad norm scaling
    if algo_cfg.name.lower() in {"td3", "ddpg", "sac"}:
        algo_params.pop("max_grad_norm", None)

    action_noise = make_action_noise(env, algo_cfg.name.lower(), algo_params)
    if action_noise is not None:
        algo_params["action_noise"] = action_noise

    return AlgoCls(
        policy=algo_cfg.policy,
        env=env,
        **algo_params,
        policy_kwargs=policy_kwargs,
    )

# -------------------------------------------------------
# Multiprocessing utilities
# -------------------------------------------------------

def _set_thread_env(n: int = 1) -> None:
    """Limit per-process BLAS/OpenMP/Torch threads (defaults to 1)."""
    n = str(max(1, int(n)))
    os.environ.setdefault("OMP_NUM_THREADS", n)
    os.environ.setdefault("MKL_NUM_THREADS", n)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", n)
    try:
        th.set_num_threads(int(n))
        th.set_num_interop_threads(1)
    except Exception:
        pass

def _worker_run(args: Tuple[Any, AlgoConfig, Callable[..., Any], Any, int, int, str]) -> dict:
    """Top-level wrapper for pickling + per-process thread cap."""
    cfg, algo_cfg, env, params, seed, verbose, vec_env_type = args
    _set_thread_env(1)
    return train_single_seed(cfg, algo_cfg, env, params, seed, verbose=verbose, vec_env_type=vec_env_type)

# -------------------------------------------------------
# Main training functions
# -------------------------------------------------------

def train_single_seed(
    cfg: Union[Mapping[str, Any], Any],
    algo_cfg: AlgoConfig,
    env: Callable[..., Any],
    params: Union[Mapping[str, Any], Any],
    seed: int,
    verbose: int = 0,
    vec_env_type: str = "dummy",
) -> dict:
    """Train RL model for one seed and return metadata."""
    set_random_seed(seed)

    # Copy algo_cfg to avoid mutating the original
    algo_cfg = replace(
            algo_cfg,
            algo_params=dict(algo_cfg.algo_params or {}),
        )

    out_dir = cfg.out_dir / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / f"best_model_{algo_cfg.name.lower()}.zip"
    vecnorm_path = out_dir / "best_vecnormalize.pkl"

    envs = model = None
    clock = TrainingClock()
    try:
        envs, beta_eff = build_envs(env, params=params, cfg=cfg, seed=seed, reward_floor=cfg.reward_floor, vec_env_type=vec_env_type)
        algo_cfg.algo_params.update({"gamma": beta_eff}) # pass effective RL discount factor for Buffer Stock with normalization by prices

        eco_model = envs.get_attr("model")[0]
        evaluation_sample = make_evaluation_sample(
            eco_model,
            seed + cfg.eval_seed_offset,
            cfg.eval_m_grid_size,
            cfg.eval_horizon,
        )

        _set_lr_schedule(algo_cfg.algo_params, cfg, clock)
        
        
        model = build_model(algo_cfg, envs, seed, tensorboard_log=out_dir, verbose=0)
        
        max_grad_norm = algo_cfg.algo_params.get('max_grad_norm', None) if algo_cfg.algo_params else None
        
        if algo_cfg.name.lower() in {"td3", "ddpg", "sac"} and max_grad_norm is not None:
            add_grad_norm_rescaling(
                model, 
                max_grad_norm=max_grad_norm,
                clip_actor=True,
                clip_critic=True)

        budget_cb = WallClockBudgetCallback(clock=clock, seconds=cfg.max_wallclock_seconds)
        eval_cb = BestValueEvalCallback(
            cfg=cfg, algo_cfg=algo_cfg,
            clock=clock, eco_model=eco_model,
            evaluation_sample=evaluation_sample,
            out_dir=out_dir,
            seed=seed,
            verbose=verbose,
        )

        model.learn(total_timesteps=cfg.total_timesteps, callback=[budget_cb, eval_cb])

        return {
            "seed": seed,
            "best_eval_return": float(eval_cb.best_U),
            "last_eval_return": float(eval_cb.eval_returns[-1]) if eval_cb.eval_returns[-1] is not None else None,
            "best_model_path": str(best_model_path),
            "vecnorm_path": str(vecnorm_path),
            "cfg": asdict(cfg),
            "algo_cfg": asdict(algo_cfg),
            "params": vars(params) if hasattr(params, "__dict__") else dict(params),
        }
    finally:
        try:
            if envs is not None:
                envs.close()
        except Exception:
            pass
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            if th.cuda.is_available():
                th.cuda.empty_cache()
        except Exception:
            pass

def train_multiple_seeds(
    cfg: Union[Mapping[str, Any], Any],
    algo_cfg: AlgoConfig,
    env: Callable[..., Any],
    params: Union[Mapping[str, Any], Any],
    *,
    max_workers: Optional[int] = None,
    seed_parallel: bool = True,
    vec_env_type: str = "dummy",
    verbose: int = 0,
) -> List[dict]:
    """
    Parallelize across seeds. If max_workers is None, compute a safe value
    based on available CPUs.
    """
    seeds = list(cfg.seeds)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    if not seed_parallel:
        max_workers = 1

    if max_workers is None:
        total = max(1, os.cpu_count() - 1 or 1)
        max_workers = max(1, min(len(seeds), total))
        
    # If we parallelize over seeds, force DummyVecEnv inside each seed worker.
    if max_workers > 1:
        vec_env_type = "dummy"

    ctx = mp.get_context("spawn")
    tasks = [
        (cfg, algo_cfg, env, params, s, verbose, vec_env_type) 
        for s in seeds
    ]
    
    results: List[dict] = []

    if max_workers == 1:
        for t in tasks:
            results.append(_worker_run(t))
    else:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker_run, t): t[-1] for t in tasks}
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda d: d["seed"])
    with (cfg.out_dir / "train_cfgs.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    return results
