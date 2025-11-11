import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable, Union, Mapping, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch as th
from stable_baselines3 import PPO, TD3, DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from ecomodrl.envs.setup_envs import build_envs
from ecomodrl.utils.stopper_sb import EarlyStoppingEvalCallback

# -------------------------------------------------------
# Build deep RL algorithm
# -------------------------------------------------------

ALGO_REGISTRY = {
    "ppo":  PPO,
    "td3":  TD3,
    "ddpg": DDPG,
    "sac":  SAC,
}

def _make_action_noise(env, algo_name: str, algo_params: dict):
    """For TD3/DDPG with continuous actions, build NormalActionNoise if requested."""
    if algo_name in {"td3", "ddpg"}:
        std = float(algo_params.pop("action_noise_std", 0.0))
        if std > 0.0:
            action_dim = env.action_space.shape[0]
            return NormalActionNoise(mean=np.zeros(action_dim), sigma=std * np.ones(action_dim))
    return None

@dataclass
class AlgoConfig:
    name: str                    # "ppo" | "td3" | "ddpg" | "sac"
    policy: str = "MlpPolicy"
    algo_params: Optional[dict] = None
    policy_kwargs: Optional[dict] = None

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

    action_noise = _make_action_noise(env, algo_cfg.name.lower(), algo_params)
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

def _worker_run(args: Tuple[Any, AlgoConfig, Callable[..., Any], Any, int]) -> dict:
    """Top-level wrapper for pickling + per-process thread cap."""
    cfg, algo_cfg, env, params, seed = args
    _set_thread_env(1)
    return train_single_seed(cfg, algo_cfg, env, params, seed)

# -------------------------------------------------------
# Main training functions
# -------------------------------------------------------

def train_single_seed(
    cfg: Union[Mapping[str, Any], Any],
    algo_cfg: AlgoConfig,
    env: Callable[..., Any],
    params: Union[Mapping[str, Any], Any],
    seed: int
) -> dict:
    """Train RL model for one seed and return metadata."""
    set_random_seed(seed)

    out_dir = cfg.out_dir / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "best_model_ppo.zip"
    vecnorm_path = out_dir / "best_vecnormalize.pkl"

    train_env = eval_env = model = None
    try:
        train_env, eval_env = build_envs(env, params=params, cfg=cfg, seed=seed)

        model = build_model(algo_cfg, train_env, seed, tensorboard_log=out_dir, verbose=0)

        callback = EarlyStoppingEvalCallback(
            eval_env=eval_env,
            beta=params.beta,
            eval_freq=cfg.eval_every,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            n_eval_episodes=getattr(cfg, "n_eval_episodes", None) or 1.0,    # increase for stochastic envs
            deterministic_eval=True,
            best_model_path=best_model_path,
            vecnorm_path=vecnorm_path,
            verbose=1,
        )

        model.learn(total_timesteps=cfg.total_timesteps, callback=callback)

        return {
            "seed": seed,
            "best_eval_return": float(callback.best_score),
            "num_timesteps": int(callback.num_timesteps),
            "best_model_path": str(best_model_path),
            "vecnorm_path": str(vecnorm_path),
            "cfg": asdict(cfg),
            "algo_cfg": asdict(algo_cfg),
            "params": vars(params) if hasattr(params, "__dict__") else dict(params),
        }
    finally:
        try:
            if train_env is not None:
                train_env.close()
        except Exception:
            pass
        try:
            if eval_env is not None:
                eval_env.close()
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

    ctx = mp.get_context("spawn")
    tasks = [(cfg, algo_cfg, env, params, s) for s in seeds]
    results: List[dict] = []

    if max_workers == 1:
        for t in tasks:
            results.append(_worker_run(t))
    else:
        os.environ["ECOMODRL_SEED_PARALLEL"] = "1" # tell env builder to avoid SubprocVecEnv in seed workers
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker_run, t): t[-1] for t in tasks}
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda d: d["seed"])
    with (cfg.out_dir / "train_cfgs.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    return results
