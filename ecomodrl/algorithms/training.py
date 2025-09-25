import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable, Union, Mapping, Any

import numpy as np
import torch as th
from stable_baselines3 import PPO, TD3, DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from ecomodrl.envs.setup_envs import build_envs
from ecomodrl.utils.stopper_sb import EarlyStoppingEvalCallback

ALGO_REGISTRY = {
    "ppo":  PPO,
    "td3":  TD3,
    "ddpg": DDPG,
    "sac":  SAC,
}

def _make_action_noise(env, algo_name: str, algo_params: dict):
    """For TD3/DDPG with continuous actions, build NormalActionNoise if requested."""
    if algo_name in {"td3", "ddpg"}:
        std = float(algo_params.pop("action_noise_std", 0.0))  # opt-in, default 0.0
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

    # common settings
    algo_params.setdefault("verbose", verbose)
    algo_params.setdefault("seed", seed)
    algo_params.setdefault("tensorboard_log", str(tensorboard_log))

    # attach noise if needed
    action_noise = _make_action_noise(env, algo_cfg.name.lower(), algo_params)
    if action_noise is not None:
        algo_params["action_noise"] = action_noise

    return AlgoCls(
        policy=algo_cfg.policy,
        env=env,
        **algo_params,
        policy_kwargs=policy_kwargs,
    )
    
def train_single_seed(
    cfg: Union[Mapping[str, Any], Any], 
    algo_cfg: AlgoConfig, 
    env: Callable[..., Any], 
    params: Union[Mapping[str, Any], Any], 
    seed: int
    ) -> dict:
    """
    Train reinforcement learning model for one seed and save artifacts (model, vecnorm, learning_curve, meta).
    """
    # global seeding (numpy/torch/python)
    set_random_seed(seed)
    
    # Output dirs
    out_dir = cfg.out_dir / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "best_model_ppo.zip"
    vecnorm_path = out_dir / "best_vecnormalize.pkl"

    train_env, eval_env = build_envs(
        env,
        params=params, 
        cfg=cfg,
        seed=seed,
        )
    
    model = build_model(
        algo_cfg, 
        train_env, 
        seed, 
        tensorboard_log=out_dir, 
        verbose=0
    )

    callback = EarlyStoppingEvalCallback(
        eval_env=eval_env,
        beta=params.beta,
        eval_freq=cfg.eval_every,
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        n_eval_episodes=1,            # increase for stochastic envs
        deterministic_eval=True,
        best_model_path=best_model_path,
        vecnorm_path=vecnorm_path,
        verbose=1,
    )

    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)
    
    train_cfg = {
        "seed": seed,
        "best_eval_return": float(callback.best_score),
        "num_timesteps": int(callback.num_timesteps),
        "best_model_path": str(best_model_path),
        "vecnorm_path": str(vecnorm_path),
        "cfg": asdict(cfg),
        "algo_cfg": asdict(algo_cfg),
        "params": vars(params)
    }
    return train_cfg

def train_multiple_seeds(
    cfg: Union[Mapping[str, Any], Any], 
    algo_cfg: AlgoConfig, 
    env: Callable[..., Any],
    params: Union[Mapping[str, Any], Any]):
    """ 
    Train multiple seeds and save summary JSON.
    """
    train_cfgs = [train_single_seed(cfg, algo_cfg, env, params, s) for s in cfg.seeds]
    with (cfg.out_dir / "train_cfgs.json").open("w", encoding="utf-8") as f:
        json.dump(train_cfgs, f, indent=2, default=str, ensure_ascii=False)