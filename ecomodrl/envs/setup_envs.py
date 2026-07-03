from __future__ import annotations

import os
from typing import Callable, Any, Union, Mapping

from gymnasium.wrappers import TimeLimit, TransformReward
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

def make_env(
    env_class: Callable[..., Any],
    params: Any,
    *,
    max_episode_steps: int,
    reward_floor: float | None = 100.0,
    ) -> Callable[..., Any]:
    """
    Factory that returns a thunk creating a configured Environment of type env_class.
    The custom gym environment env_class must have the following arguments:
    
    params: Model-specific parameters
    """
    def _thunk():
        env = env_class(params=params,)
        
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        # stabilize: avoid large negative rewards during training
        if reward_floor is not None:
            env = TransformReward(env, lambda r: max(r, -float(reward_floor)))  
        return env
    return _thunk

def build_envs(
    env_class: Callable[..., Any],
    params: Any,
    cfg: Union[Mapping[str, Any], Any],
    seed: int,
    *,
    reward_floor: float | None = 100,
    vec_env_type: str = "dummy",
    ):
    """ 
    Creates vectorized training env and eval env with normalization and explicit seeding.
    
    env_class: Gym environment
    params: model-specific parameters,
    cfg: Run configs
    seed: seed
    reward_floor: In training mode, reward is clipped from below (at -reward_floor) to stabilize training
    
    Returns: vectorized training env, effective RL discount (beta_eff)
    """
    
    if vec_env_type == "dummy":
        vec_cls = DummyVecEnv
        vec_env_kwargs = None
    elif vec_env_type == "subproc":
        vec_cls = SubprocVecEnv
        vec_env_kwargs = {"start_method": "spawn"}
    else:
        raise ValueError(f"Unknown vec_env_type={vec_env_type!r}")

    # Avoid Gymnasium wrapper attribute warning entirely.
    model = env_class.MODEL_CLS(params=params)
    beta_eff = float(model.beta_eff)

    envs = make_vec_env(
        make_env(
            env_class,
            params,
            max_episode_steps=cfg.max_episode_steps,
            reward_floor=reward_floor,
        ),
        n_envs=cfg.n_envs,
        seed=seed,
        vec_env_cls=vec_cls,
        vec_env_kwargs=vec_env_kwargs,
    )

    envs = VecNormalize(
        envs,
        norm_obs=cfg.normalize_obs,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=reward_floor,
        gamma=beta_eff,
    )

    return envs, beta_eff