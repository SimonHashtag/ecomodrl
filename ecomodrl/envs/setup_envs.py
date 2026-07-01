from __future__ import annotations

import os
from typing import Callable, Any, Union, Mapping

from gymnasium.wrappers import TimeLimit, TransformReward
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
    # Avoid SubProcVecEnv when parallelizing across multiple seeds.
    vec_cls = DummyVecEnv if os.getenv("ECOMODRL_SEED_PARALLEL") == "1" else None
    
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
        ) # Seeding: VecEnv seeds each sub-env with seed + env_number
    beta_eff = float(envs.env_method("get_wrapper_attr", "beta_eff")[0])
    envs = VecNormalize(
        envs, norm_obs=cfg.normalize_obs, norm_reward=False, 
        clip_obs=10.0, clip_reward=1000.0, gamma=beta_eff,
    )

    return envs, beta_eff