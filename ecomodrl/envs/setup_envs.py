from __future__ import annotations

import logging
from typing import Callable, Any, Union, Mapping

import numpy as np
from gymnasium.wrappers import TimeLimit, TransformReward
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env(
    env_class: Callable[..., Any],
    params: Any,
    *,
    max_episode_steps: int,
    fixed_start: bool,
    fixed_state: float,
    state_min: float,
    state_max: float,
    floattype=np.float32,
    training: bool,
    reward_floor: float | None = 100.0,
    ) -> Callable[..., Any]:
    """
    Factory that returns a thunk creating a configured Environment of type env_class.
    The custom gym environment env_class must have the following arguments:
    
    params: Model-specific parameters
    max_episode_steps: Number of periods until truncation of episode
    fixed_start: whether state in first period is fixed or drawn from uniform distribution in [state_min, state_max]
    fixed_state: If fixed_start=True: state in first period
    state_min: Lower bound of start state grid
    state_max: Upper bound of start state grid
    floattype: Type of float precision
    training: If true and reward_floor is not None, then reward is clipped from below to stabilize training
    """
    def _thunk():
        env = env_class(
            model_params=params,
            fixed_start=fixed_start,
            reset_state=fixed_state,
            state_min=state_min,
            state_max=state_max,
            floattype=floattype,
        )
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        # stabilize: avoid large negative rewards during training
        if training and (reward_floor is not None):
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
    """
    train_env = make_vec_env(
        make_env(
            env_class,
            params,
            max_episode_steps=cfg.max_eval_episode_steps,
            fixed_start=False,
            fixed_state=float(np.nan),  # ignored when fixed_start=False
            state_min=cfg.state_min,
            state_max=cfg.state_max,
            training=True,
            reward_floor=reward_floor,
        ), 
        n_envs=cfg.n_envs, 
        seed=seed
        ) # Seeding: VecEnv seeds each sub-env with seed + env_number
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, 
        clip_obs=10.0, clip_reward=10.0, gamma=params.beta,
    )

    # Eval env: fixed-start grid, deterministic eval
    eval_grid = np.linspace(cfg.state_min, 
                            cfg.state_max, 
                            num=20, 
                            dtype=np.float32)
    eval_env_thunks = [
        make_env(
            env_class,
            params,
            max_episode_steps=cfg.max_eval_episode_steps,
            fixed_start=True,
            fixed_state=float(state0),
            state_min=cfg.state_min,
            state_max=cfg.state_max,
            training=False,
            reward_floor=None,
        ) 
        for state0 in eval_grid
        ]
    eval_env_raw = DummyVecEnv(
        eval_env_thunks,
    )
    try:
        eval_env_raw.seed(seed)
        eval_env_raw.action_space.seed(seed)
        eval_env_raw.observation_space.seed(seed)
    except Exception as exc:
        logging.warning("Eval env seeding failed (non-fatal): %s", exc)
    eval_env = VecNormalize(
        eval_env_raw, training=False, norm_obs=True, norm_reward=False, 
        clip_obs=10.0, clip_reward=1000.0, gamma=params.beta,
    )
    return train_env, eval_env