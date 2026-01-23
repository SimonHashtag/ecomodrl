
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

def make_action_noise(env, algo_name: str, algo_params: dict):
    """For TD3/DDPG build NormalActionNoise if specified in algo_params."""
    if algo_name in {"td3", "ddpg"}:
        std = float(algo_params.pop("action_noise_std", 0.0))
        if std > 0.0:
            action_dim = env.action_space.shape[0]
            return NormalActionNoise(mean=np.zeros(action_dim), sigma=std * np.ones(action_dim))
    return None