import logging
import multiprocessing as mp
from dataclasses import replace

import torch as th
import math

from ecomodrl.experiments.brockmirman.configs import RunConfig
from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.envs.brockmirman import BrockMirmanEnv
from ecomodrl.utils.analysis_BM import analyze_all_seeds
from ecomodrl.envs.parameters import BrockMirmanParams
from ecomodrl.utils.learning_schedule import LinearScheduleByWallClockTime


if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    cfg = RunConfig()
    params = BrockMirmanParams()
    
    # TD3 Hyperparameters
    INITIAL_LR = 0.0009405396428
    N_EPISODES = 4
    BATCH_SIZE = 64
    UTD_RATIO = 3.9217385608281
    TAU = 0.0184336640767
    ACTION_NOISE_STD = 0.0585842514503  # exploration_sigma
    TARGET_POLICY_NOISE = 0.1018948072371
    # max(hp["target_noise_clip"], hp["target_policy_noise"])
    # max(0.1115663097238, 0.1018948072371) = 0.1115663097238
    TARGET_NOISE_CLIP = 0.1115663097238

    lr_schedule = LinearScheduleByWallClockTime(
        initial_lr=INITIAL_LR,
        final_lr=3e-5,
        max_wallclock_seconds=cfg.max_wallclock_seconds,
    )

    train_freq = cfg.max_episode_steps * N_EPISODES
    gradient_steps = math.ceil(cfg.n_envs * train_freq * UTD_RATIO / BATCH_SIZE)

    algo_cfg = AlgoConfig(
        name="td3",
        policy="MlpPolicy",
        algo_params={
            "learning_rate": lr_schedule,
            "buffer_size": 200_000,
            "batch_size": BATCH_SIZE,
            "tau": TAU,
            "gamma": params.beta,
            "max_grad_norm": 0.5,
            "train_freq": (train_freq, "step"),
            "gradient_steps": gradient_steps,
            "learning_starts": 10_000,
            "policy_delay": 2,
            "action_noise_std": ACTION_NOISE_STD,
            "target_policy_noise": TARGET_POLICY_NOISE,
            "target_noise_clip": TARGET_NOISE_CLIP,
        },
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": {"pi": [128, 128], "qf": [128, 128]},
        },
    )
    
    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, BrockMirmanEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, BrockMirmanEnv, params)
