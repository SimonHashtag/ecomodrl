import logging
import multiprocessing as mp
from dataclasses import replace

import torch as th
import math

from ecomodrl.experiments.casskoopmans.configs import RunConfig
from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.envs.parameters import CassKoopmansParams
from ecomodrl.utils.analysis_CK import analyze_all_seeds
from ecomodrl.utils.learning_schedule import LinearScheduleByWallClockTime


if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    cfg = RunConfig()
    params = CassKoopmansParams()
    
    # DDPG Hyperparameters
    INITIAL_LR = 0.0029973147496
    N_EPISODES = 2
    BATCH_SIZE = 64
    UTD_RATIO = 3.9858314745995
    TAU = 0.0120665108631
    ACTION_NOISE_STD = 0.0992773977939 # exploration_sigma

    lr_schedule = LinearScheduleByWallClockTime(
        initial_lr=INITIAL_LR,
        final_lr=3e-5,
        max_wallclock_seconds=cfg.max_wallclock_seconds,
    )

    train_freq = cfg.max_episode_steps * N_EPISODES
    gradient_steps = math.ceil(cfg.n_envs * train_freq * UTD_RATIO / BATCH_SIZE)

    algo_cfg = AlgoConfig(
        name="ddpg",
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
            "action_noise_std": ACTION_NOISE_STD,
        },
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": {"pi": [64, 64], "qf": [64, 64]},
        },
    )
    
    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, CassKoopmansEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, CassKoopmansEnv, params)