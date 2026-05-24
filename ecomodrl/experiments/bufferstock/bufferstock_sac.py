import logging
import multiprocessing as mp
from dataclasses import replace

import torch as th
import math

from ecomodrl.experiments.bufferstock.configs import RunConfig
from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.envs.bufferstock import BufferStockSavingsEnv
from ecomodrl.utils.analysis_BS import analyze_all_seeds
from ecomodrl.envs.parameters import BufferStockParams
from ecomodrl.utils.learning_schedule import LinearScheduleByWallClockTime


if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    cfg = RunConfig()
    params = BufferStockParams()
    
    # SAC Hyperparameters
    INITIAL_LR = 0.0018227279408
    N_EPISODES = 4
    BATCH_SIZE = 32
    UTD_RATIO = 3.6038083197985
    TAU = 0.0125005329741

    lr_schedule = LinearScheduleByWallClockTime(
        initial_lr=INITIAL_LR,
        final_lr=3e-5,
        max_wallclock_seconds=cfg.max_wallclock_seconds,
    )

    train_freq = cfg.max_episode_steps * N_EPISODES
    gradient_steps = math.ceil(cfg.n_envs * train_freq * UTD_RATIO / BATCH_SIZE)

    algo_cfg = AlgoConfig(
        name="sac",
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
            "ent_coef": "auto",
        },
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": {"pi": [128, 128], "qf": [128, 128]},
        },
    )
    
    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, BufferStockSavingsEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, BufferStockSavingsEnv, params)
