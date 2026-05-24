import logging
import multiprocessing as mp
from dataclasses import replace

import torch as th

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

    # PPO Hyperparameters
    INITIAL_LR = 0.0003177782943
    N_EPISODES = 8
    N_EPOCHS = 10
    NUM_MINIBATCHES = 5
    CLIP_RANGE = 0.2833321070741
    ENT_COEF = 1.6759064e-06
    GAE_LAMBDA = 0.9616553953087
    VF_COEF = 0.967557240519

    lr_schedule = LinearScheduleByWallClockTime(
        initial_lr=INITIAL_LR,
        final_lr=3e-5,
        max_wallclock_seconds=cfg.max_wallclock_seconds,
    )

    algo_cfg = AlgoConfig(
        name="ppo",
        policy="MlpPolicy",
        algo_params={
            "gamma": params.beta,
            "n_steps": cfg.max_episode_steps * N_EPISODES,
            "batch_size": (cfg.n_envs * cfg.max_episode_steps * N_EPISODES) // NUM_MINIBATCHES,
            "n_epochs": N_EPOCHS,
            "learning_rate": lr_schedule,
            "ent_coef": ENT_COEF,
            "gae_lambda": GAE_LAMBDA,
            "clip_range": CLIP_RANGE,
            "vf_coef": VF_COEF,
            "max_grad_norm": 0.5,
        },
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "log_std_init": -2.0,
        },
    )

    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, BufferStockSavingsEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, BufferStockSavingsEnv, params)