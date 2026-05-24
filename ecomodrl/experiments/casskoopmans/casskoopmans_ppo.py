import logging
import multiprocessing as mp
from dataclasses import replace

import torch as th

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
    
    # PPO Hyperparameters
    INITIAL_LR = 0.0004322994492
    N_EPISODES = 2
    N_EPOCHS = 7
    NUM_MINIBATCHES = 4
    CLIP_RANGE = 0.0545356382642
    ENT_COEF = 2.2816594e-06
    GAE_LAMBDA = 0.9515547720523
    VF_COEF = 0.8745344247954

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
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "log_std_init": -2.0,
        },
    )
    
    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, CassKoopmansEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, CassKoopmansEnv, params)