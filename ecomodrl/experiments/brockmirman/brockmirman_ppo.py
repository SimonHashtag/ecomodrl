import logging
import multiprocessing as mp
from dataclasses import replace

import torch as th

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
    
    # PPO Hyperparameters
    INITIAL_LR = 0.0003904664776
    N_EPISODES = 4
    N_EPOCHS = 2
    NUM_MINIBATCHES = 4
    CLIP_RANGE = 0.2470436633579
    ENT_COEF = 0.0002568147945
    GAE_LAMBDA = 0.9083026725455
    VF_COEF = 0.8030369879929

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

    train_multiple_seeds(cfg, algo_cfg, BrockMirmanEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, BrockMirmanEnv, params)
