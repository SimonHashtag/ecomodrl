import logging
from dataclasses import replace

import multiprocessing as mp
import torch as th

from ecomodrl.experiments.portfoliochoice.configs import RunConfig
from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.envs.portfoliochoice import PortfolioChoiceEnv
from ecomodrl.envs.parameters import PortfolioChoiceParams
#from ecomodrl.utils.analysis_BS import analyze_all_seeds

if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
    cfg = RunConfig()
    params = PortfolioChoiceParams()
    
    algo_cfg = AlgoConfig(
        name="ppo",
        policy="MlpPolicy",
        algo_params={
            "gamma": params.beta,
            "n_steps": cfg.max_episode_steps,
            "batch_size": cfg.n_envs * cfg.max_episode_steps // 5,
            "n_epochs": 2,
            "learning_rate": 3.5e-4,
            "ent_coef": 4.4e-5,
            "gae_lambda": 0.925,
            "clip_range": 0.27,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": {"pi": [32, 32], "vf": [32, 32]},
            "log_std_init": -2.0,  # smaller initial exploration to avoid extremes
        },
    )

    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    train_multiple_seeds(cfg, algo_cfg, PortfolioChoiceEnv, params, verbose=1)
    #analyze_all_seeds(cfg, algo_cfg, PortfolioChoiceEnv, params, n_boot=1000, alpha=0.05, bootstrap_seed=12345)