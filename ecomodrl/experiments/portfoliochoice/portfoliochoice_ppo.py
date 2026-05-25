import logging
from dataclasses import replace

import multiprocessing as mp
from ecomodrl.experiments.portfoliochoice.configs import RunConfig
from ecomodrl.algorithms.training import train_multiple_seeds
from ecomodrl.algorithms.hyperparams import (
    ALGO_REGISTRY,
    get_algo_fixedcfgs,
)
from ecomodrl.envs.portfoliochoice import PortfolioChoiceEnv
from ecomodrl.envs.parameters import PortfolioChoiceParams
from ecomodrl.utils.hyperparam_import import load_tuned_config
#from ecomodrl.utils.analysis_BS import analyze_all_seeds

if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
    cfg = RunConfig()
    params = PortfolioChoiceParams()
    
    logger = logging.getLogger(__name__)
    env_name = "PortfolioChoice"
    algo_key = "ppo"
    hp = load_tuned_config(env_name=env_name, algo_name=algo_key)

    base_cfg = get_algo_fixedcfgs(algo_key, params, hidden_nodes=128)
    algo_cfg = ALGO_REGISTRY[algo_key](hp, cfg, base_cfg)

    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    train_multiple_seeds(cfg, algo_cfg, PortfolioChoiceEnv, params, verbose=1)
    #analyze_all_seeds(cfg, algo_cfg, PortfolioChoiceEnv, params, n_boot=1000, alpha=0.05, bootstrap_seed=12345)