import logging
import multiprocessing as mp
from dataclasses import replace

from ecomodrl.configs import (
    BufferStockRunConfig, 
    parse_run_args, 
)
from ecomodrl.algorithms.hyperparams import ADAPTER_REGISTRY, get_algo_fixedcfgs
from ecomodrl.utils.hyperparam_import import load_tuned_hp
from ecomodrl.algorithms.training import train_multiple_seeds

# E.g., run with: 
# `python ecomodrl/experiments/bufferstock.py --algo_name ppo --hidden_nodes 64 --n_seeds 20`

if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    cfg = parse_run_args(BufferStockRunConfig())
    cfg = replace(cfg, out_dir=cfg.out_dir / cfg.env_name / cfg.algo_name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    params = cfg.make_params()
    env = cfg.env_class()

    hp = load_tuned_hp(env_name=cfg.env_name, algo_name=cfg.algo_name)

    base_algo_cfg = get_algo_fixedcfgs(cfg.algo_name, params, hidden_nodes=cfg.hidden_nodes)
    algo_cfg = ADAPTER_REGISTRY[cfg.algo_name](hp, cfg, base_algo_cfg)
    
    train_multiple_seeds(cfg, algo_cfg, env, params, verbose=1)