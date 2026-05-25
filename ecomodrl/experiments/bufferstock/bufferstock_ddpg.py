import logging
import multiprocessing as mp
from dataclasses import replace

from ecomodrl.experiments.bufferstock.configs import RunConfig
from ecomodrl.algorithms.training import train_multiple_seeds
from ecomodrl.algorithms.hyperparams import (
    ALGO_REGISTRY,
    get_algo_fixedcfgs,
)
from ecomodrl.envs.bufferstock import BufferStockSavingsEnv
from ecomodrl.utils.analysis_BS import analyze_all_seeds
from ecomodrl.envs.parameters import BufferStockParams
from ecomodrl.utils.hyperparam_import import load_tuned_config


if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    cfg = RunConfig()
    params = BufferStockParams()

    env_name = "BufferStock"
    algo_key = "ddpg"
    hp = load_tuned_config(env_name=env_name, algo_name=algo_key)

    base_cfg = get_algo_fixedcfgs(algo_key, params, hidden_nodes=128)
    algo_cfg = ALGO_REGISTRY[algo_key](hp, cfg, base_cfg)

    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, BufferStockSavingsEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, BufferStockSavingsEnv, params)