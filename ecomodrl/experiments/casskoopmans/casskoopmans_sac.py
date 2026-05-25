import logging
import multiprocessing as mp
from dataclasses import replace

from ecomodrl.algorithms.training import train_multiple_seeds
from ecomodrl.algorithms.hyperparams import (
    ALGO_REGISTRY,
    get_algo_fixedcfgs,
)
from ecomodrl.experiments.casskoopmans.configs import RunConfig
from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.envs.parameters import CassKoopmansParams
from ecomodrl.utils.analysis_CK import analyze_all_seeds
from ecomodrl.utils.hyperparam_import import load_tuned_config


if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    cfg = RunConfig()
    params = CassKoopmansParams()
    
    env_name = "CassKoopmans"
    algo_key = "sac"
    hp = load_tuned_config(env_name=env_name, algo_name=algo_key)

    base_cfg = get_algo_fixedcfgs(algo_key, params, hidden_nodes=64)
    algo_cfg = ALGO_REGISTRY[algo_key](hp, cfg, base_cfg)
    
    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, CassKoopmansEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, CassKoopmansEnv, params)