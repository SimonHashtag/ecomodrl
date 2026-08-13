import logging
import multiprocessing as mp
from dataclasses import replace
from typing import Type

from ecomodrl.configs.runconfigs import (
    parse_run_args,
)
from ecomodrl.algorithms.hyperparams import ADAPTER_REGISTRY, get_algo_fixedcfgs
from ecomodrl.tune.hyperparam_import import load_tuned_hp
from ecomodrl.algorithms.training import train_multiple_seeds

def run_experiment(run_config_cls: Type):
    """
    Generic experiment runner for all environments. 
    This function is called by the individual experiment scripts for each environment.
    
    run_config_cls: Type
        The run configuration class for the specific environment.
    """
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    base_cfg = run_config_cls()
    cfg = parse_run_args(base_cfg)
    cfg = replace(cfg, out_dir=cfg.out_dir / cfg.env_name / cfg.algo_name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    params = cfg.make_params()
    env = cfg.env_class()

    hp = load_tuned_hp(env_name=cfg.env_name, algo_name=cfg.algo_name)

    base_algo_cfg = get_algo_fixedcfgs(cfg.algo_name, params, hidden_nodes=cfg.hidden_nodes)
    algo_cfg = ADAPTER_REGISTRY[cfg.algo_name](hp, cfg, base_algo_cfg)
    
    train_multiple_seeds(cfg, algo_cfg, env, params, verbose=1)