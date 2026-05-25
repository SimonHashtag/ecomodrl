import logging
import multiprocessing as mp
from dataclasses import replace

from ecomodrl.experiments.casskoopmans.configs import parse_run_args
from ecomodrl.algorithms.hyperparams import ALGO_REGISTRY, get_algo_fixedcfgs
from ecomodrl.utils.hyperparam_import import load_tuned_config
from ecomodrl.algorithms.training import train_multiple_seeds
from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.envs.parameters import CassKoopmansParams
from ecomodrl.utils.analysis_CK import analyze_all_seeds

if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    cfg = parse_run_args()
    params = CassKoopmansParams()

    hp = load_tuned_config(env_name=cfg.env_name, algo_name=cfg.algo_name)

    base_cfg = get_algo_fixedcfgs(cfg.algo_name, params, hidden_nodes=64)
    algo_cfg = ALGO_REGISTRY[cfg.algo_name](hp, cfg, base_cfg)
    
    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, CassKoopmansEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, CassKoopmansEnv, params)