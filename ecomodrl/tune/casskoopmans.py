import logging
import os
import pickle
import time
from dataclasses import replace

from dehb import DEHB

from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.tune.tuning import TUNE_REGISTRY, target_function
from ecomodrl.algorithms.hyperparams import get_algo_fixedcfgs
from ecomodrl.configs.parameters import CassKoopmansParams
from ecomodrl.configs.tuneconfigs import CassKoopmansTuneConfig, parse_tune_args

# Run file with e.g.: python ecomodrl/tune/casskoopmans.py --algo_name="ddpg" --total_cost=3600 --hidden_nodes=64

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    params = CassKoopmansParams()
    base_cfg = CassKoopmansTuneConfig()
    cfg = parse_tune_args(base_cfg)
    algo_cfg = get_algo_fixedcfgs(cfg.algo_name, params, hidden_nodes=cfg.hidden_nodes)
    
    cfg = replace(cfg, out_dir=cfg.out_dir / cfg.env_name / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    space_fn, _ = TUNE_REGISTRY[algo_cfg.name.lower()]
    cs = space_fn(seed=cfg.base_seed)

    eta = cfg.eta
    rungs = cfg.rungs
    MIN_FIDELITY_SECONDS = cfg.min_fidelity_seconds
    min_fidelity = max(cfg.max_wallclock_seconds // (eta**(rungs-1)), MIN_FIDELITY_SECONDS)
    max_fidelity = cfg.max_wallclock_seconds
    n_workers = max(1, (os.cpu_count() or 2) - 1)

    dehb = DEHB(
        f=target_function,
        cs=cs,
        dimensions=len(list(cs.values())),
        min_fidelity=min_fidelity,
        max_fidelity=max_fidelity,
        n_workers=n_workers,
        output_path=cfg.out_dir,
        log_level="INFO",
        eta=eta,
        crossover_prob=cfg.crossover_prob,
        mutation_factor=cfg.mutation_factor,
    )

    traj, runtime, history = dehb.run(
        total_cost=cfg.total_cost,
        cfg=cfg,
        algo_cfg=algo_cfg,
        params=params,
        env=CassKoopmansEnv,
        cs=cs,
        min_fidelity=min_fidelity,   
        max_fidelity=max_fidelity, 
        train_verbose=0,
    )

    # Save optimization trace history
    name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(dehb.start))
    out_file = cfg.out_dir / f"history_{name}.pkl"
    with open(os.path.join(out_file), "wb") as f:
        pickle.dump(history, f)