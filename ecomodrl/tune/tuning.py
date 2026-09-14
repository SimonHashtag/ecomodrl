import logging
import os
import pickle
import time
from dataclasses import replace
from typing import Type

import dask
from dehb import DEHB

from ecomodrl.configs.tuneconfigs import parse_tune_args
from ecomodrl.algorithms.hyperparams import get_algo_fixedcfgs
from ecomodrl.tune.tuning_target import target_function
from ecomodrl.tune.search_spaces import TUNE_REGISTRY


def run_tuning(tune_config_cls: Type, train_verbose: int = 0):
    """
    Shared DEHB tuning runner.

    Parameters
    ----------
    tune_config_cls:
        A TuneConfig class, e.g. BrockMirmanTuneConfig.
    train_verbose:
        Verbosity passed to the training function.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    base_cfg = tune_config_cls()
    params = base_cfg.make_params()
    env = base_cfg.env_class()

    cfg = parse_tune_args(base_cfg)

    algo_cfg = get_algo_fixedcfgs(
        cfg.algo_name,
        params,
        hidden_nodes=cfg.hidden_nodes,
    )

    cfg = replace(cfg, out_dir=cfg.out_dir / cfg.env_name / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    space_fn, _ = TUNE_REGISTRY[algo_cfg.name.lower()]
    cs = space_fn(seed=cfg.base_seed)

    min_fidelity = max(
        cfg.max_wallclock_seconds // (cfg.eta ** (cfg.rungs - 1)),
        cfg.min_fidelity_seconds,
    )
    max_fidelity = cfg.max_wallclock_seconds

    n_workers = max(1, (os.cpu_count() or 2) - 1)

    # Disable the per-worker thresholds and let the job's total memory (e.g. SLURM --mem) be the only bound.
    dask.config.set({
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.pause": False,
        "distributed.worker.memory.terminate": False,
    })

    dehb = DEHB(
        f=target_function,
        cs=cs,
        dimensions=len(list(cs.values())),
        min_fidelity=min_fidelity,
        max_fidelity=max_fidelity,
        n_workers=n_workers,
        output_path=cfg.out_dir,
        log_level="INFO",
        eta=cfg.eta,
        crossover_prob=cfg.crossover_prob,
        mutation_factor=cfg.mutation_factor,
    )

    traj, runtime, history = dehb.run(
        total_cost=cfg.total_cost,
        cfg=cfg,
        algo_cfg=algo_cfg,
        params=params,
        env=env,
        cs=cs,
        max_fidelity=max_fidelity,
        train_verbose=train_verbose,
    )

    name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(dehb.start))
    out_file = cfg.out_dir / f"history_{name}.pkl"

    with out_file.open("wb") as f:
        pickle.dump(history, f)

    return traj, runtime, history