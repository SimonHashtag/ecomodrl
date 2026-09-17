import logging
import os
import pickle
import time
from dataclasses import replace
from typing import Type

from dehb import DEHB

from ecomodrl.configs.tuneconfigs import parse_tune_args
from ecomodrl.configs.variants import algo_dir_name, enforce_variant_constraints
from ecomodrl.algorithms.hyperparams import get_algo_fixedcfgs
from ecomodrl.tune.tuning_target import target_function
from ecomodrl.tune.search_spaces import TUNE_REGISTRY


def _worker_count(max_workers: int | None = None) -> int:
    """Use CPUs available to this process"""
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # Windows or unavailable affinity data
        available_cpus = os.cpu_count() or 1

    workers = max(1, available_cpus - 1)
    if max_workers is not None:
        if max_workers < 1:
            raise ValueError("max_tune_workers must be at least one.")
        workers = min(workers, max_workers)
    return workers


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

    cfg = enforce_variant_constraints(parse_tune_args(base_cfg))

    algo_cfg = get_algo_fixedcfgs(
        cfg.algo_name,
        params,
        hidden_nodes=cfg.hidden_nodes,
        variant=cfg.variant,
    )

    cfg = replace(cfg, out_dir=cfg.out_dir / cfg.env_name / algo_dir_name(algo_cfg.name, cfg.variant))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    space_fn, _ = TUNE_REGISTRY[algo_cfg.name.lower()]
    cs = space_fn(seed=cfg.base_seed, variant=cfg.variant)

    min_fidelity = max(
        cfg.max_wallclock_seconds // (cfg.eta ** (cfg.rungs - 1)),
        cfg.min_fidelity_seconds,
    )
    max_fidelity = cfg.max_wallclock_seconds

    n_workers = _worker_count(cfg.max_tune_workers)
    logging.getLogger(__name__).info("DEHB workers=%d", n_workers)

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
