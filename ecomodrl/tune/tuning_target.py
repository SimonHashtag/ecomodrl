from __future__ import annotations

import gc
import json
import time
import uuid
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Union

import ConfigSpace as CS
import numpy as np

from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.algorithms.hyperparams import ADAPTER_REGISTRY
from ecomodrl.algorithms.stopper import TrainingClock


def _to_dict(
    hp: Union[CS.Configuration, Mapping, Iterable, np.ndarray],
    cs: CS.ConfigurationSpace,
) -> Dict:
    """Normalize DEHB hyperparameter payload to a plain dict."""
    if isinstance(hp, CS.Configuration):
        return dict(hp)
    if isinstance(hp, Mapping):
        return dict(hp)
    return {hpdef.name: val for hpdef, val in zip(list(cs.values()), hp)} 


def target_function(
    hp: Union[CS.Configuration, Dict, List, np.ndarray],
    fidelity: Union[int, float] = None,
    *,
    cfg: Union[Mapping[str, Any], Any],
    algo_cfg: AlgoConfig,
    params: Any,
    env,
    cs: CS.ConfigurationSpace,
    max_fidelity: int,
    train_verbose: int = 0
) -> Dict:
    """
    Train N seeded runs for given hyperparameters and return a scalar objective.

    Returns dict with:
      - 'fitness': (float) lower is better (negative avg last eval return)
      - 'cost':    (float) elapsed wall-time seconds
      - 'info':    (dict)  small diagnostics and paths
    """
    hp = _to_dict(hp, cs)

    # Build a per-trial run config (unique output dir)
    trial_dir = cfg.out_dir / f"trial_{int(np.round(max_fidelity/fidelity))}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    trial_cfg = replace(cfg, out_dir=trial_dir)

    if fidelity is not None:
        trial_cfg = replace(
            trial_cfg,
            max_wallclock_seconds=int(fidelity),
            n_seeds= cfg.n_seeds,
        )
    trial_dir.mkdir(parents=True, exist_ok=True)

    clock = TrainingClock()

    try:
        adapter_fn = ADAPTER_REGISTRY[algo_cfg.name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown algorithm '{algo_cfg.name}'. Available: {list(ADAPTER_REGISTRY)}"
        ) from exc

    # Pass 'cfg' (Global) instead of 'trial_cfg' (Truncated) to the adapter.
    # This ensures the LR Schedule is calculated based on the full 10M steps,
    # even though training will stop early due to trial_cfg in the next step.
    trial_algo = adapter_fn(hp, cfg, algo_cfg) 
    
    trial_algo.policy_kwargs = {**(algo_cfg.policy_kwargs or {})}

    # Run training
    try:
        try:
            train_multiple_seeds(
                trial_cfg, trial_algo, env, params,
                seed_parallel=False, max_workers=1, verbose=train_verbose)
        except Exception as exc:
            # Fail-safe: return a bad fitness but with measured cost
            cost = clock.now()
            return {"fitness": 1e9, "cost": cost, "info": {"error": repr(exc)}}
        cost = clock.now()

        # Read results
        train_cfgs_path = trial_dir / "train_cfgs.json"
        if not train_cfgs_path.exists():
            return {"fitness": 1e9, "cost": cost, "info": {"error": "train_cfgs.json missing"}}

        with open(train_cfgs_path, "r") as f:
            train_cfgs = json.load(f)

        if not train_cfgs:
            return {"fitness": 1e9, "cost": cost, "info": {"error": "empty train_cfgs.json"}}

        avg_last_return = sum(s["last_eval_return"] for s in train_cfgs) / len(train_cfgs) # TODO: Last eval return?

        # Minimize
        result = {
            "fitness": -float(avg_last_return),
            "cost": cost,
            "info": {
                "trial_dir": str(trial_dir),
                "n_seeds": trial_cfg.n_seeds,
                "seeds": trial_cfg.seeds,
            },
        }
        return result
    finally:
        gc.collect()
