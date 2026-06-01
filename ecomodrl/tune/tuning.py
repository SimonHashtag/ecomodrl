# DEHB hyperparameter optimization for PPO, TD3, SAC, DDPG
from __future__ import annotations

import gc
import json
import time
import uuid
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple, Union

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.algorithms.hyperparams import ALGO_REGISTRY

SearchSpaceFn = Callable[[Union[Mapping[str, Any], Any], int | None], CS.ConfigurationSpace]


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


def _ppo_space(seed: int | None = None) -> CS.ConfigurationSpace:
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add(
        [
            CSH.UniformFloatHyperparameter("learning_rate", 1e-5, 3e-3, log=True),
            CSH.UniformFloatHyperparameter("clip_range", 0.05, 0.30),
            CSH.UniformFloatHyperparameter("gae_lambda", 0.90, 0.995),
            CSH.UniformFloatHyperparameter("ent_coef", 1e-6, 1e-2, log=True),
            CSH.UniformFloatHyperparameter("vf_coef", 0.5, 1.0),
            CSH.UniformIntegerHyperparameter("n_epochs", 2, 10),
            CSH.CategoricalHyperparameter("n_episodes", [1, 2, 4, 8]),
            CSH.CategoricalHyperparameter("num_minibatches", [4, 5, 6, 8, 10]),
        ]
    )
    return cs


def _offpolicy_space(seed: int | None = None) -> CS.ConfigurationSpace:
    """Shared space for TD3/SAC/DDPG (off-policy)."""
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add(
        [
            CSH.UniformFloatHyperparameter("learning_rate", 1e-5, 3e-3, log=True),
            CSH.CategoricalHyperparameter("batch_size", [32, 64, 128, 256, 512]),
            CSH.UniformFloatHyperparameter("tau", 2e-3, 2e-2),  # soft-update
            CSH.CategoricalHyperparameter("n_episodes", [1, 2, 4, 8]),
            CSH.UniformFloatHyperparameter("utd_ratio", 1.0, 5.0),  # updates-to-data ratio
            # Used by DDPG/TD3
            CSH.UniformFloatHyperparameter("exploration_sigma", 0.05, 0.50),
            CSH.UniformFloatHyperparameter("target_policy_noise", 0.05, 0.5),
            CSH.UniformFloatHyperparameter("target_noise_clip", 0.1, 1.0),
        ]
    )
    return cs


SEARCH_SPACE_REGISTRY: Dict[str, SearchSpaceFn] = {
    "ppo": _ppo_space,
    "td3": _offpolicy_space,
    "sac": _offpolicy_space,
    "ddpg": _offpolicy_space,
}

TUNE_REGISTRY: Dict[str, Tuple[SearchSpaceFn, Any]] = {
    algo: (space_fn, ALGO_REGISTRY[algo])
    for algo, space_fn in SEARCH_SPACE_REGISTRY.items()
}

# ----------------------------
# Target function for DEHB
# ----------------------------
def target_function(
    hp: Union[CS.Configuration, Dict, List, np.ndarray],
    fidelity: Union[int, float] = None,
    *,
    cfg: Union[Mapping[str, Any], Any],
    algo_cfg: AlgoConfig,
    params: Any,
    env,
    cs: CS.ConfigurationSpace,
    min_fidelity: int,
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

    try:
        adapter_fn = ALGO_REGISTRY[algo_cfg.name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown algorithm '{algo_cfg.name}'. Available: {list(ALGO_REGISTRY)}"
        ) from exc

    # Pass 'cfg' (Global) instead of 'trial_cfg' (Truncated) to the adapter.
    # This ensures the LR Schedule is calculated based on the full 10M steps,
    # even though training will stop early due to trial_cfg in the next step.
    trial_algo = adapter_fn(hp, cfg, algo_cfg) 
    
    trial_algo.policy_kwargs = {**(algo_cfg.policy_kwargs or {})}

    # Run training
    start = time.time()
    try:
        try:
            train_multiple_seeds(
                trial_cfg, trial_algo, env, params,
                seed_parallel=False, max_workers=1, verbose=train_verbose)
        except Exception as exc:
            # Fail-safe: return a bad fitness but with measured cost
            cost = time.time() - start
            return {"fitness": 1e9, "cost": cost, "info": {"error": repr(exc)}}
        cost = time.time() - start

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
