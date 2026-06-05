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
from ecomodrl.algorithms.hyperparams import ADAPTER_REGISTRY

SearchSpaceFn = Callable[[Any, int | None], CS.ConfigurationSpace]


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


def _common_space(seed: int | None = None) -> CS.ConfigurationSpace:
    """Shared search space for all algorithms.

    Divisibility of (n_envs × max_episode_steps × n_episodes) by
    num_minibatches is enforced at adapter time, not here, because
    DEHB's DE operators are unaware of forbidden clauses.
    """
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add([
        CSH.UniformFloatHyperparameter("learning_rate", 1e-5, 3e-3, log=True),
        CSH.CategoricalHyperparameter("n_episodes", choices=[1, 2, 4, 8]),
        CSH.CategoricalHyperparameter("num_minibatches", choices=[4, 5, 6, 7, 8, 9, 10, 12, 16, 32, 64]),
        CSH.UniformIntegerHyperparameter("sample_reuse", 1, 16),
    ])

    return cs


def _ppo_space(seed: int | None = None) -> CS.ConfigurationSpace:
    cs = _common_space(seed)
    cs.add([
        CSH.UniformFloatHyperparameter("clip_range", 0.05, 0.30),
        CSH.UniformFloatHyperparameter("gae_lambda", 0.85, 0.995),
        CSH.CategoricalHyperparameter("ent_coef", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
        CSH.UniformFloatHyperparameter("vf_coef", 0.25, 1.0),
    ])
    return cs


def _offpolicy_common_space(seed: int | None = None) -> CS.ConfigurationSpace:
    """Shared space for TD3 / SAC / DDPG."""
    cs = _common_space(seed)
    cs.add([
        CSH.UniformFloatHyperparameter("tau", 1e-3, 2e-2, log=True),
    ])
    return cs


def _sac_space(seed: int | None = None) -> CS.ConfigurationSpace:
    cs = _offpolicy_common_space(seed)
    # SAC has no extra tuned HPs beyond the off-policy common ones
    # (ent_coef is fixed to "auto" in the adapter)
    return cs


def _td3_space(seed: int | None = None) -> CS.ConfigurationSpace:
    cs = _offpolicy_common_space(seed)
    cs.add([
        CSH.UniformFloatHyperparameter("exploration_sigma", 0.02, 0.50),
        CSH.UniformFloatHyperparameter("target_policy_noise", 0.05, 0.50),
        CSH.UniformFloatHyperparameter("target_noise_clip", 0.1, 1.0),
    ])
    return cs


def _ddpg_space(seed: int | None = None) -> CS.ConfigurationSpace:
    cs = _offpolicy_common_space(seed)
    cs.add([
        CSH.UniformFloatHyperparameter("exploration_sigma", 0.02, 0.50),
    ])
    return cs


SEARCH_SPACE_REGISTRY: Dict[str, SearchSpaceFn] = {
    "ppo": _ppo_space,
    "td3": _td3_space,
    "sac": _sac_space,
    "ddpg": _ddpg_space,
}

TUNE_REGISTRY: Dict[str, Tuple[SearchSpaceFn, Any]] = {
    algo: (space_fn, ADAPTER_REGISTRY[algo])
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
