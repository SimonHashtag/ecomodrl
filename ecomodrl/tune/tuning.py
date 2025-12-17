# DEHB hyperparameter optimization for PPO, TD3, SAC, DDPG
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple, Union

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import torch as th

from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.utils.params_loader import BaseParams

# ----------------------------
# Utilities
# ----------------------------
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

def _divisors(n: int) -> List[int]:
    """Positive divisors of n."""
    ds = set()
    root = int(math.sqrt(n))
    for i in range(1, root + 1):
        if n % i == 0:
            ds.add(i)
            ds.add(n // i)
    return sorted(ds)

def _candidate_minibatches(batch: int) -> List[int]:
    """Small set of useful divisors for PPO minibatches."""
    preferred = [2, 4, 5, 8, 10]
    avail = [m for m in preferred if batch % m == 0]
    if avail:
        return avail
    # fallback: first few divisors >= 2
    alt = [m for m in _divisors(batch) if m >= 2]
    return alt[:5] if alt else [2]


def _seeds_for_fidelity(fidelity: int, min_fid: int, max_fid: int) -> int:
    """Simple seed ramping schedule based on relative budget."""
    phi = (fidelity - min_fid) / max(1, (max_fid - min_fid))
    if phi < 0.05:
        return 1
    if phi < 0.10:
        return 2
    if phi < 0.40:
        return 5
    return 10 # TODO: Let be set by Configs 


# ----------------------------
# Algorithm registry
# ----------------------------
SearchSpaceFn = Callable[[Union[Mapping[str, Any], Any], int | None], CS.ConfigurationSpace]
AdapterFn = Callable[[Dict, Union[Mapping[str, Any], Any], AlgoConfig], AlgoConfig]


def _ppo_space(cfg: Union[Mapping[str, Any], Any], seed: int | None = None) -> CS.ConfigurationSpace:
    cs = CS.ConfigurationSpace(seed=seed)
    B = cfg.n_envs * cfg.max_episode_steps
    cs.add(
        [
            CSH.UniformFloatHyperparameter("learning_rate", 5e-5, 1e-3, log=True),
            CSH.UniformFloatHyperparameter("clip_range", 0.10, 0.30),
            CSH.UniformFloatHyperparameter("gae_lambda", 0.90, 0.98),
            CSH.UniformFloatHyperparameter("ent_coef", 1e-6, 1e-2, log=True),
            CSH.UniformIntegerHyperparameter("n_epochs", 2, 6),
            CSH.CategoricalHyperparameter("num_minibatches", _candidate_minibatches(B)),
        ]
    )
    return cs


def _ppo_adapter(hp: Dict, cfg: Union[Mapping[str, Any], Any], base: AlgoConfig) -> AlgoConfig:
    n_steps = cfg.max_episode_steps
    B = cfg.n_envs * n_steps
    M = int(hp["num_minibatches"])
    n_epochs = int(hp["n_epochs"])
    U = n_epochs * M
    if U > 10:  # * consistent guard
        n_epochs = max(1, min(n_epochs, 10 // M))
    batch_size = B // M

    algo = replace(base)
    algo.algo_params = {
        **(base.algo_params or {}),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "learning_rate": float(hp["learning_rate"]),
        "gae_lambda": float(hp["gae_lambda"]),
        "clip_range": float(hp["clip_range"]),
        "ent_coef": float(hp["ent_coef"]),
    }
    return algo


def _offpolicy_space(cfg: Union[Mapping[str, Any], Any], seed: int | None = None) -> CS.ConfigurationSpace:
    """Shared space for TD3/SAC/DDPG (off-policy)."""
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add(
        [
            CSH.UniformFloatHyperparameter("learning_rate", 1e-5, 3e-3, log=True),
            CSH.CategoricalHyperparameter("batch_size", [64, 128, 256, 512]),
            CSH.UniformFloatHyperparameter("tau", 5e-3, 2e-2),  # soft-update
            CSH.CategoricalHyperparameter("train_freq", [100, 800, 1600, 3200]),
            CSH.CategoricalHyperparameter("gradient_steps", [1, 8, 16, 32]),
            # SAC-only entropy handling will be adapted in adapter:
            CSH.CategoricalHyperparameter("ent_coef", ["auto", 0.0]),
        ]
    )
    return cs


def _offpolicy_adapter_common(hp: Dict, base: AlgoConfig) -> Dict:
    """Translate shared HPs to SB3-style kwargs."""
    # SB3 accepts train_freq as int or (int, "step"); we use the latter
    return {
        **(base.algo_params or {}),
        "learning_rate": float(hp["learning_rate"]),
        "batch_size": int(hp["batch_size"]),
        "tau": float(hp["tau"]),
        "train_freq": (int(hp["train_freq"]), "step"),
        "gradient_steps": int(hp["gradient_steps"]),
        # TODO: buffer_size/learning_starts remain in base defaults unless you add them to space
    }


def _td3_adapter(hp: Dict, cfg: Union[Mapping[str, Any], Any], base: AlgoConfig) -> AlgoConfig:
    algo = replace(base)
    params = _offpolicy_adapter_common(hp, base)
    params.setdefault("policy_delay", 2) # TODO: Tune?
    algo.algo_params = params
    return algo


def _ddpg_adapter(hp: Dict, cfg: Union[Mapping[str, Any], Any], base: AlgoConfig) -> AlgoConfig:
    algo = replace(base)
    algo.algo_params = _offpolicy_adapter_common(hp, base)
    return algo


def _sac_adapter(hp: Dict, cfg: Union[Mapping[str, Any], Any], base: AlgoConfig) -> AlgoConfig:
    algo = replace(base)
    params = _offpolicy_adapter_common(hp, base)
    params["ent_coef"] = hp["ent_coef"]
    algo.algo_params = params
    return algo

ALGO_REGISTRY: Dict[str, Tuple[SearchSpaceFn, AdapterFn]] = {
    "ppo": (_ppo_space, _ppo_adapter),
    "td3": (_offpolicy_space, _td3_adapter),
    "sac": (_offpolicy_space, _sac_adapter),
    "ddpg": (_offpolicy_space, _ddpg_adapter),
}

def get_algo_fixedcfgs(algo_name: str, params: BaseParams) -> AlgoConfig:  # TODO: checks default configs
    algo_name = algo_name.lower()
    if algo_name == "ppo":
        return AlgoConfig(
            name="ppo",
            policy="MlpPolicy",
            algo_params={
                "gamma": params.beta,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
            },
            policy_kwargs={
                "activation_fn": th.nn.ReLU,
                "net_arch": {"pi": [32, 32], "vf": [32, 32]},
                "log_std_init": -2.0,
            },
        )
    elif algo_name in {"td3", "ddpg"}:
        return AlgoConfig(
            name=algo_name,
            policy="MlpPolicy",
            algo_params={
                "gamma": params.beta,
                "buffer_size": 500_000,
                "learning_starts": 1_000,
            },
            policy_kwargs={"activation_fn": th.nn.ReLU, "net_arch": [32, 32]},
        )
    elif algo_name == "sac":
        return AlgoConfig(
            name="sac",
            policy="MlpPolicy",
            algo_params={
                "gamma": params.beta,
                "buffer_size": 500_000,
                "learning_starts": 1_000,
            },
            policy_kwargs={"activation_fn": th.nn.ReLU, "net_arch": [32, 32]},
        )
    else:
        raise ValueError("Unsupported algo_name")

# ----------------------------
# Target function for DEHB
# ----------------------------
def target_function(
    hp: Union[CS.Configuration, Dict, List, np.ndarray],
    fidelity: Union[int, float] = None,
    *,
    cfg: Union[Mapping[str, Any], Any],
    algo_cfg: AlgoConfig,
    params: BaseParams,
    env,
    cs: CS.ConfigurationSpace,
    min_fidelity: int,
    max_fidelity: int,
) -> Dict:
    """
    Train N seeded runs for given hyperparameters and return a scalar objective.

    Returns dict with:
      - 'fitness': (float) lower is better (negative avg best eval return)
      - 'cost':    (float) elapsed wall-time seconds
      - 'info':    (dict)  small diagnostics and paths
    """
    hp = _to_dict(hp, cs)

    # Build a per-trial run config (unique output dir)
    trial_dir = cfg.out_dir / f"trial_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    trial_cfg = replace(cfg, out_dir=trial_dir)

    if fidelity is not None:
        trial_cfg = replace(
            trial_cfg,
            total_timesteps=int(fidelity),
            n_seeds= _seeds_for_fidelity(int(fidelity), int(min_fidelity), int(max_fidelity)), # Reduce number of training runs for low fertilities
        )
    trial_dir.mkdir(parents=True, exist_ok=True)

    try:
        _, adapter_fn = ALGO_REGISTRY[algo_cfg.name.lower()]  # *
    except KeyError as exc:  # *
        raise ValueError(f"Unknown algorithm '{algo_cfg.name}'. Available: {list(ALGO_REGISTRY)}") from exc

    trial_algo = adapter_fn(hp, trial_cfg, algo_cfg)
    trial_algo.policy_kwargs = {**(algo_cfg.policy_kwargs or {})}

    # Run training
    start = time.time()
    try:
        train_multiple_seeds(
            trial_cfg, trial_algo, env, params,
            seed_parallel=False, max_workers=1)
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

    avg_best_return = sum(s["best_eval_return"] for s in train_cfgs) / len(train_cfgs)

    # Minimize
    result = {
        "fitness": -float(avg_best_return),
        "cost": cost,
        "info": {
            "trial_dir": str(trial_dir),
            "n_seeds": trial_cfg.n_seeds,
            "seeds": trial_cfg.seeds,
        },
    }
    return result