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
import torch as th
import math

from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.utils.learning_schedule import LinearScheduleByWallClockTime

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


# def _seeds_for_fidelity(fidelity: int, min_fid: int, max_fid: int, min_seeds: int, max_seeds: int) -> int:
#     """Simple seed ramping schedule based on relative budget."""
#     phi = (fidelity - min_fid) / float(max_fid - min_fid)
#     seeds = int(round((max_seeds - min_seeds) * phi + min_seeds))
#     return max(min_seeds, min(seeds, max_seeds))


# ----------------------------
# Algorithm registry
# ----------------------------
SearchSpaceFn = Callable[[Union[Mapping[str, Any], Any], int | None], CS.ConfigurationSpace]
AdapterFn = Callable[[Dict, Union[Mapping[str, Any], Any], AlgoConfig], AlgoConfig]


def _ppo_space(seed: int | None = None) -> CS.ConfigurationSpace:
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add(
        [
            CSH.UniformFloatHyperparameter("learning_rate", 5e-5, 1e-3, log=True),
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


def _ppo_adapter(hp: Dict, cfg: Union[Mapping[str, Any], Any], base: AlgoConfig) -> AlgoConfig:
    n_steps = cfg.max_episode_steps * int(hp["n_episodes"])
    B = cfg.n_envs * n_steps
    M = int(hp["num_minibatches"])
    batch_size = B // M

    initial_lr = float(hp["learning_rate"])
    lr_schedule = LinearScheduleByWallClockTime(
        initial_lr=initial_lr,
        final_lr=min(initial_lr, 3e-5),
        max_wallclock_seconds=cfg.max_wallclock_seconds, 
    )

    algo = replace(base)
    algo.algo_params = {
        **(base.algo_params or {}),
        "learning_rate": lr_schedule,
        "clip_range": float(hp["clip_range"]),
        "gae_lambda": float(hp["gae_lambda"]),
        "ent_coef": float(hp["ent_coef"]),
        "vf_coef": float(hp["vf_coef"]),
        "n_epochs": int(hp["n_epochs"]),
        "n_steps": n_steps,
        "batch_size": batch_size,
    }
    return algo


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


def _offpolicy_adapter_common(hp: Dict, cfg: Union[Mapping[str, Any], Any], base: AlgoConfig) -> Dict:
    """Translate shared HPs to SB3-style kwargs."""
    n_steps = int(cfg.max_episode_steps) * int(hp["n_episodes"])
    B = int(cfg.n_envs) * n_steps
    gradient_steps = math.ceil(B * float(hp["utd_ratio"]) / int(hp["batch_size"]))
    
    initial_lr = float(hp["learning_rate"])
    lr_schedule = LinearScheduleByWallClockTime(
        initial_lr=initial_lr,
        final_lr=min(initial_lr, 3e-5),
        max_wallclock_seconds=cfg.max_wallclock_seconds,
    )

    return {
        **(base.algo_params or {}),
        "learning_rate": lr_schedule,
        "batch_size": int(hp["batch_size"]),
        "tau": float(hp["tau"]),
        "train_freq": (int(n_steps), "step"),
        "gradient_steps": int(gradient_steps),
    }


def _td3_adapter(hp: Dict, cfg: Union[Mapping[str, Any], Any], base: AlgoConfig) -> AlgoConfig:
    algo = replace(base)
    params = _offpolicy_adapter_common(hp, cfg, base)
    params.setdefault("policy_delay", 2)
    params["action_noise_std"] = float(hp["exploration_sigma"])
    params["target_policy_noise"] = float(hp["target_policy_noise"])
    params["target_noise_clip"] = float(max(hp["target_noise_clip"], hp["target_policy_noise"]))
    algo.algo_params = params
    return algo


def _ddpg_adapter(hp: Dict, cfg: Union[Mapping[str, Any], Any], base: AlgoConfig) -> AlgoConfig:
    algo = replace(base)
    params = _offpolicy_adapter_common(hp, cfg, base)
    params["action_noise_std"] = float(hp["exploration_sigma"])
    algo.algo_params = params
    return algo


def _sac_adapter(hp: Dict, cfg: Union[Mapping[str, Any], Any], base: AlgoConfig) -> AlgoConfig:
    algo = replace(base)
    params = _offpolicy_adapter_common(hp, cfg, base)
    params.setdefault("ent_coef", "auto")
    algo.algo_params = params
    return algo

ALGO_REGISTRY: Dict[str, Tuple[SearchSpaceFn, AdapterFn]] = {
    "ppo": (_ppo_space, _ppo_adapter),
    "td3": (_offpolicy_space, _td3_adapter),
    "sac": (_offpolicy_space, _sac_adapter),
    "ddpg": (_offpolicy_space, _ddpg_adapter),
}

def get_algo_fixedcfgs(algo_name: str, params: Any, hidden_nodes: int) -> AlgoConfig:
    algo_name = algo_name.lower()
    if algo_name == "ppo":
        return AlgoConfig(
            name="ppo",
            policy="MlpPolicy",
            algo_params={
                "gamma": params.beta,
                "max_grad_norm": 0.5,
                "normalize_advantage": True,
                "clip_range_vf": None,
                "target_kl": None,
                "use_sde": False,
            },
            policy_kwargs={
                "activation_fn": th.nn.ReLU,
                "net_arch": {"pi": [hidden_nodes, hidden_nodes], "vf": [hidden_nodes, hidden_nodes]},
                "log_std_init": -2.0,
            },
        )
    elif algo_name in {"td3", "ddpg"}:
        return AlgoConfig(
            name=algo_name,
            policy="MlpPolicy",
            algo_params={
                "gamma": params.beta,
                "max_grad_norm": 0.5,
                "buffer_size": 100_000,
                "learning_starts": 10_000,
            },
            policy_kwargs={"activation_fn": th.nn.ReLU, "net_arch": [hidden_nodes, hidden_nodes]},
        )
    elif algo_name == "sac":
        return AlgoConfig(
            name="sac",
            policy="MlpPolicy",
            algo_params={
                "gamma": params.beta,
                "max_grad_norm": 0.5,
                "action_noise": None,
                "target_update_interval": 1,
                "use_sde": False,
                "buffer_size": 100_000,
                "learning_starts": 10_000,
            },
            policy_kwargs={
                "activation_fn": th.nn.ReLU, 
                "net_arch": [hidden_nodes, hidden_nodes],
                },
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
        _, adapter_fn = ALGO_REGISTRY[algo_cfg.name.lower()]
    except KeyError as exc:  # *
        raise ValueError(f"Unknown algorithm '{algo_cfg.name}'. Available: {list(ALGO_REGISTRY)}") from exc

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
