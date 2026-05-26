from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping

import torch as th

from ecomodrl.algorithms.training import AlgoConfig
from ecomodrl.utils.learning_schedule import LinearScheduleByWallClockTime

AdapterFn = Callable[[Dict[str, Any], Mapping[str, Any] | Any, AlgoConfig], AlgoConfig]

DEFAULT_HP_BY_ALGO: Dict[str, Dict[str, Any]] = {
    "ppo": {
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "n_epochs": 10,
        "n_episodes": 7,
        "num_minibatches": 33,
    },
    "td3": {
        "learning_rate": 1e-3,
        "batch_size": 256,
        "tau": 0.005,
        "n_episodes": 1,
        "utd_ratio": 1.0,
        "exploration_sigma": 0.1,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
    },
    "ddpg": {
        "learning_rate": 1e-3,
        "batch_size": 256,
        "tau": 0.005,
        "n_episodes": 1,
        "utd_ratio": 1.0,
        "exploration_sigma": 0.1,
    },
    "sac": {
        "learning_rate": 3e-4,
        "batch_size": 256,
        "tau": 0.005,
        "n_episodes": 1,
        "utd_ratio": 1.0,
        "exploration_sigma": 0.1,
    },
}


def _ppo_adapter(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> AlgoConfig:
    base_params = dict(base.algo_params or {})
    n_steps = cfg.max_episode_steps * int(hp["n_episodes"])
    batch_size = (cfg.n_envs * n_steps) // int(hp["num_minibatches"])

    initial_lr = float(hp["learning_rate"])
    final_lr_multiplier = float(base_params.pop("final_lr_multiplier", 1.0))
    lr_schedule = LinearScheduleByWallClockTime(
        initial_lr=initial_lr,
        final_lr=initial_lr * final_lr_multiplier,
        max_wallclock_seconds=cfg.max_wallclock_seconds,
    )

    algo = replace(base)
    algo.algo_params = {
        **base_params,
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


def _offpolicy_adapter_common(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> Dict[str, Any]:
    base_params = dict(base.algo_params or {})
    n_steps = int(cfg.max_episode_steps) * int(hp["n_episodes"])
    total_steps = int(cfg.n_envs) * n_steps
    gradient_steps = math.ceil(
        total_steps * float(hp["utd_ratio"]) / int(hp["batch_size"])
    )

    initial_lr = float(hp["learning_rate"])
    final_lr_multiplier = float(base_params.pop("final_lr_multiplier", 1.0))
    lr_schedule = LinearScheduleByWallClockTime(
        initial_lr=initial_lr,
        final_lr=initial_lr * final_lr_multiplier,
        max_wallclock_seconds=cfg.max_wallclock_seconds,
    )

    return {
        **base_params,
        "learning_rate": lr_schedule,
        "batch_size": int(hp["batch_size"]),
        "tau": float(hp["tau"]),
        "train_freq": (int(n_steps), "step"),
        "gradient_steps": int(gradient_steps),
    }


def _td3_adapter(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> AlgoConfig:
    algo = replace(base)
    params = _offpolicy_adapter_common(hp, cfg, base)
    params.setdefault("policy_delay", 2)
    params["action_noise_std"] = float(hp["exploration_sigma"])
    params["target_policy_noise"] = float(hp["target_policy_noise"])
    params["target_noise_clip"] = float(
        max(hp["target_noise_clip"], hp["target_policy_noise"])
    )
    algo.algo_params = params
    return algo


def _ddpg_adapter(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> AlgoConfig:
    algo = replace(base)
    params = _offpolicy_adapter_common(hp, cfg, base)
    params["action_noise_std"] = float(hp["exploration_sigma"])
    algo.algo_params = params
    return algo


def _sac_adapter(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> AlgoConfig:
    algo = replace(base)
    params = _offpolicy_adapter_common(hp, cfg, base)
    params.setdefault("ent_coef", "auto")
    algo.algo_params = params
    return algo


ALGO_REGISTRY: Dict[str, AdapterFn] = {
    "ppo": _ppo_adapter,
    "td3": _td3_adapter,
    "sac": _sac_adapter,
    "ddpg": _ddpg_adapter,
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
                "final_lr_multiplier": 1.0, # TODO: Decide for a final LR multiplier
            },
            policy_kwargs={
                "activation_fn": th.nn.ReLU,
                "net_arch": {
                    "pi": [hidden_nodes, hidden_nodes],
                    "vf": [hidden_nodes, hidden_nodes],
                },
                "log_std_init": -2.0,
            },
        )
    if algo_name in {"td3", "ddpg"}:
        return AlgoConfig(
            name=algo_name,
            policy="MlpPolicy",
            algo_params={
                "gamma": params.beta,
                "max_grad_norm": 0.5,
                "buffer_size": 100_000,
                "learning_starts": 10_000,
                "final_lr_multiplier": 1.0, # TODO: Decide for a final LR multiplier
            },
            policy_kwargs={
                "activation_fn": th.nn.ReLU,
                "net_arch": [hidden_nodes, hidden_nodes],
            },
        )
    if algo_name == "sac":
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
                "final_lr_multiplier": 1.0, # TODO: Decide for a final LR multiplier
            },
            policy_kwargs={
                "activation_fn": th.nn.ReLU,
                "net_arch": [hidden_nodes, hidden_nodes],
            },
        )
    raise ValueError("Unsupported algo_name")
