from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping

import torch as th

from ecomodrl.algorithms.training import AlgoConfig
from ecomodrl.utils.learning_schedule import LinearScheduleByWallClockTime

AdapterFn = Callable[[Dict[str, Any], Mapping[str, Any] | Any, AlgoConfig], AlgoConfig]

# ---------------------------------------------------------------------------
# Approximate SB3 defaults:
#   PPO:  lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
#         clip=0.2, gae_λ=0.95, ent_coef=0, vf_coef=0.5
#         → num_minibatches≈32, sample_reuse=10
#   TD3:  lr=1e-3, batch_size=100, tau=0.005, gradient_steps=-1 (UTD≈1),
#         σ_explore=0.1, σ_target=0.2, clip_target=0.5
#   DDPG: same as TD3 without target-policy smoothing
#   SAC:  lr=3e-4, batch_size=256, tau=0.005, ent_coef="auto"
# ---------------------------------------------------------------------------
DEFAULT_HP_BY_ALGO: Dict[str, Dict[str, Any]] = {
    "ppo": {
        "learning_rate": 3e-4,
        "n_episodes": 4,
        "num_minibatches": 32,
        "sample_reuse": 10,          # = n_epochs
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
    },
    "td3": {
        "learning_rate": 1e-3,
        "n_episodes": 1,
        "num_minibatches": 4,
        "sample_reuse": 1,           # gradient_steps = sample_reuse * num_minibatches = 4
        "tau": 0.005,
        "exploration_sigma": 0.1,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
    },
    "ddpg": {
        "learning_rate": 1e-3,
        "n_episodes": 1,
        "num_minibatches": 4,
        "sample_reuse": 1,
        "tau": 0.005,
        "exploration_sigma": 0.1,
    },
    "sac": {
        "learning_rate": 3e-4,
        "n_episodes": 1,
        "num_minibatches": 4,
        "sample_reuse": 1,
        "tau": 0.005,
        # ent_coef fixed to "auto" in adapter; no exploration_sigma for SAC
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap_minibatches(total_steps: int, requested: int) -> int:
    """Return the candidate closest to *requested* that divides *total_steps*.

    If no candidate divides total_steps, fall back to 1 (single batch).
    """
    valid = [m for m in range(4, 64) if total_steps % m == 0]
    if not valid:
        return 1
    return min(valid, key=lambda m: abs(m - requested))


def _compute_steps(hp: Dict[str, Any], cfg: Any):
    """Derive step / batch quantities from the unified search-space params.

    Returns ``(steps_per_env, total_steps, batch_size, gradient_steps)``.

    ``num_minibatches`` is snapped to the nearest value that actually
    divides ``total_steps`` so the caller never sees a remainder.
    """
    steps_per_env = int(cfg.max_episode_steps) * int(hp["n_episodes"])
    total_steps = int(cfg.n_envs) * steps_per_env

    num_mb = _snap_minibatches(total_steps, int(hp["num_minibatches"]))

    batch_size = total_steps // num_mb
    gradient_steps = int(hp["sample_reuse"]) * num_mb
    return steps_per_env, total_steps, batch_size, gradient_steps


def _set_lr_schedule(
    hp: Dict[str, Any],
    cfg: Any,
    params: Dict[str, Any],
) -> None:
    initial_lr = float(hp["learning_rate"])
    final_lr_multiplier = float(params.pop("final_lr_multiplier", 1.0))

    params["learning_rate"] = LinearScheduleByWallClockTime(
        initial_lr=initial_lr,
        final_lr=initial_lr * final_lr_multiplier,
        max_wallclock_seconds=cfg.max_wallclock_seconds,
    )


# ---------------------------------------------------------------------------
# Per-algorithm adapters
# ---------------------------------------------------------------------------
def _ppo_adapter(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> AlgoConfig:
    steps_per_env, total_steps, batch_size, _ = _compute_steps(hp, cfg)
    params = dict(base.algo_params or {})

    _set_lr_schedule(hp, cfg, params)

    params.update({
        "n_steps": steps_per_env,
        "batch_size": batch_size,
        "n_epochs": int(hp["sample_reuse"]),

        # PPO-specific
        "clip_range": float(hp["clip_range"]),
        "gae_lambda": float(hp["gae_lambda"]),
        "ent_coef": float(hp["ent_coef"]),
        "vf_coef": float(hp["vf_coef"]),
    })

    return replace(base, algo_params=params)


def _offpolicy_adapter_common(
    hp: Dict[str, Any], cfg: Any, base: AlgoConfig,
) -> Dict[str, Any]:
    steps_per_env, total_steps, batch_size, gradient_steps = _compute_steps(hp, cfg)
    params = dict(base.algo_params or {})

    _set_lr_schedule(hp, cfg, params)

    params.update({
        "train_freq": (int(steps_per_env), "step"),
        "batch_size": batch_size,
        "gradient_steps": gradient_steps,

        # Off-policy common
        "tau": float(hp["tau"]),
    })
    return params


def _td3_adapter(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> AlgoConfig:
    params = _offpolicy_adapter_common(hp, cfg, base)
    params.update({
        "policy_delay": 2,
        "action_noise_std": float(hp["exploration_sigma"]),
        "target_policy_noise": float(hp["target_policy_noise"]),
        "target_noise_clip": float(
            max(hp["target_noise_clip"], hp["target_policy_noise"])
        ),
    })
    return replace(base, algo_params=params)


def _ddpg_adapter(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> AlgoConfig:
    params = _offpolicy_adapter_common(hp, cfg, base)
    params["action_noise_std"] = float(hp["exploration_sigma"])
    return replace(base, algo_params=params)


def _sac_adapter(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> AlgoConfig:
    params = _offpolicy_adapter_common(hp, cfg, base)
    params.setdefault("ent_coef", "auto")
    return replace(base, algo_params=params)


# Renamed to avoid collision with ecomodrl.algorithms.hyperparams.ALGO_REGISTRY
ADAPTER_REGISTRY: Dict[str, AdapterFn] = {
    "ppo": _ppo_adapter,
    "td3": _td3_adapter,
    "sac": _sac_adapter,
    "ddpg": _ddpg_adapter,
}


# ---------------------------------------------------------------------------
# Fixed (non-tuned) config builder
# ---------------------------------------------------------------------------
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
                "final_lr_multiplier": 1.0,
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
                "final_lr_multiplier": 1.0,
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
                "target_update_interval": 1,
                "use_sde": False,
                "buffer_size": 100_000,
                "learning_starts": 10_000,
                "final_lr_multiplier": 1.0,
            },
            policy_kwargs={
                "activation_fn": th.nn.ReLU,
                "net_arch": [hidden_nodes, hidden_nodes],
            },
        )
    raise ValueError(f"Unsupported algo_name: {algo_name!r}")