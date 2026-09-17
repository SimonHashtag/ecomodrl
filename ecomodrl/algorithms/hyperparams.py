from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping

import torch as th

from ecomodrl.algorithms.training import AlgoConfig

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
        "steps_per_env": 200,
        "batch_size": 64,
        "sample_reuse": 10,          # = n_epochs
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "drop": 1.0,
    },
    "td3": {
        "learning_rate": 1e-3,
        "steps_per_env": 60,
        "batch_size": 100,
        "sample_reuse": 1,
        "tau": 0.005,
        "exploration_sigma": 0.1,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "drop": 1.0,
    },
    "ddpg": {
        "learning_rate": 1e-3,
        "steps_per_env": 60,
        "batch_size": 100,
        "sample_reuse": 1,
        "tau": 0.005,
        "exploration_sigma": 0.1,
        "drop": 1.0,
    },
    "sac": {
        "learning_rate": 3e-4,
        "steps_per_env": 120,
        "batch_size": 256,
        "sample_reuse": 1,
        "tau": 0.005,
        "drop": 1.0,
        "ent_coef": "auto",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_steps(hp: Dict[str, Any], cfg: Any):
    """Derive rollout/update quantities from explicit common HPs.

    Returns:
        steps_per_env:
            Number of vector-env steps before one algorithm update phase.

        batch_size:
            Minibatch size for PPO or replay batch size for off-policy methods.

        gradient_steps:
            Number of off-policy gradient steps chosen so that

                gradient_steps * batch_size ≈ sample_reuse * total_steps.

            For PPO this value is unused; PPO uses sample_reuse as n_epochs.
    """
    steps_per_env = int(hp["steps_per_env"])
    batch_size = int(hp["batch_size"])
    sample_reuse = int(hp["sample_reuse"])

    total_steps = int(cfg.n_envs) * steps_per_env

    target_gradient_steps = sample_reuse * total_steps / batch_size
    gradient_steps = max(1, int(math.floor(target_gradient_steps + 0.5)))

    return steps_per_env, batch_size, gradient_steps


# ---------------------------------------------------------------------------
# Per-algorithm adapters
# ---------------------------------------------------------------------------
def _ppo_adapter(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> AlgoConfig:
    steps_per_env, batch_size, _ = _compute_steps(hp, cfg)
    params = dict(base.algo_params or {})

    params.update({
        "learning_rate": float(hp["learning_rate"]),
        "drop": float(hp["drop"]),
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


def _offpolicy_adapter_common(hp: Dict[str, Any], cfg: Any, base: AlgoConfig,) -> Dict[str, Any]:
    steps_per_env, batch_size, gradient_steps = _compute_steps(hp, cfg)
    params = dict(base.algo_params or {})

    params.update({
        "learning_rate": float(hp["learning_rate"]),
        "drop": float(hp["drop"]),
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
        "target_noise_clip": float(hp["target_noise_clip"]),
    })
    return replace(base, algo_params=params)


def _ddpg_adapter(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> AlgoConfig:
    params = _offpolicy_adapter_common(hp, cfg, base)
    params["action_noise_std"] = float(hp["exploration_sigma"])
    return replace(base, algo_params=params)


def _sac_adapter(hp: Dict[str, Any], cfg: Any, base: AlgoConfig) -> AlgoConfig:
    params = _offpolicy_adapter_common(hp, cfg, base)
    ent_coef = hp.get("ent_coef", "auto")
    if ent_coef == "auto":
        params["ent_coef"] = "auto"
    else:
        try:
            params["ent_coef"] = float(ent_coef)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid SAC ent_coef value: {ent_coef!r}") from exc
    return replace(base, algo_params=params)


ADAPTER_REGISTRY: Dict[str, AdapterFn] = {
    "ppo": _ppo_adapter,
    "td3": _td3_adapter,
    "sac": _sac_adapter,
    "ddpg": _ddpg_adapter,
}


# ---------------------------------------------------------------------------
# Fixed (non-tuned) config builder
# ---------------------------------------------------------------------------
def get_algo_fixedcfgs(algo_name: str, params: Any, hidden_nodes: int, variant: str = "mf") -> AlgoConfig:
    """Fixed (non-tuned) part of the algorithm config.

    `variant` selects the model-free ("mf") or post-decision ("pd") classes in
    `build_model`; the fixed settings are shared by both variants.
    """
    algo_name = algo_name.lower()
    if algo_name == "ppo":
        return AlgoConfig(
            name="ppo",
            variant=variant,
            policy="MlpPolicy",
            algo_params={
                "gamma": params.beta,
                "max_grad_norm": 1.0,
                "normalize_advantage": True,
                "clip_range_vf": None,
                "target_kl": None,
                "use_sde": False,
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
            variant=variant,
            policy="MlpPolicy",
            algo_params={
                "gamma": params.beta,
                "max_grad_norm": 1.0,
                "buffer_size": 10_000,
                "learning_starts": 0,
            },
            policy_kwargs={
                "activation_fn": th.nn.ReLU,
                "net_arch": [hidden_nodes, hidden_nodes],
            },
        )
    if algo_name == "sac":
        return AlgoConfig(
            name="sac",
            variant=variant,
            policy="MlpPolicy",
            algo_params={
                "gamma": params.beta,
                "max_grad_norm": 1.0,
                "target_update_interval": 1,
                "use_sde": False,
                "buffer_size": 10_000,
                "learning_starts": 0,
            },
            policy_kwargs={
                "activation_fn": th.nn.ReLU,
                "net_arch": [hidden_nodes, hidden_nodes],
            },
        )
    raise ValueError(f"Unsupported algo_name: {algo_name!r}")