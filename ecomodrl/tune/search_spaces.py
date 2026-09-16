from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from ecomodrl.algorithms.hyperparams import ADAPTER_REGISTRY
from ecomodrl.configs.variants import DEFAULT_VARIANT

# Search space functions take (seed, variant); `variant` ("mf" | "pd") only
# changes the PPO gae_lambda range (see _ppo_space).
SearchSpaceFn = Callable[[int | None, str], CS.ConfigurationSpace]


def _common_space(seed: int | None = None) -> CS.ConfigurationSpace:
    """Shared search space for all algorithms.

    We tune batch_size explicitly and identically for PPO, TD3, DDPG and SAC.
    This avoids algorithm-specific implicit batch-size ranges and prevents
    tiny accidental minibatches.

    With n_envs=10, the steps_per_env choices imply total rollout/update
    blocks between 600 and 2400 transitions.
    """
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add([
        CSH.UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=3e-3, log=True),
        # Log10 drop in learning rate by the end of training.
        CSH.UniformFloatHyperparameter("drop", lower=0.0, upper=2.0),
        # Number of vector-env steps collected before an update.
        # total_steps = n_envs * steps_per_env.
        CSH.OrdinalHyperparameter(
            "steps_per_env",
            sequence=[60, 80, 96, 100, 120, 160, 192, 200, 240],
        ),
        CSH.OrdinalHyperparameter(
            "batch_size", 
            sequence=[32, 48, 64, 75, 96, 100, 128, 150, 192, 200, 256, 384, 512],
        ),

        # PPO: n_epochs.
        # Off-policy: target replay-sample reuse per newly collected transition.
        CSH.OrdinalHyperparameter(
            "sample_reuse", 
            sequence=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ),
    ])

    return cs


def _ppo_space(seed: int | None = None, variant: str = DEFAULT_VARIANT) -> CS.ConfigurationSpace:
    cs = _common_space(seed)
    # Post-decision PPO bootstraps through the quadrature expectation of V, so
    # lambda = 0 is the pure one-step quadrature advantage (no shock noise).
    gae_lambda_lower = 0.0 if variant == "pd" else 0.85
    cs.add([
        CSH.UniformFloatHyperparameter("clip_range", 0.05, 0.30),
        CSH.UniformFloatHyperparameter("gae_lambda", gae_lambda_lower, 0.995),
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


def _sac_space(seed: int | None = None, variant: str = DEFAULT_VARIANT) -> CS.ConfigurationSpace:
    cs = _offpolicy_common_space(seed)
    cs.add([
        CSH.CategoricalHyperparameter(
            "ent_coef",
            choices=["auto", "0.0", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0"],
        ),
    ])
    return cs


def _td3_space(seed: int | None = None, variant: str = DEFAULT_VARIANT) -> CS.ConfigurationSpace:
    cs = _offpolicy_common_space(seed)
    cs.add([
        CSH.UniformFloatHyperparameter("exploration_sigma", 0.02, 0.50),
        CSH.UniformFloatHyperparameter("target_policy_noise", 0.05, 0.50),
        CSH.UniformFloatHyperparameter("target_noise_clip", 0.1, 1.0),
    ])
    return cs


def _ddpg_space(seed: int | None = None, variant: str = DEFAULT_VARIANT) -> CS.ConfigurationSpace:
    cs = _offpolicy_common_space(seed)
    cs.add([
        CSH.UniformFloatHyperparameter("exploration_sigma", 0.02, 0.50),
    ])
    return cs

# Maps algorithm name to its search space function.
SEARCH_SPACE_REGISTRY: Dict[str, SearchSpaceFn] = {
    "ppo": _ppo_space,
    "td3": _td3_space,
    "sac": _sac_space,
    "ddpg": _ddpg_space,
}

# Maps algorithm name to its search space function and hyperparameter adapter function.
TUNE_REGISTRY: Dict[str, Tuple[SearchSpaceFn, Any]] = {
    algo: (space_fn, ADAPTER_REGISTRY[algo])
    for algo, space_fn in SEARCH_SPACE_REGISTRY.items()
}