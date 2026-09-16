# -----------------------------------------------------------------------------
# Algorithm variants: a second design axis next to the algorithm family.
#
#     "mf"  model-free (standard SB3 critics / bootstraps)
#     "pd"  post-decision (model-informed critics and quadrature bootstraps,
#           see ecomodrl/algorithms/post_decision)
#
# Model-free paths are unchanged; post-decision results get a suffix:
#     <Env>/<ALGO>/         ->  <Env>/<ALGO>_PD/
#     <Env>/figures/        ->  <Env>/figures_PD/
#     <Env>/<ALGO>/incumbent.json  ->  <Env>/<ALGO>_PD/incumbent.json
#
# This module imports nothing project-internal so every layer can use it.
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import replace

VARIANTS = ("mf", "pd")
DEFAULT_VARIANT = "mf"


def check_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
    return variant


def variant_suffix(variant: str = DEFAULT_VARIANT) -> str:
    """Directory-name suffix: '' for model-free, '_PD' for post-decision."""
    return "" if check_variant(variant) == "mf" else "_PD"


def algo_dir_name(algo_name: str, variant: str = DEFAULT_VARIANT) -> str:
    """Directory name of one algorithm's results, e.g. 'PPO' or 'PPO_PD'."""
    return algo_name.upper() + variant_suffix(variant)


def figure_dir_name(variant: str = DEFAULT_VARIANT) -> str:
    """Directory name for figures and tables, e.g. 'figures' or 'figures_PD'."""
    return "figures" + variant_suffix(variant)


def enforce_variant_constraints(cfg):
    """Validate `cfg.variant` and apply variant-specific constraints.

    The post-decision critics invert the observation transform
    obs = log(m / m_ref), so observation normalization is switched off for
    variant "pd" (with a warning if it was requested).
    """
    variant = check_variant(getattr(cfg, "variant", DEFAULT_VARIANT))

    if variant == "pd" and getattr(cfg, "normalize_obs", False):
        logging.getLogger(__name__).warning(
            "variant='pd' inverts the observation transform obs = log(m / m_ref); "
            "forcing normalize_obs=False."
        )
        cfg = replace(cfg, normalize_obs=False)

    return cfg
