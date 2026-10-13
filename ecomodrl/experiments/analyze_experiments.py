from __future__ import annotations

import argparse

from ecomodrl.configs.analysisconfigs import (
    CassKoopmansAnalysisConfig,
    BrockMirmanAnalysisConfig,
    BufferStockAnalysisConfig,
    PortfolioChoiceAnalysisConfig,
    DurableGoodsAnalysisConfig,
)
from ecomodrl.configs.analysisconfigs import parse_analysis_args
from ecomodrl.configs.variants import COMBINED_VARIANT, enforce_variant_constraints
from ecomodrl.analysis.analysis import run_analysis

# Example usage: 
# `python ecomodrl/experiments/analyze_experiments.py --env bufferstock --variant mf`

ANALYSIS_CONFIGS = {
    "casskoopmans": CassKoopmansAnalysisConfig,
    "brockmirman": BrockMirmanAnalysisConfig,
    "bufferstock": BufferStockAnalysisConfig,
    "portfoliochoice": PortfolioChoiceAnalysisConfig,
    "durablegoods": DurableGoodsAnalysisConfig,
}


def parse_env_arg_run_analysis() -> tuple[str, list[str]]:
    """
    Parse only the environment argument first.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--env",
        required=True,
        choices=ANALYSIS_CONFIGS.keys(),
        help="Environment to analyze.",
    )

    args, remaining_args = parser.parse_known_args()
    return args.env, remaining_args


def main() -> None:
    env_name, remaining_args = parse_env_arg_run_analysis()

    cfg_cls = ANALYSIS_CONFIGS[env_name]
    base_cfg = cfg_cls()

    cfg = parse_analysis_args(base_cfg, argv=remaining_args)

    if cfg.variant != COMBINED_VARIANT:
        cfg = enforce_variant_constraints(cfg)

    run_analysis(cfg)


if __name__ == "__main__":
    main()