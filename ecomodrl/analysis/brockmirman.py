# -----------------------------------------------------------------------------
# Entry point for Brock-Mirman post-training analysis.
#
# Examples
# --------
# Analyze all algorithms:
#     python analyze_brockmirman.py
#
# Analyze only PPO:
#     python analyze_brockmirman.py --algo ppo
#
# Only generate plots from existing numerical results:
#     python analyze_brockmirman.py --skip_testing
# -----------------------------------------------------------------------------

from __future__ import annotations

from configs.analysisconfigs import (
    BrockMirmanAnalysisConfig,
    parse_analysis_args,
)
from ecomodrl.analysis.analysis import run_analysis


def main():
    cfg = parse_analysis_args(BrockMirmanAnalysisConfig())
    run_analysis(cfg)


if __name__ == "__main__":
    main()