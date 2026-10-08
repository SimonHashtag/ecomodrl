from __future__ import annotations

import argparse
import os
from pathlib import Path

from ecomodrl.analysis.latex_tables import MODEL_LABELS, make_selected_hyperparameters_latex_table
from ecomodrl.configs.analysisconfigs import ALL_ALGORITHMS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write selected-hyperparameter LaTeX tables.")
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_LABELS), default=tuple(MODEL_LABELS),
        help="Model directories to include (default: all four).",
    )
    parser.add_argument(
        "--variant", choices=("mf", "pd", "both"), default="both",
        help="Algorithm variants to include (default: both).",
    )
    parser.add_argument(
        "--tune-results-dir", type=Path,
        default=Path(os.environ.get("TUNERESULTS_PATH") or "./tune_results"),
        help="Root containing <Model>/<ALGO>/incumbent.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("TUNERESULTS_PATH") or "./tune_results") / "./tables")
    args = parser.parse_args(argv)

    variants = ("mf", "pd") if args.variant == "both" else (args.variant,)
    for algo in ALL_ALGORITHMS:
        path = args.output_dir / f"hp_selected_{algo}.tex"
        make_selected_hyperparameters_latex_table(
            algo, args.models, variants, args.tune_results_dir, path,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
