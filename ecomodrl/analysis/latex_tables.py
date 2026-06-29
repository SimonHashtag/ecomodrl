from __future__ import annotations

from pathlib import Path

import numpy as np

from ecomodrl.algorithms.training import ALGO_REGISTRY


DEFAULT_ALGORITHMS = tuple(ALGO_REGISTRY.keys())

ALGO_LABELS = {
    "ppo": "PPO",
    "ddpg": "DDPG",
    "td3": "TD3",
    "sac": "SAC",
}

MODEL_LABELS = {
    "CassKoopmans": "Cass--Koopmans",
    "BrockMirman": "Brock--Mirman",
    "BufferStock": "Buffer Stock",
    "PortfolioChoice": "Portfolio Choice",
}

def ensure_parent(path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

TABLE_ROWS = [
    (r"$U_{\mathrm{RL}}$", "u_rl", "max", 1.0),
    ("CEV (%)", "cev_regret", "min", 100.0),
    ("Euler error mean", "ee_mean", "min", 1.0),
    ("Wasserstein", "wasserstein", "min", 1.0),
]

NUM_FMT = "%.4f"


def format_iqm_ci(point, lo, hi, fmt=NUM_FMT):
    return (
        f"\\makecell{{${fmt % point}$ \\\\ "
        f"{{\\scriptsize $[{fmt % lo},\\ {fmt % hi}]$}}}}"
    )


def make_latex_table(model, summaries_by_algo, save_path, *, algorithms=DEFAULT_ALGORITHMS):
    """Write a LaTeX table for one economic model."""
    lines = []
    lines.append(r"\begin{minipage}{\linewidth}")
    lines.append(r"\centering")
    lines.append(r"\adjustbox{max width=\linewidth}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(algorithms) + r"}")
    lines.append(r"\toprule")
    lines.append("Metric & " + " & ".join(ALGO_LABELS.get(a, a.upper()) for a in algorithms) + r" \\")
    lines.append(r"\midrule")

    for label, key, direction, scale in TABLE_ROWS:
        values = []
        cells = []

        for algo in algorithms:
            stats = summaries_by_algo.get(algo, {}).get(key, {})

            if not stats or stats.get("n", 0) == 0:
                values.append(np.nan)
                cells.append("--")
                continue

            point = stats["iqm"] * scale
            lo = stats["ci_lo"] * scale
            hi = stats["ci_hi"] * scale

            values.append(point)
            cells.append(format_iqm_ci(point, lo, hi))

        arr = np.asarray(values, dtype=np.float64)

        if np.any(np.isfinite(arr)):
            best = np.nanargmax(arr) if direction == "max" else np.nanargmin(arr)
            if cells[best] != "--":
                cells[best] = r"{\boldmath\bfseries " + cells[best] + "}"

        lines.append(label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\par\vspace{4pt}")

    model_name = type(model).__name__
    lines.append(
        r"{\footnotesize Multi-seed analysis for "
        + model_name
        + r". Each cell reports the IQM with a 95\% crossed-bootstrap confidence interval.}"
    )
    lines.append(r"\end{minipage}")

    ensure_parent(save_path)
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_combined_latex_table(
    summaries_by_model,
    save_path,
    *,
    algorithms=DEFAULT_ALGORITHMS,
):
    """Write a compact point-estimate table across models."""
    lines = []
    lines.append(r"\begin{frame}[t]{Results}")
    lines.append(r"  \centering\footnotesize")
    lines.append(r"  \renewcommand{\arraystretch}{1.15}")
    lines.append(r"  \begin{tabular}{ll" + "r" * len(algorithms) + r"}")
    lines.append(r"    \toprule")

    header = (
        r"    \textbf{Model} & \textbf{Metric} & "
        + " & ".join(r"\textbf{" + ALGO_LABELS.get(a, a.upper()) + "}" for a in algorithms)
        + r" \\"
    )
    lines.append(header)
    lines.append(r"    \midrule")

    for model_idx, (model_key, summaries_by_algo) in enumerate(summaries_by_model):
        model_label = MODEL_LABELS.get(model_key, model_key)

        for row_idx, (_, key, direction, scale) in enumerate(TABLE_ROWS):
            values = []
            cells = []

            for algo in algorithms:
                stats = summaries_by_algo.get(algo, {}).get(key, {})

                if not stats or stats.get("n", 0) == 0:
                    values.append(np.nan)
                    cells.append("--")
                    continue

                point = stats["iqm"] * scale
                values.append(point)
                cells.append("$" + (NUM_FMT % point) + "$")

            arr = np.asarray(values, dtype=np.float64)

            if np.any(np.isfinite(arr)):
                best = np.nanargmax(arr) if direction == "max" else np.nanargmin(arr)
                if cells[best] != "--":
                    cells[best] = r"$\mathbf{" + (NUM_FMT % arr[best]) + "}$"

            first_cell = (
                r"    \multirow{" + str(len(TABLE_ROWS)) + r"}{*}{\textbf{" + model_label + r"}}"
                if row_idx == 0
                else r"   "
            )

            metric_label = key.replace("_", r"\_")
            lines.append(first_cell + " & " + metric_label + " & " + " & ".join(cells) + r" \\")

        if model_idx < len(summaries_by_model) - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{frame}")

    ensure_parent(save_path)
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")