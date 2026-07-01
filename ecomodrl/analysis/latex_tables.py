from __future__ import annotations

from pathlib import Path

import numpy as np

from ecomodrl.configs.analysisconfigs import ALL_ALGORITHMS


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

TABLE_ROWS = [
    (r"$U_{\mathrm{RL}}$", "u_rl", "max", 1.0),
    (r"$\lambda_{\mathrm{CEV}}$ (\%)", "cev_regret", "min", 100.0),
    (r"$\mathbb{E}[\log_{10}|\mathrm{EE}|]$", "ee_mean", "min", 1.0),
    (r"$W_1(\mathrm{EE}_{\mathrm{RL}},\mathrm{EE}_{\mathrm{EGM}})$", "wasserstein", "min", 1.0),
]

COMBINED_ROW_LABELS = {
    "u_rl": r"$U_{\mathrm{RL}}$",
    "cev_regret": r"$\lambda_{\mathrm{CEV}}$ (\%)",
    "ee_mean": r"$\mathbb{E}[\log_{10}|\mathrm{EE}|]$",
    "wasserstein": r"$W_1(\mathrm{EE})$",
}

NUM_FMT = "%.4f"


def ensure_parent(path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def format_iqm_ci(point, lo, hi, fmt=NUM_FMT):
    """Format IQM and confidence interval for LaTeX tables."""
    return (
        f"\\makecell{{${fmt % point}$ \\\\ "
        f"{{\\scriptsize $[{fmt % lo},\\ {fmt % hi}]$}}}}"
    )


def _bold(cell: str) -> str:
    return r"{\boldmath\bfseries " + cell + "}"


def make_latex_table(model, summaries_by_algo, save_path, *, algorithms=ALL_ALGORITHMS):
    """Write a LaTeX table for one economic model.

    Each cell reports the interquartile mean and a crossed-bootstrap confidence
    interval. The best point estimate in each row is bolded.
    """
    lines = [
        r"\begin{minipage}{\linewidth}",
        r"\centering",
        r"\adjustbox{max width=\linewidth}{%",
        r"\begin{tabular}{l" + "c" * len(algorithms) + r"}",
        r"\toprule",
        "Metric & " + " & ".join(ALGO_LABELS.get(a, a.upper()) for a in algorithms) + r" \\",
        r"\midrule",
    ]

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
                cells[best] = _bold(cells[best])

        lines.append(label + " & " + " & ".join(cells) + r" \\")

    model_name = type(model).__name__

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\par\vspace{4pt}",
        (
            r"{\footnotesize Multi-seed analysis for "
            + MODEL_LABELS.get(model_name, model_name)
            + r". Each cell reports the IQM with a 95\% crossed-bootstrap confidence interval.}"
        ),
        r"\end{minipage}",
    ]

    ensure_parent(save_path)
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_combined_latex_table(
    summaries_by_model,
    save_path,
    *,
    algorithms=ALL_ALGORITHMS,
):
    """Write a compact point-estimate LaTeX table across economic models."""
    lines = [
        r"\begin{frame}[t]{Results}",
        r"  \centering\footnotesize",
        r"  \renewcommand{\arraystretch}{1.15}",
        r"  \begin{tabular}{ll" + "r" * len(algorithms) + r"}",
        r"    \toprule",
        (
            r"    \textbf{Model} & \textbf{Metric} & "
            + " & ".join(r"\textbf{" + ALGO_LABELS.get(a, a.upper()) + "}" for a in algorithms)
            + r" \\"
        ),
        r"    \midrule",
    ]

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

            metric_label = COMBINED_ROW_LABELS[key]
            lines.append(first_cell + " & " + metric_label + " & " + " & ".join(cells) + r" \\")

        if model_idx < len(summaries_by_model) - 1:
            lines.append(r"    \midrule")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{frame}",
    ]

    ensure_parent(save_path)
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")