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

NUM_FMT = "%.4f"

ACCURACY_ROWS = [
    (r"$U_{\mathrm{RL}}$", "u_rl", "max", 1.0, NUM_FMT),
    (r"$\lambda_{\mathrm{CEV}}$ (\%)", "cev_regret", "min", 100.0, NUM_FMT),
    (r"$\mathbb{E}[\log_{10}|\mathrm{EE}|]$", "ee_mean", "min", 1.0, NUM_FMT),
    (r"$W_1(\mathrm{EE}_{\mathrm{RL}},\mathrm{EE}_{\mathrm{EGM}})$", "wasserstein", "min", 1.0, NUM_FMT),
]

EFFICIENCY_ROWS = [
    (r"AULC", "aulc", "min", 1.0, "%.4f"),
    (r"Time-to-best [s]", "time_to_best", "min", 1.0, "%.0f"),
    (r"Time-to-conv. [s]", "time_to_converged", "min", 1.0, "%.0f"),
]

COMPLEXITY_ROWS = [
    (r"FLOPs/update ($\log_{10}$)", "flops_per_update", "min", 1.0, "%.2f"),
    (r"FLOPs to conv.\ ($\log_{10}$)", "flops_to_conv", "min", 1.0, "%.2f"),
    (r"Sampled states [M]", "sampled_states", "min", 1e-6, "%.2f"),
]

COMBINED_ACCURACY_ROWS = [
    ("u_rl",        "max", 1.0,   NUM_FMT),
    ("cev_regret",  "min", 100.0, NUM_FMT),
    ("ee_mean",     "min", 1.0,   NUM_FMT),
    ("wasserstein", "min", 1.0,   NUM_FMT),
]

COMBINED_EFFICIENCY_ROWS = [
    ("aulc",            "min", 1.0, "%.3f"),
    ("time_to_best",    "min", 1.0, "%.0f"),
    ("time_to_converged", "min", 1.0, "%.0f"),
]

COMBINED_COMPLEXITY_ROWS = [
    ("flops_per_update", "min", 1.0, "%.2f"),
    ("flops_to_conv",    "min", 1.0, "%.2f"),
    ("sampled_states",   "min", 1e-6, "%.2f"),
]

COMBINED_ROW_LABELS = {
    "u_rl": r"$U_{\mathrm{RL}}$",
    "cev_regret": r"$\lambda_{\mathrm{CEV}}$ (\%)",
    "ee_mean": r"$\mathbb{E}[\log_{10}|\mathrm{EE}|]$",
    "wasserstein": r"$W_1(\mathrm{EE})$",
    "aulc": r"AULC",
    "time_to_best": r"$t_{\mathrm{best}}$ [s]",
    "time_to_converged": r"$t_{\varepsilon}$ [s]",
    "flops_per_update": r"FLOPs/upd. ($\log_{10}$)",
    "flops_to_conv": r"FLOPs$_{\mathrm{conv}}$ ($\log_{10}$)",
    "sampled_states": r"States [M]",
}


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def format_iqm_ci(point: float, lo: float, hi: float, fmt: str = NUM_FMT) -> str:
    """Format IQM and confidence interval for LaTeX tables."""
    return (
        f"\\makecell{{${fmt % point}$ \\\\ "
        f"{{\\scriptsize $[{fmt % lo},\\ {fmt % hi}]$}}}}"
    )


def _bold(cell: str) -> str:
    return r"{\boldmath\bfseries " + cell + "}"


def _section_row(title: str, ncols: int) -> str:
    return r"\multicolumn{" + str(ncols + 1) + r"}{l}{\textit{" + title + r"}} \\"


def _apply_transform(key: str, value: float) -> float:
    """Apply a per-metric display transform (log10 for FLOPs metrics)."""
    if key in {"flops_per_update", "flops_to_conv"}:
        return float(np.log10(value)) if (np.isfinite(value) and value > 0.0) else float("nan")
    return value


def _has_ci(stats: dict) -> bool:
    return (
        "ci_lo" in stats
        and "ci_hi" in stats
        and np.isfinite(stats.get("ci_lo", np.nan))
        and np.isfinite(stats.get("ci_hi", np.nan))
        and int(stats.get("n", 0)) > 1
    )


def _emit_rows(lines: list[str], rows: list[tuple], summaries_by_algo: dict, algorithms: list[str]):
    """Append metric rows, bolding the best point estimate per row.

    Cells with a confidence interval are formatted as IQM + CI; deterministic
    point metrics (e.g. FLOPs/update, n=1) are formatted as a bare value.
    """
    for label, key, direction, scale, fmt in rows:
        values = []
        cells = []

        for algo in algorithms:
            stats = summaries_by_algo.get(algo, {}).get(key, {})

            if not stats or stats.get("n", 0) == 0:
                values.append(np.nan)
                cells.append("--")
                continue

            point = _apply_transform(key, stats["iqm"] * scale)
            values.append(point)

            if _has_ci(stats):
                lo = _apply_transform(key, stats["ci_lo"] * scale)
                hi = _apply_transform(key, stats["ci_hi"] * scale)
                cells.append(format_iqm_ci(point, lo, hi, fmt))
            else:
                cells.append("$" + (fmt % point) + "$")

        arr = np.asarray(values, dtype=np.float64)
        if np.any(np.isfinite(arr)):
            best = np.nanargmax(arr) if direction == "max" else np.nanargmin(arr)
            if cells[best] != "--":
                cells[best] = _bold(cells[best])

        lines.append(label + " & " + " & ".join(cells) + r" \\")


def _present_rows(rows: list[tuple], summaries_by_algo: dict, algorithms: list[str]) -> list[tuple]:
    """Keep only rows that at least one algorithm reports."""
    return [
        r for r in rows
        if any(summaries_by_algo.get(a, {}).get(r[1], {}).get("n", 0) for a in algorithms)
    ]


def make_latex_table(
    model: object,
    summaries_by_algo: dict,
    save_path: str,
    *,
    algorithms: list[str] = ALL_ALGORITHMS,
):
    """Write a LaTeX table with Accuracy, Efficiency, and Complexity sections."""
    ncols = len(algorithms)

    lines = [
        r"\begin{table}",
        r"{\centering",
        r"\adjustbox{max width=\linewidth}{%",
        r"\begin{tabular}{l" + "c" * ncols + r"}",
        r"\toprule",
        "Metric & " + " & ".join(ALGO_LABELS.get(a, a.upper()) for a in algorithms) + r" \\",
        r"\midrule",
        _section_row("Accuracy", ncols),
    ]

    _emit_rows(lines, ACCURACY_ROWS, summaries_by_algo, algorithms)

    # Efficiency section
    eff_rows = list(EFFICIENCY_ROWS)
    has_conv = any(
        summaries_by_algo.get(a, {}).get("time_to_converged", {}).get("n", 0)
        for a in algorithms
    )
    if not has_conv:
        eff_rows = [r for r in eff_rows if r[1] != "time_to_converged"]

    lines.append(r"\midrule")
    lines.append(_section_row("Efficiency", ncols))
    _emit_rows(lines, eff_rows, summaries_by_algo, algorithms)

    # Complexity section
    comp_rows = _present_rows(COMPLEXITY_ROWS, summaries_by_algo, algorithms)
    if comp_rows:
        lines.append(r"\midrule")
        lines.append(_section_row("Complexity", ncols))
        _emit_rows(lines, comp_rows, summaries_by_algo, algorithms)

    model_name = type(model).__name__

    lines += [
        r"\bottomrule",
        r"\end{tabular}}}",
        r"\par\vspace{4pt}",
        (
            r"\caption{Multi-seed analysis for "
            + MODEL_LABELS.get(model_name, model_name)
            + r". Accuracy cells report the IQM with a 95\% crossed "
            r"(train$\times$test seed) bootstrap CI. Efficiency cells report the "
            r"IQM with a 95\% bootstrap CI over training seeds; AULC is the "
            r"time-averaged normalized regret. Complexity reports cumulative "
            r"FLOPs to convergence ($\log_{10}$, IQM $+$ CI over seeds) and the "
            r"deterministic FLOPs per minibatch update; network forward/backward "
            r"FLOPs are measured with PyTorch's FlopCounter, with Adam and "
            r"target-update FLOPs added analytically. Lower is better for all "
            r"efficiency and complexity metrics.}"
        ),
        r"\end{table}",
    ]

    ensure_parent(save_path)
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _combined_cells(
    summaries_by_algo: dict,
    key: str,
    direction: str,
    scale: float,
    fmt: str,
    algorithms: list[str],
):
    """Point-estimate cells for one combined-table row, bolding the best."""
    values, cells = [], []

    for algo in algorithms:
        stats = summaries_by_algo.get(algo, {}).get(key, {})
        if not stats or stats.get("n", 0) == 0:
            values.append(np.nan)
            cells.append("--")
            continue
        point = _apply_transform(key, stats["iqm"] * scale)
        values.append(point)
        cells.append("$" + (fmt % point) + "$")

    arr = np.asarray(values, dtype=np.float64)
    if np.any(np.isfinite(arr)):
        best = np.nanargmax(arr) if direction == "max" else np.nanargmin(arr)
        if cells[best] != "--":
            cells[best] = r"$\mathbf{" + (fmt % arr[best]) + "}$"

    return cells


def make_combined_latex_table(
    summaries_by_model: list[tuple[str, dict]],
    save_path: str,
    *,
    algorithms: list[str] = ALL_ALGORITHMS,
):
    """Compact point-estimate LaTeX table across economic models."""
    ncols = len(algorithms)
    end_col = ncols + 2  # columns: Model | Metric | <algorithms...>

    lines = [
        r"\begin{frame}[t]{Results}",
        r"  \centering\footnotesize",
        r"  \renewcommand{\arraystretch}{1.15}",
        r"  \begin{tabular}{ll" + "r" * ncols + r"}",
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

        acc_rows = list(COMBINED_ACCURACY_ROWS)

        eff_rows = []
        for key, direction, scale, fmt in COMBINED_EFFICIENCY_ROWS:
            present = any(
                summaries_by_algo.get(a, {}).get(key, {}).get("n", 0) for a in algorithms
            )
            if key == "time_to_converged" and not present:
                continue
            eff_rows.append((key, direction, scale, fmt))

        comp_rows = [
            (key, direction, scale, fmt)
            for key, direction, scale, fmt in COMBINED_COMPLEXITY_ROWS
            if any(summaries_by_algo.get(a, {}).get(key, {}).get("n", 0) for a in algorithms)
        ]

        total = len(acc_rows) + len(eff_rows) + len(comp_rows)

        # --- accuracy sub-block  ---
        for row_idx, (key, direction, scale, fmt) in enumerate(acc_rows):
            cells = _combined_cells(summaries_by_algo, key, direction, scale, fmt, algorithms)
            first_cell = (
                r"    \multirow{" + str(total) + r"}{*}{\textbf{" + model_label + r"}}"
                if row_idx == 0
                else r"   "
            )
            lines.append(
                first_cell + " & " + COMBINED_ROW_LABELS[key] + " & " + " & ".join(cells) + r" \\"
            )

        # --- efficiency sub-block ---
        if eff_rows:
            lines.append(r"    \cmidrule(lr){2-" + str(end_col) + r"}")
            for key, direction, scale, fmt in eff_rows:
                cells = _combined_cells(summaries_by_algo, key, direction, scale, fmt, algorithms)
                lines.append(
                    r"    & " + COMBINED_ROW_LABELS[key] + " & " + " & ".join(cells) + r" \\"
                )

        # --- complexity sub-block ---
        if comp_rows:
            lines.append(r"    \cmidrule(lr){2-" + str(end_col) + r"}")
            for key, direction, scale, fmt in comp_rows:
                cells = _combined_cells(summaries_by_algo, key, direction, scale, fmt, algorithms)
                lines.append(
                    r"    & " + COMBINED_ROW_LABELS[key] + " & " + " & ".join(cells) + r" \\"
                )

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