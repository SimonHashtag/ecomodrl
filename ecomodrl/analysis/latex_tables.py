from __future__ import annotations

from pathlib import Path

import numpy as np

from ecomodrl.configs.analysisconfigs import ALL_ALGORITHMS


ALGO_LABELS = {"ppo": "PPO", "ddpg": "DDPG", "td3": "TD3", "sac": "SAC"}
MODEL_LABELS = {
    "CassKoopmans": "Cass--Koopmans",
    "BrockMirman": "Brock--Mirman",
    "BufferStock": "Buffer Stock",
    "PortfolioChoice": "Portfolio Choice",
}
MODEL_TABLE_LABELS = {
    "CassKoopmans": "ck_results",
    "BrockMirman": "bm_results",
    "BufferStock": "bs_results",
    "PortfolioChoice": "pc_results",
}
VARIANT_TABLE_LABELS = {"mf": "Standard", "pd": "Post-decision"}
NUM_FMT = "%.4f"

ACCURACY_ROWS = [
    (r"Expected Utility ($U_{\mathrm{RL}}$)", "u_rl", "max", 1.0, NUM_FMT),
    (r"Consumption Equiv. Variation ($\lambda_{\mathrm{CEV}}$ in \%)", "cev_regret", "min", 100.0, NUM_FMT),
    (r"Euler Equation Error ($\mathbb{E}[\log_{10}|\mathrm{EE}|]$)", "ee_mean", "min", 1.0, NUM_FMT),
]
EFFICIENCY_ROWS = [
    (r"Area Under Learning Curve (AULC)", "aulc", "min", 1.0, "%.4f"),
    (r"Time-to-best [s]", "time_to_best", "min", 1.0, "%.0f"),
    (r"Time-to-near-best [s]", "time_to_converged", "min", 1.0, "%.0f"),
]
COMPLEXITY_ROWS = [
    (r"FLOPs/update ($\log_{10}$)", "flops_per_update", "min", 1.0, "%.2f"),
    (r"FLOPs to near-best ($\log_{10}$)", "flops_to_conv", "min", 1.0, "%.2f"),
]

COMBINED_ACCURACY_ROWS = [(key, direction, scale, fmt) for _, key, direction, scale, fmt in ACCURACY_ROWS]
COMBINED_EFFICIENCY_ROWS = [(key, direction, scale, fmt) for _, key, direction, scale, fmt in EFFICIENCY_ROWS]
COMBINED_COMPLEXITY_ROWS = [(key, direction, scale, fmt) for _, key, direction, scale, fmt in COMPLEXITY_ROWS]
COMBINED_ROW_LABELS = {
    "u_rl": r"$U_{\mathrm{RL}}$", "cev_regret": r"$\lambda_{\mathrm{CEV}}$ (\%)",
    "ee_mean": r"$\mathbb{E}[\log_{10}|\mathrm{EE}|]$", "aulc": r"AULC",
    "time_to_best": r"$t_{\mathrm{best}}$ [s]", "time_to_converged": r"$t_{\varepsilon}$ [s]",
    "flops_per_update": r"FLOPs/upd. ($\log_{10}$)",
    "flops_to_conv": r"FLOPs$_{\mathrm{near-best}}$ ($\log_{10}$)",
}


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def format_iqm_ci(point: float, lo: float, hi: float, fmt: str = NUM_FMT) -> str:
    return f"\\makecell{{${fmt % point}$ \\\\[-2pt] {{\\scriptsize $[{fmt % lo},\\ {fmt % hi}]$}}}}"


def _bold(cell: str) -> str:
    return r"{\boldmath\bfseries " + cell + "}"


def _apply_transform(key: str, value: float) -> float:
    if key in {"flops_per_update", "flops_to_conv"}:
        return float(np.log10(value)) if np.isfinite(value) and value > 0.0 else float("nan")
    return value


def _has_ci(stats: dict) -> bool:
    return all(np.isfinite(stats.get(key, np.nan)) for key in ("ci_lo", "ci_hi")) and int(stats.get("n", 0)) > 1


def _present_rows(rows: list[tuple], summaries_by_algo: dict, algorithms: list[str]) -> list[tuple]:
    return [row for row in rows if any(summaries_by_algo.get(algo, {}).get(row[1], {}).get("n", 0) for algo in algorithms)]


def _cells_for_summaries(summaries_by_algo: dict, key: str, direction: str, scale: float, fmt: str, algorithms: list[str]) -> list[str]:
    values, cells = [], []
    for algo in algorithms:
        stats = summaries_by_algo.get(algo, {}).get(key, {})
        if not stats or not stats.get("n", 0):
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
    values = np.asarray(values, dtype=np.float64)
    if np.any(np.isfinite(values)):
        best = np.nanmax(values) if direction == "max" else np.nanmin(values)
        for index, value in enumerate(values):
            if not np.isfinite(value) or (fmt % value) != (fmt % best):
                continue
            if cells[index] != "--":
                cells[index] = _bold(cells[index])
    return cells


def _panel(lines: list[str], rows: list[tuple], summaries_by_algo: dict, algorithms: list[str], *, first: bool = False) -> None:
    if not first:
        lines.extend([r"\midrule", r"\addlinespace[2pt]"])
    for title, key, direction, scale, fmt in rows:
        lines.extend([rf"\multicolumn{{{len(algorithms) + 1}}}{{@{{}}l}}{{\textbf{{{title}}}}} \\", r"\addlinespace[2pt]"])
        lines.append(r"\quad Results & " + " & ".join(_cells_for_summaries(summaries_by_algo, key, direction, scale, fmt, algorithms)) + r" \\")
        lines.append(r"\addlinespace[4pt]")


def _notes() -> str:
    return (r"\item Notes. The first panel displays the IQM of accuracy metrics with corresponding 95\% crossed "
            r"(train$\times$test seed) bootstrap CI. The second and third panels report efficiency and complexity metrics, respectively, with 95\% bootstrap CIs computed over training seeds. Network forward and backward FLOPs are measured using PyTorch's \texttt{FlopCounter}, with Adam and target-update FLOPs added analytically. Bold values indicate the best performance per metric.")


def _model_caption(model_name: str, variant: str | None = None) -> str:
    name = MODEL_LABELS.get(model_name, model_name)
    if variant is None:
        return rf"Results for the {name} model."
    return rf"Results for the {name} model ({VARIANT_TABLE_LABELS[variant].lower()} variant)."


def _table_label(model_name: str, variant: str | None = None) -> str:
    label = MODEL_TABLE_LABELS.get(model_name, model_name.lower() + "_results")
    return "tab:" + label + ("_" + variant if variant is not None else "")


def make_latex_table(model: object, summaries_by_algo: dict, save_path: str | Path, *, algorithms: list[str] = ALL_ALGORITHMS, variant: str = "mf") -> None:
    """Write one paper-style results table for a single algorithm variant."""
    model_name = type(model).__name__
    panels = [
        _present_rows(ACCURACY_ROWS, summaries_by_algo, algorithms),
        _present_rows(EFFICIENCY_ROWS, summaries_by_algo, algorithms),
        _present_rows(COMPLEXITY_ROWS, summaries_by_algo, algorithms),
    ]
    lines = [r"\begin{table*}[!htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{5.5pt}", r"\adjustbox{max width=\linewidth}{%", r"\begin{tabular}{l" + "c" * len(algorithms) + r"}", r"\toprule", "Metric & " + " & ".join(ALGO_LABELS.get(a, a.upper()) for a in algorithms) + r" \\", r"\midrule", r"\addlinespace[2pt]"]
    for index, rows in enumerate(panels):
        if rows:
            _panel(lines, rows, summaries_by_algo, algorithms, first=index == 0)
    lines += [r"\bottomrule", r"\end{tabular}}", r"\par\vspace{4pt}", r"\begin{tablenotes}", r"\footnotesize", _notes(), r"\end{tablenotes}", rf"\caption{{{_model_caption(model_name, variant)}}}", rf"\label{{{_table_label(model_name, variant)}}}", r"\end{table*}"]
    ensure_parent(save_path)
    Path(save_path).write_text("\n".join(lines) + "\n")


def _variant_cells(summaries_by_variant: dict, key: str, direction: str, scale: float, fmt: str, algorithms: list[str]) -> dict[str, list[str]]:
    values, cells = [], {variant: [] for variant in VARIANT_TABLE_LABELS}
    ordered = [(variant, algo) for variant in VARIANT_TABLE_LABELS for algo in algorithms]
    for variant, algo in ordered:
        stats = summaries_by_variant.get(variant, {}).get(algo, {}).get(key, {})
        if not stats or not stats.get("n", 0):
            values.append(np.nan); cells[variant].append("--"); continue
        point = _apply_transform(key, stats["iqm"] * scale)
        values.append(point)
        if _has_ci(stats):
            lo = _apply_transform(key, stats["ci_lo"] * scale); hi = _apply_transform(key, stats["ci_hi"] * scale)
            cells[variant].append(format_iqm_ci(point, lo, hi, fmt))
        else:
            cells[variant].append("$" + (fmt % point) + "$")
    values = np.asarray(values, dtype=np.float64)
    if np.any(np.isfinite(values)):
        best = np.nanmax(values) if direction == "max" else np.nanmin(values)
        for index, value in enumerate(values):
            if not np.isfinite(value) or (fmt % value) != (fmt % best):
                continue
            variant, algo = ordered[int(index)]
            position = algorithms.index(algo)
            cells[variant][position] = _bold(cells[variant][position])
    return cells


def make_variant_comparison_latex_table(model_name: str, summaries_by_variant: dict, save_path: str | Path, *, algorithms: list[str] = ALL_ALGORITHMS) -> None:
    """Write the standard-versus-post-decision paper table for one model."""
    lines = [r"\begin{table*}[!htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{5.5pt}", r"\adjustbox{max width=\linewidth}{%", r"\begin{tabular}{l" + "c" * len(algorithms) + r"}", r"\toprule", "Variant & " + " & ".join(ALGO_LABELS.get(a, a.upper()) for a in algorithms) + r" \\", r"\midrule", r"\addlinespace[2pt]"]
    for panel_index, rows in enumerate((ACCURACY_ROWS, EFFICIENCY_ROWS, COMPLEXITY_ROWS)):
        if panel_index:
            lines.extend([r"\midrule", r"\addlinespace[2pt]"])
        for title, key, direction, scale, fmt in rows:
            cells = _variant_cells(summaries_by_variant, key, direction, scale, fmt, algorithms)
            lines.extend([rf"\multicolumn{{{len(algorithms) + 1}}}{{@{{}}l}}{{\textbf{{{title}}}}} \\", r"\addlinespace[2pt]"])
            for variant, label in VARIANT_TABLE_LABELS.items():
                lines.append(r"\quad " + label + " & " + " & ".join(cells[variant]) + r" \\")
            lines.append(r"\addlinespace[4pt]")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\par\vspace{4pt}", r"\begin{tablenotes}", r"\footnotesize", _notes(), r"\end{tablenotes}", rf"\caption{{{_model_caption(model_name)}}}", rf"\label{{{_table_label(model_name)}}}", r"\end{table*}"]
    ensure_parent(save_path)
    Path(save_path).write_text("\n".join(lines) + "\n")


def _combined_cells(summaries_by_algo: dict, key: str, direction: str, scale: float, fmt: str, algorithms: list[str]) -> list[str]:
    """Point-estimate-only cells for the retained cross-model Beamer table."""
    values, cells = [], []
    for algo in algorithms:
        stats = summaries_by_algo.get(algo, {}).get(key, {})
        if not stats or not stats.get("n", 0):
            values.append(np.nan)
            cells.append("--")
            continue
        point = _apply_transform(key, stats["iqm"] * scale)
        values.append(point)
        cells.append("$" + (fmt % point) + "$")
    values = np.asarray(values, dtype=np.float64)
    if np.any(np.isfinite(values)):
        best = np.nanmax(values) if direction == "max" else np.nanmin(values)
        for index, value in enumerate(values):
            if not np.isfinite(value) or (fmt % value) != (fmt % best):
                continue
            if cells[index] != "--":
                cells[index] = r"$\mathbf{" + (fmt % values[index]) + "}$"
    return cells

def make_combined_latex_table(summaries_by_model: list[tuple[str, dict]], save_path: str | Path, *, algorithms: list[str] = ALL_ALGORITHMS) -> None:
    """Retain the compact cross-model Beamer comparison table."""
    ncols, end_col = len(algorithms), len(algorithms) + 2
    rows = COMBINED_ACCURACY_ROWS + COMBINED_EFFICIENCY_ROWS + COMBINED_COMPLEXITY_ROWS
    lines = [r"\begin{frame}[t]{Results}", r"  \centering\footnotesize", r"  \renewcommand{\arraystretch}{1.15}", r"  \begin{tabular}{ll" + "r" * ncols + r"}", r"    \toprule", r"    \textbf{Model} & \textbf{Metric} & " + " & ".join(r"\textbf{" + ALGO_LABELS.get(a, a.upper()) + "}" for a in algorithms) + r" \\", r"    \midrule"]
    for model_index, (model_name, summaries) in enumerate(summaries_by_model):
        present = [(key, direction, scale, fmt) for key, direction, scale, fmt in rows if any(summaries.get(a, {}).get(key, {}).get("n", 0) for a in algorithms)]
        for index, (key, direction, scale, fmt) in enumerate(present):
            first = rf"    \multirow{{{len(present)}}}{{*}}{{\textbf{{{MODEL_LABELS.get(model_name, model_name)}}}}}" if index == 0 else r"    "
            lines.append(first + " & " + COMBINED_ROW_LABELS[key] + " & " + " & ".join(_combined_cells(summaries, key, direction, scale, fmt, algorithms)) + r" \\")
        if model_index < len(summaries_by_model) - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{frame}"]
    ensure_parent(save_path)
    Path(save_path).write_text("\n".join(lines) + "\n")
