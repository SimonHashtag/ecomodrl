from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np

from ecomodrl.configs.analysisconfigs import ALL_ALGORITHMS
from ecomodrl.configs.variants import algo_dir_name


ALGO_LABELS = {"ppo": "PPO", "ddpg": "DDPG", "td3": "TD3", "sac": "SAC"}
MODEL_LABELS = {
    "CassKoopmans": "Cass--Koopmans",
    "BrockMirman": "Brock--Mirman",
    "BufferStock": "Buffer Stock",
    "PortfolioChoice": "Portfolio Choice",
    "DurableGoods": "Durable goods (Fella 2014)",
}
MODEL_TABLE_LABELS = {
    "CassKoopmans": "ck_results",
    "BrockMirman": "bm_results",
    "BufferStock": "bs_results",
    "PortfolioChoice": "pc_results",
    "DurableGoods": "dg_results",
}
VARIANT_TABLE_LABELS = {"mf": "Standard", "pd": "Post-decision"}
NUM_FMT = "%.4f"

COMMON_HP_ROWS = [
    (r"$\alpha$", "learning_rate"), (r"$d$", "drop"),
    (r"$n_{\text{steps}}$", "steps_per_env"), (r"$M$", "batch_size"),
    (r"$N$", "sample_reuse"),
]
OFFPOLICY_HP_ROWS = [(r"$\tau$", "tau")]
SELECTED_HP_ROWS = {
    "ppo": COMMON_HP_ROWS + [
        (r"$c$", "clip_range"),
        (r"$\lambda/\lambda_{\text{PD}}$", "gae_lambda"),
        (r"$\rho_{\text{PPO}}$", "ent_coef"), (r"$w_v$", "vf_coef"),
    ],
    "td3": COMMON_HP_ROWS + OFFPOLICY_HP_ROWS + [
        (r"$\sigma$", "exploration_sigma"),
        (r"$\tilde{\sigma}$", "target_policy_noise"),
        (r"$c_{\text{clip}}$", "target_noise_clip"),
    ],
    "ddpg": COMMON_HP_ROWS + OFFPOLICY_HP_ROWS + [
        (r"$\sigma$", "exploration_sigma"),
    ],
    "sac": COMMON_HP_ROWS + OFFPOLICY_HP_ROWS + [
        (r"$\rho_{\text{SAC}}$", "ent_coef"),
    ],
}

ACCURACY_ROWS = [
    (r"Empirical Finite-Horizon Return ($U_{\mathrm{RL}}$)", "u_rl", "max", 1.0, NUM_FMT),
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


def _selected_hp_cell(key: str, value: object) -> str:
    if value == "auto" and key == "ent_coef":
        return r"\text{auto}"
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value for {key}: {value!r}")
    if key in {"steps_per_env", "batch_size", "sample_reuse"}:
        if not number.is_integer():
            raise ValueError(f"Expected an integer for {key}: {value!r}")
        return str(int(number))
    if number == 0:
        return "0"
    if key == "learning_rate" or key == "ent_coef" or abs(number) < 0.01:
        mantissa, exponent = f"{number:.1e}".split("e")
        if key == "ent_coef" and mantissa == "1.0":
            return rf"10^{{{int(exponent)}}}"
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    places = max(0, 1 - math.floor(math.log10(abs(number))))
    return f"{number:.{places}f}"


def _selected_incumbent(path: Path, keys: list[str]) -> Mapping | None:
    if not path.exists():
        warnings.warn(f"Missing incumbent: {path}", stacklevel=2)
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read incumbent {path}: {exc}") from exc
    config = data.get("config") if isinstance(data, Mapping) else None
    if not isinstance(config, Mapping):
        raise ValueError(f"Incumbent {path} has no config mapping")
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Incumbent {path} is missing parameters: {', '.join(missing)}")
    for key in keys:
        try:
            _selected_hp_cell(key, config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {key} in incumbent {path}: {exc}") from exc
    return config


def make_selected_hyperparameters_latex_table(
    algo_name: str,
    models: list[str] | tuple[str, ...],
    variants: list[str] | tuple[str, ...],
    tune_results_dir: str | Path,
    save_path: str | Path,
) -> None:
    """Write selected hyperparameters for one algorithm from its incumbents."""
    algo = algo_name.lower()
    if algo not in SELECTED_HP_ROWS:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
    if not models or any(model not in MODEL_LABELS for model in models):
        raise ValueError(f"Unknown or empty model selection: {models}")
    if not variants or any(variant not in ("mf", "pd") for variant in variants):
        raise ValueError(f"Unknown or empty variant selection: {variants}")

    rows = SELECTED_HP_ROWS[algo]
    root = Path(tune_results_dir)
    configs = {}
    for model in models:
        for variant in variants:
            path = root / model / algo_dir_name(algo, variant) / "incumbent.json"
            configs[model, variant] = _selected_incumbent(path, [key for _, key in rows])

    width = len(models) * len(variants)
    model_headers = []
    rules = []
    for index, model in enumerate(models):
        first = 2 + index * len(variants)
        last = first + len(variants) - 1
        model_headers.append(rf"\multicolumn{{{len(variants)}}}{{c}}{{{MODEL_LABELS[model]}}}")
        rules.append(rf"\cmidrule(lr){{{first}-{last}}}")
    lines = [
        r"\begin{table}[H]",
        r"\setlength{\tabcolsep}{4pt}",
        r"\centering",
        rf"\caption{{{ALGO_LABELS[algo]} -- Selected hyperparameters}}",
        rf"\label{{tab:hp_selected_{algo}}}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "c" * width + "}",
        r"\toprule",
        " & " + " & ".join(model_headers) + r" \\",
        "".join(rules),
        " & " + " & ".join(variant.upper() for _ in models for variant in variants) + r" \\",
        r"\midrule",
    ]
    for label, key in rows:
        cells = []
        for model in models:
            for variant in variants:
                config = configs[model, variant]
                cells.append("--" if config is None else "$" + _selected_hp_cell(key, config[key]) + "$")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    notes = (r"\textbf{Notes:} MF = model-free variant; PD = post-decision variant. "
             r"Continuous values are rounded to two significant digits. ")
    if algo == "ppo":
        notes += r"For PPO, $\lambda/\lambda_{\text{PD}}$ gives the smoothing parameter of the respective variant. "
    notes += r"Hyperparameters were selected via the procedure in Section~\ref{sec:tuning}."
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{4pt}",
        r"\parbox{\textwidth}{\footnotesize",
        notes,
        r"}",
        r"\end{table}",
    ]
    ensure_parent(save_path)
    Path(save_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
