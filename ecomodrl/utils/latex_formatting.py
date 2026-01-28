import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def generate_latex_table(
    data: Dict[str, Dict[str, float]], 
    algos: List[str], 
    save_path: Path
) -> None:
    """
    Generates a LaTeX table with IQM and 95% CI [lo, hi].
    
    Args:
        data: Dictionary mapping algo_name -> {metric_key: value, ...}
        algos: List of algorithm names to include in the table (defines column order).
        save_path: Full path where the .tex file should be saved.
    """
    
    # Define Metrics configuration
    # (JSON Key, Display Name, Optimization Direction)
    # Direction: 'max' means higher is better, 'min' means lower is better
    row_defs: List[Tuple[str, str, str]] = [
        ("iqm_U_rl", r"Mean Utility ($U$)", "max"),
        ("iqm_cev", r"CEV (Grid)", "min"),
        ("iqm_cev_relevant", r"CEV (Visited States)", "min"),
        ("iqm_max_abs_euler_error_rl", r"Max Euler Error ($\log_{10}$)", "min"),
        ("iqm_wasserstein_dist", r"Wasserstein Dist.", "min"),
        ("iqm_compute_time", r"Compute Time (s)", "min"),
    ]

    # Build LaTeX String
    num_cols = len(algos)
    col_str = "c" * num_cols
    
    latex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.1}", 
        rf"\begin{{tabular}}{{l{col_str}}}",
        r"\toprule",
        r"Metric & " + " & ".join(rf"\textbf{{{algo}}}" for algo in algos) + r" \\",
        r"\midrule"
    ]

    for json_key, name, direction in row_defs:
        # Identify related CI keys (assumes standard naming convention: iqm_X, lo_X, hi_X)
        base_key = json_key.replace("iqm_", "") 
        lo_key = f"lo_{base_key}"
        hi_key = f"hi_{base_key}"

        # --- Pass 1: Find the winner ---
        values = []
        for algo in algos:
            if algo in data and json_key in data[algo]:
                values.append(data[algo][json_key])
        
        best_val: Optional[float] = None
        if values:
            if direction == "max":
                best_val = max(values)
            else:
                best_val = min(values)

        # --- Pass 2: Build Main Row (Mean/IQM) ---
        row_main = f"{name}"
        
        for algo in algos:
            if algo in data and json_key in data[algo]:
                val = data[algo][json_key]
                
                # Math mode for proper minus signs ($...$)
                val_str = f"${val:.4f}$"
                
                # Bold the winner (accounting for float precision)
                if best_val is not None and np.isclose(val, best_val, atol=1e-8):
                    row_main += rf" & \bm{{{val_str}}}"
                else:
                    row_main += rf" & {val_str}"
            else:
                row_main += r" & -"
        row_main += r" \\"
        latex.append(row_main)

        # --- Pass 3: Build Secondary Row (CI) ---
        row_ci = r"" 
        for algo in algos:
            if algo in data and lo_key in data[algo] and hi_key in data[algo]:
                lo = data[algo][lo_key]
                hi = data[algo][hi_key]
                
                # Scriptsize and brackets
                ci_str = f"\\scriptsize{{[{lo:.4f}, {hi:.4f}]}}"
                row_ci += rf" & {ci_str}"
            else:
                row_ci += r" & "
        row_ci += r" \\"
        latex.append(row_ci)
        
        # Add spacing after the pair
        latex.append(r"\addlinespace")

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Performance Comparison (IQM $\pm$ 95\% CI). Best results in \textbf{bold}.}",
        r"\label{tab:rl_comparison}",
        r"\end{table}"
    ])

    # Save to file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))