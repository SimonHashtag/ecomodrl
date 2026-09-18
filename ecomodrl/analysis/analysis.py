# -----------------------------------------------------------------------------
# Post-training analysis for one economic model.
#
# Responsibilities
# ----------------
# 1. Solve the EGM / benchmark policy.
# 2. Build fixed evaluation samples for common-random-number comparisons.
# 3. Evaluate trained RL checkpoints by algorithm, training seed, and test seed.
# 4. Compute Euler-error diagnostics.
# 5. Save summaries for plotting and LaTeX tables.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np

from ecomodrl.algorithms.training import loaded_model_variant, resolve_algo_class
from ecomodrl.configs.variants import (
    COMBINED_FIGURE_DIR_NAME,
    COMBINED_VARIANT,
    DEFAULT_VARIANT,
    VARIANTS,
    VARIANT_LABELS,
    algo_dir_name,
    enforce_variant_constraints,
    figure_dir_name,
)
from ecomodrl.analysis.policy_interface import EGMPolicy, RLPolicy
from ecomodrl.analysis import accuracy_metrics
from ecomodrl.analysis.efficiency_metrics import (
    aulc_per_seed,
    time_to_best_per_seed,
    time_to_converged_per_seed,
)
from ecomodrl.analysis import complexity_metrics
from ecomodrl.analysis.plotting import (
    plot_policy_distribution,
    plot_cev_distribution,
    plot_euler_error_distribution,
    plot_euler_error_distribution_combined,
    plot_training_convergence,
    plot_training_convergence_combined,
    plot_portfolio_share_heatmaps,
    plot_portfolio_share_diff_heatmaps,
    plot_portfolio_share_slices,
)
from ecomodrl.analysis.latex_tables import (
    make_latex_table,
    make_combined_latex_table,
)


# =============================================================================
# Paths
# =============================================================================

def model_results_dir(cfg: Union[Mapping[str, Any], Any]) -> Path:
    """Root directory for one economic model's training and analysis results."""
    return Path(cfg.out_dir) / cfg.model_name


def _variant(cfg: Union[Mapping[str, Any], Any]) -> str:
    """Algorithm variant of the analysis ("mf" | "pd"); selects the *_PD directories."""
    return getattr(cfg, "variant", DEFAULT_VARIANT)


def _load_checkpoint(cfg: Union[Mapping[str, Any], Any], algo: str, model_path: Path, **load_kwargs):
    """Load one SB3 checkpoint with the class of the configured variant and verify the variant."""
    sb3_model = resolve_algo_class(algo, _variant(cfg)).load(str(model_path), **load_kwargs)
    found = loaded_model_variant(sb3_model)
    if found != _variant(cfg):
        raise RuntimeError(
            f"{model_path} was trained with variant {found!r} but the analysis runs with "
            f"variant {_variant(cfg)!r}; pass --variant {found}."
        )
    return sb3_model


def figure_dir(cfg: Union[Mapping[str, Any], Any]) -> Path:
    """Directory for figures and tables ("figures" or "figures_PD")."""
    return model_results_dir(cfg) / figure_dir_name(_variant(cfg))


def combined_figure_dir(cfg: Union[Mapping[str, Any], Any]) -> Path:
    """Directory for figures overlaying both variants ("figures_combined"); variant-independent."""
    return model_results_dir(cfg) / COMBINED_FIGURE_DIR_NAME


def egm_analysis_dir(cfg: Union[Mapping[str, Any], Any]) -> Path:
    """Directory for benchmark-policy analysis output."""
    return model_results_dir(cfg) / "egm_analysis"


def evaluation_sample_path(
    cfg: Union[Mapping[str, Any], Any], 
    test_seed: int,
) -> Path:
    """Path to the fixed common-random-number evaluation sample."""
    return egm_analysis_dir(cfg) / f"evaluation_sample_seed_{int(test_seed)}.pkl"


def checkpoint_dir(
    cfg: Union[Mapping[str, Any], Any], 
    algo: str, 
    train_seed: int,
) -> Path:
    """Directory containing one trained RL checkpoint.

    Training code writes algorithm names in uppercase directories, with a
    "_PD" suffix for the post-decision variant.
    """
    return model_results_dir(cfg) / algo_dir_name(algo, _variant(cfg)) / f"seed_{int(train_seed)}"


def analysis_result_dir(
    cfg: Union[Mapping[str, Any], Any], 
    algo: str, 
    train_seed: int,
) -> Path:
    """Directory for one checkpoint's analysis output."""
    return checkpoint_dir(cfg, algo, train_seed) / "analysis"


def value_grid_path(
    cfg: Union[Mapping[str, Any], Any], 
    algo: str, 
    train_seed: int,
) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "value_grid.npz"


def test_values_path(
    cfg: Union[Mapping[str, Any], Any], 
    algo: str, 
    train_seed: int,
) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "test_values.json"


def euler_errors_path(
    cfg: Union[Mapping[str, Any], Any], 
    algo: str, 
    train_seed: int,
) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "euler_errors.npz"


def visited_states_path(
    cfg: Union[Mapping[str, Any], Any], 
    algo: str, 
    train_seed: int,
) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "visited_states.npz"


def algorithm_summary_path(
    cfg: Union[Mapping[str, Any], Any], 
    algo: str,
) -> Path:
    """Summary path for one algorithm (and variant) and one economic model."""
    return model_results_dir(cfg) / algo_dir_name(algo, _variant(cfg)) / "analysis_summary.json"


# =============================================================================
# Small I/O helpers
# =============================================================================

def read_json(path: Path, default):
    """Read JSON if it exists; otherwise return `default`."""
    if not path.exists():
        return default
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    """Write JSON with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_npz_dict(path: Path) -> dict[str, np.ndarray]:
    """Read an npz file into an owned dictionary of arrays."""
    if not path.exists():
        return {}
    with np.load(path) as d:
        return {k: d[k].copy() for k in d.files}


def write_npz_dict(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a dictionary of arrays to npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def save_value_grid(path: Path, grid, V: np.ndarray) -> None:
    """Save a value function on a model grid.

    Grid convention
    ---------------
    axes = (m_grid, x1_grid, ..., xd_grid)

    Saved keys
    ----------
    V:
        Value array.

    axis_0, axis_1, ...:
        Generic grid axes.

    m_grid:
        Alias for axis_0.

    z_grid:
        Alias for axis_1 when there is exactly one exogenous state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    axes = grid if isinstance(grid, tuple) else (grid,)
    data = {"V": np.asarray(V)}

    for i, axis in enumerate(axes):
        data[f"axis_{i}"] = np.asarray(axis)

    data["m_grid"] = np.asarray(axes[0])

    if len(axes) == 2:
        data["z_grid"] = np.asarray(axes[1])

    np.savez(path, **data)


def load_value_grid(path: Path):
    """Load a value grid saved by `save_value_grid`."""
    with np.load(path) as d:
        V = d["V"].copy()

        axes = []
        i = 0
        while f"axis_{i}" in d.files:
            axes.append(d[f"axis_{i}"].copy())
            i += 1

        if not axes:
            axes = [d["m_grid"].copy()]
            if "z_grid" in d.files:
                axes.append(d["z_grid"].copy())

    return tuple(axes), V


# =============================================================================
# Model and policy helpers
# =============================================================================

def solve_egm_model(cfg: Union[Mapping[str, Any], Any]):
    """Construct and solve the economic model with its configured solver."""
    model = cfg.make_economic_model()
    solver = cfg.make_egm_solver(model)
    solver.solve(verbose=False)

    if hasattr(solver, "policy"):
        model.policy = solver.policy

    return model, solver


def is_deterministic_one_state(model) -> bool:
    """Return True for deterministic one-dimensional models such as CK."""
    return model.weights is None and model.exogenous_dim == 0


def is_portfolio_model(model) -> bool:
    """Return True for models with a safe/risky portfolio choice (PortfolioChoice)."""
    return (
        hasattr(model, "portfolio_weights")
        and getattr(model, "n_risky", 0) >= 1
        and model.exogenous_dim >= 1
    )


def effective_test_seeds(cfg: Union[Mapping[str, Any], Any], model) -> list[int]:
    """Return relevant test seeds.

    Deterministic one-state models do not need repeated test seeds because the
    evaluation samples and Euler-error grids are identical across seeds.
    """
    if is_deterministic_one_state(model):
        return [cfg.test_seeds[0]]
    return list(cfg.test_seeds)


def load_rl_policy(
    cfg: Union[Mapping[str, Any], Any], 
    algo: str, 
    train_seed: int, 
    economic_model,
) -> RLPolicy:
    """Load one trained RL checkpoint as an `RLPolicy`."""
    seed_dir = checkpoint_dir(cfg, algo, train_seed)

    model_path = seed_dir / "model.zip"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    normalize_obs = bool(getattr(cfg, "normalize_obs", True))

    vecnormalize = None

    if normalize_obs:
        vecnorm_path = seed_dir / "vecnormalize.pkl"

        if not vecnorm_path.exists():
            raise FileNotFoundError(
                f"Missing VecNormalize statistics: {vecnorm_path}. "
                "Set --normalize_obs False only if this checkpoint was trained "
                "with norm_obs=False."
            )

        vecnormalize = cfg.load_vecnormalize(vecnorm_path, seed=train_seed)
        sb3_model = _load_checkpoint(cfg, algo, model_path, env=vecnormalize)

    else:
        sb3_model = _load_checkpoint(cfg, algo, model_path)

    return RLPolicy(
        sb3_model=sb3_model,
        model=economic_model,
        vecnormalize=vecnormalize,
    )
    

def _make_FLOPs_measurement_model(cfg: Union[Mapping[str, Any], Any], algo: str):
    """Load the first available checkpoint of `algo` as a THROWAWAY SB3 model
    for offline FLOPs measurement.
    """
    for seed in cfg.seeds:
        model_path = checkpoint_dir(cfg, algo, seed) / "model.zip"
        if model_path.exists():
            env = cfg.make_unnormalized_vecenv(seed)
            return _load_checkpoint(cfg, algo, model_path, env=env)

    raise FileNotFoundError(f"No checkpoint found to measure FLOPs for algo={algo!r}")


def check_required_checkpoints(cfg: Union[Mapping[str, Any], Any]) -> None:
    """Fail early if requested checkpoints are missing."""
    missing = []

    for algo in cfg.algorithms:
        for seed in cfg.seeds:
            model_path = checkpoint_dir(cfg, algo, seed) / "model.zip"
            if not model_path.exists():
                missing.append(str(model_path))

    if missing:
        raise RuntimeError(
            f"Analysis aborted: {len(missing)} trained checkpoints are missing. "
            "First few missing files:\n  "
            + "\n  ".join(missing[:10])
        )


# =============================================================================
# EGM reference objects
# =============================================================================

def prepare_evaluation_samples(
    cfg: Union[Mapping[str, Any], Any], 
    economic_model, 
    egm_policy,
) -> None:
    """Create fixed evaluation samples and EGM reference returns."""
    out_dir = egm_analysis_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = effective_test_seeds(cfg, economic_model)

    egm_returns_path = out_dir / "egm_returns.json"
    egm_returns = {
        int(k): float(v)
        for k, v in read_json(egm_returns_path, {}).items()
    }

    for test_seed in seeds:
        path = evaluation_sample_path(cfg, test_seed)

        if cfg.overwrite or not path.exists():
            sample = accuracy_metrics.make_evaluation_sample(
                economic_model,
                seed=test_seed,
                n_states=cfg.eval_m_grid_size,
                horizon=cfg.eval_horizon,
            )
            with open(path, "wb") as f:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)

        if cfg.overwrite or test_seed not in egm_returns:
            with open(path, "rb") as f:
                sample = pickle.load(f)

            egm_returns[test_seed] = float(
                accuracy_metrics.eval_return_grid(economic_model, egm_policy, sample)
            )

    write_json(
        egm_returns_path,
        {str(k): float(v) for k, v in sorted(egm_returns.items())},
    )


def cache_egm_value_grid(
    cfg: Union[Mapping[str, Any], Any], 
    economic_model: Any, 
    egm_policy: EGMPolicy,
) -> None:
    """Cache the benchmark value function on the model grid."""
    path = egm_analysis_dir(cfg) / "value_grid.npz"

    if path.exists() and not cfg.overwrite:
        return

    grid, V = accuracy_metrics.policy_value_on_grid(
        economic_model, 
        egm_policy,
        tol=1e-5,
    )
    save_value_grid(path, grid, V)


def cache_egm_euler_errors(
    cfg: Union[Mapping[str, Any], Any], 
    economic_model: Any, 
    egm_policy: EGMPolicy,
) -> None:
    """Cache benchmark Euler-error reference distributions."""
    out_dir = egm_analysis_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_deterministic_one_state(economic_model):
        path = out_dir / "euler_errors.npy"

        if path.exists() and not cfg.overwrite:
            return

        ee = accuracy_metrics.euler_errors(
            economic_model,
            egm_policy,
            n_sim=cfg.euler_n_states,
        )
        np.save(path, np.asarray(ee, dtype=np.float64))
        return

    for test_seed in effective_test_seeds(cfg, economic_model):
        path = out_dir / f"egm_euler_errors_{test_seed}.npy"

        if path.exists() and not cfg.overwrite:
            continue

        ee = accuracy_metrics.euler_errors(
            economic_model,
            egm_policy,
            n_sim=cfg.euler_n_states,
            seed=test_seed,
        )
        np.save(path, np.asarray(ee, dtype=np.float64))


def cache_train_egm_returns(
    cfg: Union[Mapping[str, Any], Any], 
    economic_model: Any, 
    egm_policy: EGMPolicy,
) -> None:
    """Cache EGM returns on training-time evaluation samples.

    The convergence plot compares checkpoint returns to EGM returns on the same
    common-random-number sample used during training-time evaluation.
    """
    path = egm_analysis_dir(cfg) / "train_egm_returns.json"
    existing = {
        int(k): float(v)
        for k, v in read_json(path, {}).items()
    }

    missing = [s for s in cfg.seeds if cfg.overwrite or s not in existing]

    for train_seed in missing:
        sample = accuracy_metrics.make_evaluation_sample(
            economic_model,
            seed=train_seed + cfg.eval_seed_offset,
            n_states=cfg.eval_m_grid_size,
            horizon=cfg.eval_horizon,
        )
        existing[train_seed] = float(
            accuracy_metrics.eval_return_grid(economic_model, egm_policy, sample)
        )

    write_json(path, {str(k): float(v) for k, v in sorted(existing.items())})


def load_egm_euler_reference(
    cfg: Union[Mapping[str, Any], Any], 
    economic_model: Any, 
    test_seed: int | None
) -> np.ndarray | None:
    """Load cached EGM Euler-error reference."""
    if is_deterministic_one_state(economic_model):
        path = egm_analysis_dir(cfg) / "euler_errors.npy"
        return np.load(path) if path.exists() else None

    path = egm_analysis_dir(cfg) / f"egm_euler_errors_{int(test_seed)}.npy"
    return np.load(path) if path.exists() else None


# =============================================================================
# Checkpoint analysis
# =============================================================================

def analyze_checkpoint_worker(args):
    """Worker entry point for one `(algorithm, training seed)` checkpoint."""
    import torch

    torch.set_num_threads(1)
    
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    cfg, algo, train_seed = args

    economic_model = cfg.make_economic_model()
    seeds = effective_test_seeds(cfg, economic_model)

    samples = {}
    for test_seed in seeds:
        with open(evaluation_sample_path(cfg, test_seed), "rb") as f:
            samples[test_seed] = pickle.load(f)

    egm_returns = {
        int(k): float(v)
        for k, v in read_json(egm_analysis_dir(cfg) / "egm_returns.json", {}).items()
    }

    try:
        analyze_policy_checkpoint(
            cfg=cfg,
            economic_model=economic_model,
            algo=algo,
            train_seed=train_seed,
            test_samples=samples,
            egm_returns=egm_returns,
        )
        status = "ok"
    except FileNotFoundError:
        status = "missing"
    except Exception as exc:
        status = repr(exc)

    return algo, int(train_seed), status


def analyze_policy_checkpoint(
    *,
    cfg: Union[Mapping[str, Any], Any],
    economic_model: Any,
    algo: str,
    train_seed: int,
    test_samples: dict[int, object],
    egm_returns: dict[int, float],
) -> None:
    """Analyze one trained RL policy checkpoint."""
    policy = load_rl_policy(cfg, algo, train_seed, economic_model)

    out_dir = analysis_result_dir(cfg, algo, train_seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Grid Bellman value for CEV maps.
    vg_path = value_grid_path(cfg, algo, train_seed)
    if cfg.overwrite or not vg_path.exists():
        grid, V = accuracy_metrics.policy_value_on_grid(economic_model, policy, tol=1e-5)
        save_value_grid(vg_path, grid, V)

    rl_axes, V_rl_grid = load_value_grid(vg_path)
    _, V_egm_grid = load_value_grid(egm_analysis_dir(cfg) / "value_grid.npz")

    # 2. Test returns, CEV regret, Euler errors, and visited states.
    values_path = test_values_path(cfg, algo, train_seed)
    values = read_json(values_path, {})

    eulers_path = euler_errors_path(cfg, algo, train_seed)
    eulers = read_npz_dict(eulers_path)

    visited_path = visited_states_path(cfg, algo, train_seed)
    visited = read_npz_dict(visited_path)

    changed_values = False
    changed_eulers = False
    changed_visited = False

    for test_seed in test_samples:
        key = str(test_seed)
        array_key = f"seed_{test_seed}"

        need_values = cfg.overwrite or key not in values
        need_eulers = cfg.overwrite or array_key not in eulers
        need_visited = cfg.overwrite or array_key not in visited

        states_full = None
        if need_values or need_visited:
            states_full = np.asarray(
                accuracy_metrics.simulate_states(
                    economic_model,
                    policy,
                    n_sim=cfg.euler_n_states,
                    seed=test_seed,
                ),
                dtype=np.float64,
            )
            if (
                states_full.ndim != 2
                or states_full.shape[1] != economic_model.state_dim
            ):
                raise ValueError(
                    "Simulated visited states must have shape "
                    f"(n, {economic_model.state_dim}); got {states_full.shape}."
                )

        if need_values:
            u_rl = float(
                accuracy_metrics.eval_return_grid(
                    economic_model, policy, test_samples[test_seed],
                )
            )
            u_egm = float(egm_returns[test_seed])

            cev_regret = float(
                accuracy_metrics.cev_on_states(
                    economic_model, rl_axes, V_rl_grid, V_egm_grid, states_full,
                )
            )

            values[key] = {
                "u_rl": u_rl,
                "u_egm": u_egm,
                "cev_regret": cev_regret,
            }
            changed_values = True

        if need_eulers:
            ee = accuracy_metrics.euler_errors(
                economic_model,
                policy,
                n_sim=cfg.euler_n_states,
                seed=test_seed,
            )
            eulers[array_key] = np.asarray(ee, dtype=np.float64)
            changed_eulers = True

        if need_visited:
            visited[array_key] = states_full
            changed_visited = True

    if changed_values:
        write_json(values_path, values)

    if changed_eulers:
        write_npz_dict(eulers_path, eulers)

    if changed_visited:
        write_npz_dict(visited_path, visited)


# =============================================================================
# Summaries
# =============================================================================


def summarize_matrix(matrix: np.ndarray) -> dict:
    """Return descriptive statistics and crossed-bootstrap IQM interval."""
    matrix = np.asarray(matrix, dtype=np.float64)
    finite = matrix[np.isfinite(matrix)]

    if finite.size == 0:
        return {"n": 0}

    iqm, lo, hi = accuracy_metrics.iqm_ci_crossed(matrix)

    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "q1": float(np.percentile(finite, 25)),
        "q3": float(np.percentile(finite, 75)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "iqm": float(iqm),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def summarize_algorithm(
    cfg: Union[Mapping[str, Any], Any], 
    economic_model: Any, 
    algo: str,
) -> None:
    """Aggregate train-seed × test-seed cells for one algorithm."""
    active_test_seeds = effective_test_seeds(cfg, economic_model)
    shape = (len(cfg.seeds), len(active_test_seeds))

    matrices = {
        "u_rl": np.full(shape, np.nan),
        "cev_regret": np.full(shape, np.nan),
        "ee_mean": np.full(shape, np.nan),
        "ee_p95": np.full(shape, np.nan),
        "wasserstein": np.full(shape, np.nan),
    }

    egm_euler_refs = {
        ts: load_egm_euler_reference(
            cfg,
            economic_model,
            None if is_deterministic_one_state(economic_model) else ts,
        )
        for ts in active_test_seeds
    }

    n_cells = 0

    for i, train_seed in enumerate(cfg.seeds):
        values = read_json(test_values_path(cfg, algo, train_seed), {})
        eulers = read_npz_dict(euler_errors_path(cfg, algo, train_seed))

        for j, test_seed in enumerate(active_test_seeds):
            key = str(test_seed)
            array_key = f"seed_{test_seed}"

            if key not in values or array_key not in eulers:
                continue

            d = values[key]
            ee = np.asarray(eulers[array_key], dtype=np.float64)
            ee = ee[np.isfinite(ee)]

            matrices["u_rl"][i, j] = d.get("u_rl", np.nan)
            matrices["cev_regret"][i, j] = d.get("cev_regret", np.nan)

            if ee.size:
                matrices["ee_mean"][i, j] = float(np.mean(ee))
                matrices["ee_p95"][i, j] = float(np.percentile(ee, 95))

            ref = egm_euler_refs[test_seed]
            if ref is not None:
                matrices["wasserstein"][i, j] = accuracy_metrics.wasserstein_euler(ee, ref)

            n_cells += 1

    summary = {
        "model": cfg.model_name,
        "algorithm": algo,
        "variant": _variant(cfg),
        "n_cells": int(n_cells),
    }

    for name, matrix in matrices.items():
        summary[name] = summarize_matrix(matrix)

    write_json(algorithm_summary_path(cfg, algo), summary)
    
    
def efficiency_summary(
    histories_by_algo: dict[str, list],
    egm_returns_by_algo: dict[str, list],
    wallclock_seconds: float,
    grid_step: float,
    *,
    algorithms: list[str],
    conv_tol: float = 0.05,
):
    """Efficiency metrics: AULC, time-to-best, time-to-converged.

    Each metric is aggregated over training seeds with IQM + bootstrap CI.
    """
    t_grid = np.arange(0.0, float(wallclock_seconds) + 1e-9, float(grid_step))

    summary = {}
    for algo in algorithms:
        histories = histories_by_algo.get(algo, [])
        egm_returns = egm_returns_by_algo.get(algo, [])

        entry = {}

        if not histories or len(histories) != len(egm_returns):
            entry["aulc"] = {"n": 0}
            entry["time_to_best"] = {"n": 0}
            entry["time_to_converged"] = {"n": 0}
            summary[algo] = entry
            continue

        entry["aulc"] = summarize_matrix(
            aulc_per_seed(histories, egm_returns, t_grid)
        )
        entry["time_to_best"] = summarize_matrix(
            time_to_best_per_seed(histories)
        )
        entry["time_to_converged"] = summarize_matrix(
            time_to_converged_per_seed(histories, egm_returns, conv_tol)
        )

        summary[algo] = entry

    return summary


def complexity_summary(
    cfg: Union[Mapping[str, Any], Any],
    histories_by_algo: dict[str, list],
    egm_returns_by_algo: dict[str, list],
    *,
    algorithms: list[str],
    tol: float,
):
    """FLOPs complexity per algorithm.

    - flops_per_update : deterministic point value (single measurement).
    - flops_to_conv    : per-seed cumulative FLOPs, IQM + bootstrap CI.
    - sampled_states   : per-seed sampled environment states, IQM + bootstrap CI.
    """
    summary = {}

    for algo in algorithms:
        print(f"[complexity] measuring {algo}...", flush=True)
        entry = {
            "flops_per_update": {"n": 0},
            "flops_to_conv": {"n": 0},
            "sampled_states": {"n": 0},
        }

        try:
            model = _make_FLOPs_measurement_model(cfg, algo)
        except Exception as exc:
            print(f"[complexity] {algo}: no checkpoint to measure ({exc!r})", flush=True)
            summary[algo] = entry
            continue

        env = None
        try:
            env = model.get_env()
            flops = complexity_metrics.measure_algo_flops(
                model,
                algo,
                n_envs_train=int(getattr(cfg, "n_envs", 1)),
                forward_env_steps=getattr(cfg, "flops_forward_steps", None),
            )
        except Exception as exc:
            print(f"[complexity] {algo}: measurement failed ({exc!r})", flush=True)
            summary[algo] = entry
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            continue

        try:
            if env is not None:
                env.close()
        except Exception:
            pass

        entry["flops_per_update"] = {
            "n": 1,
            "iqm": float(flops["f_per_update"]),
        }
        print(
            f"[complexity] {algo}: measured {flops['f_per_update']:.3e} FLOPs/update",
            flush=True,
        )

        histories = histories_by_algo.get(algo, [])
        egm = egm_returns_by_algo.get(algo, [])

        if histories and len(histories) == len(egm):
            cumflops, states = complexity_metrics.cumulative_flops_per_seed(
                histories,
                egm,
                flops=flops,
                n_envs_train=int(getattr(cfg, "n_envs", 1)),
                tol=tol,
            )
            entry["flops_to_conv"] = summarize_matrix(np.asarray(cumflops, dtype=np.float64))
            entry["sampled_states"] = summarize_matrix(np.asarray(states, dtype=np.float64))

        summary[algo] = entry
    
    return summary


# =============================================================================
# Main analysis
# =============================================================================

def run_testing(cfg: Union[Mapping[str, Any], Any]) -> None:
    """Run numerical post-training analysis."""
    print(f"\n========== Analysis: {cfg.model_name} ({_variant(cfg)}) ==========")
    print(f"Algorithms: {', '.join(cfg.algorithms)}")

    check_required_checkpoints(cfg)

    economic_model, _ = solve_egm_model(cfg)
    egm_policy = EGMPolicy(economic_model)

    cache_egm_value_grid(cfg, economic_model, egm_policy)
    prepare_evaluation_samples(cfg, economic_model, egm_policy)
    cache_egm_euler_errors(cfg, economic_model, egm_policy)
    cache_train_egm_returns(cfg, economic_model, egm_policy)

    units = [(cfg, algo, seed) for algo in cfg.algorithms for seed in cfg.seeds]

    worker_count = cfg.max_workers
    if worker_count is None:
        worker_count = max(1, (os.cpu_count() or 1) - 1)
    worker_count = min(worker_count, len(units))

    print(f"[analysis] {len(units)} checkpoint units; workers={worker_count}")

    if worker_count == 1:
        for unit in units:
            algo, train_seed, status = analyze_checkpoint_worker(unit)
            print(f"[analysis] {algo} seed={train_seed}: {status}", flush=True)
    else:
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
            futures = {pool.submit(analyze_checkpoint_worker, u): u for u in units}

            done = 0
            for fut in as_completed(futures):
                _, algo, seed = futures[fut]
                done += 1

                try:
                    algo_out, train_seed, status = fut.result()
                    print(
                        f"[analysis] {algo_out} seed={train_seed}: {status} "
                        f"({done}/{len(units)})",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[analysis] FAILED {algo} seed={seed}: {exc!r} "
                        f"({done}/{len(units)})",
                        flush=True,
                    )

    economic_model, _ = solve_egm_model(cfg)

    for algo in cfg.algorithms:
        summarize_algorithm(cfg, economic_model, algo)
        print(f"[analysis] wrote summary for {cfg.model_name}/{algo}")


# =============================================================================
# Plotting
# =============================================================================

def load_training_history(seed_dir: Path):
    """Load training-time evaluation history and per-eval compute counters.

    Reads eval_history.npz and returns a dict:
        t       : evaluation wall-clock times
        u       : evaluation returns
        steps   : cumulative environment steps per eval (None if not logged)
        updates : cumulative gradient updates per eval (None if not logged)

    Returns None if no usable history file is present.
    """
    path = seed_dir / "eval_history.npz"
    if not path.exists():
        return None

    with np.load(path) as d:
        if "history_t" not in d.files or "history_U" not in d.files:
            return None

        return {
            "t": d["history_t"].copy(),
            "u": d["history_U"].copy(),
            "steps": d["history_steps"].copy() if "history_steps" in d.files else None,
            "updates": d["history_updates"].copy() if "history_updates" in d.files else None,
        }


def run_plots_and_tables(cfg: Union[Mapping[str, Any], Any]) -> None:
    """Create figures and LaTeX tables from saved analysis output."""
    print(f"\n========== Figures: {cfg.model_name} ({_variant(cfg)}) ==========")

    economic_model, _ = solve_egm_model(cfg)
    egm_policy = EGMPolicy(economic_model)

    axes_egm, V_egm = load_value_grid(egm_analysis_dir(cfg) / "value_grid.npz")
    deterministic = is_deterministic_one_state(economic_model)
    active_test_seeds = effective_test_seeds(cfg, economic_model)

    rl_policies = {algo: [] for algo in cfg.algorithms}
    rl_value_grids = {algo: [] for algo in cfg.algorithms}
    histories = {algo: [] for algo in cfg.algorithms}
    history_seeds = {algo: [] for algo in cfg.algorithms}
    euler = {algo: [] for algo in cfg.algorithms}
    visited_pools = {algo: [] for algo in cfg.algorithms}

    for algo in cfg.algorithms:
        for seed in cfg.seeds:
            seed_dir = checkpoint_dir(cfg, algo, seed)

            try:
                rl_policies[algo].append(
                    load_rl_policy(cfg, algo, seed, economic_model)
                )
            except Exception:
                continue

            vg_path = value_grid_path(cfg, algo, seed)
            if vg_path.exists():
                _, V = load_value_grid(vg_path)
                rl_value_grids[algo].append(V)

            hist = load_training_history(seed_dir)
            if hist is not None:
                histories[algo].append(hist)
                history_seeds[algo].append(seed)

            eulers = read_npz_dict(euler_errors_path(cfg, algo, seed))
            visited = read_npz_dict(visited_states_path(cfg, algo, seed))

            got_deterministic = False

            for test_seed in active_test_seeds:
                key = f"seed_{test_seed}"

                if key in eulers:
                    if not got_deterministic:
                        euler[algo].append(eulers[key])

                    if deterministic:
                        got_deterministic = True

                if key in visited:
                    visited_pools[algo].append(visited[key])

    train_egm_returns = {
        int(k): float(v)
        for k, v in read_json(egm_analysis_dir(cfg) / "train_egm_returns.json", {}).items()
    }

    egm_returns_by_algo = {
        algo: [train_egm_returns.get(seed, np.nan) for seed in history_seeds[algo]]
        for algo in cfg.algorithms
    }

    pooled_visited_states = {}
    for algo, arrays in visited_pools.items():
        if not arrays:
            continue
        pooled_visited_states[algo] = np.concatenate(
            [np.asarray(x, dtype=np.float64) for x in arrays],
            axis=0,
        )

    pooled_visited_m = {
        algo: states[:, 0]
        for algo, states in pooled_visited_states.items()
    }

    out = figure_dir(cfg)

    plot_policy_distribution(
        economic_model,
        egm_policy,
        rl_policies,
        out / "policy.pdf",
        algorithms=cfg.algorithms,
    )

    plot_cev_distribution(
        economic_model,
        axes_egm,
        V_egm,
        rl_value_grids,
        out / "cev.pdf",
        algorithms=cfg.algorithms,
        visited_m=pooled_visited_m,
        support_quantiles=(0.1, 99.9),
    )

    plot_euler_error_distribution(
        economic_model,
        euler,
        out / "euler_errors.pdf",
        deterministic,
        algorithms=cfg.algorithms,
    )

    # -------------------------------------------------------------------------
    # Portfolio-share figures
    # -------------------------------------------------------------------------
    if is_portfolio_model(economic_model):
        plot_portfolio_share_heatmaps(
            economic_model,
            rl_policies,
            out / "portfolio_share.pdf",
            algorithms=cfg.algorithms,
        )

        plot_portfolio_share_diff_heatmaps(
            economic_model,
            egm_policy,
            rl_policies,
            out / "portfolio_share_diff.pdf",
            algorithms=cfg.algorithms,
            states=pooled_visited_states,
        )

        plot_portfolio_share_slices(
            economic_model,
            egm_policy,
            rl_policies,
            out / "portfolio_share_slices.pdf",
            algorithms=cfg.algorithms,
        )

        print("[figures] portfolio-share plots saved")

    # -------------------------------------------------------------------------
    # Efficiency metrics: AULC and time-to-convergence
    # -------------------------------------------------------------------------
    efficiency = efficiency_summary(
        histories,
        egm_returns_by_algo,
        cfg.max_wallclock_seconds,
        cfg.eval_interval_seconds,
        algorithms=cfg.algorithms,
        conv_tol=getattr(cfg, "conv_tol", 0.0001),
    )
    write_json(figure_dir(cfg) / "efficiency_summary.json", efficiency)

    plot_training_convergence(
        histories,
        egm_returns_by_algo,
        cfg.max_wallclock_seconds,
        out / "convergence.pdf",
        algorithms=cfg.algorithms,
        grid_step=cfg.eval_interval_seconds,
        efficiency=efficiency,
    )

    summaries = {}
    for algo in cfg.algorithms:
        path = algorithm_summary_path(cfg, algo)
        if path.exists():
            summaries[algo] = read_json(path, {})

    # Merge efficiency stats into the per-algorithm summaries for the table.
    for algo in cfg.algorithms:
        summaries.setdefault(algo, {})
        for key in ("aulc", "time_to_best", "time_to_converged"):
            if key in efficiency.get(algo, {}):
                summaries[algo][key] = efficiency[algo][key]

    # -------------------------------------------------------------------------
    # Complexity metrics: FLOPs per update and cumulative FLOPs to convergence.
    # -------------------------------------------------------------------------
    try:
        complexity = complexity_summary(
            cfg,
            histories,
            egm_returns_by_algo,
            algorithms=cfg.algorithms,
            tol=getattr(cfg, "conv_tol", 0.0001),
        )
    except Exception as exc:
        print(f"[figures] complexity metrics skipped: {exc!r}")
        complexity = {}

    write_json(figure_dir(cfg) / "complexity_summary.json", complexity)

    for algo in cfg.algorithms:
        summaries.setdefault(algo, {})
        for key in ("flops_per_update", "flops_to_conv", "sampled_states"):
            if key in complexity.get(algo, {}):
                summaries[algo][key] = complexity[algo][key]

    make_latex_table(
        economic_model,
        summaries,
        out / "table.tex",
        algorithms=cfg.algorithms,
    )

    make_combined_latex_table(
        [(cfg.model_name, summaries)],
        out / f"{cfg.model_name}_comparison_table.tex",
        algorithms=cfg.algorithms,
    )

    print(f"[figures] saved -> {out}", flush=True)


# =============================================================================
# Combined mf-vs-pd figures
# =============================================================================

def load_variant_figure_inputs(
    cfg: Union[Mapping[str, Any], Any],
    economic_model: Any,
) -> dict:
    """Inputs of the combined figures for one variant, read from saved analysis output.

    `cfg` must be the per-variant config (variant "mf" or "pd"). Loads no
    checkpoints and never raises. Returns a dict with
        histories   : {algo: [history dict]}      (load_training_history)
        egm_returns : {algo: [float]}             aligned with histories via train_egm_returns.json
        euler       : {algo: [1-D log10 arrays]}  same selection rule as run_plots_and_tables
        skipped     : [(algo, seed, what)]        units that could not be read
        n_units     : number of histories and Euler arrays found
    """
    deterministic = is_deterministic_one_state(economic_model)
    active_test_seeds = effective_test_seeds(cfg, economic_model)

    histories = {algo: [] for algo in cfg.algorithms}
    history_seeds = {algo: [] for algo in cfg.algorithms}
    euler = {algo: [] for algo in cfg.algorithms}
    skipped: list[tuple[str, int, str]] = []

    for algo in cfg.algorithms:
        for seed in cfg.seeds:
            seed_dir = checkpoint_dir(cfg, algo, seed)

            # Same seed selection as run_plots_and_tables, which skips seeds
            # whose checkpoint cannot be loaded.
            if not (seed_dir / "model.zip").exists():
                skipped.append((algo, seed, "model.zip"))
                continue

            hist = load_training_history(seed_dir)
            if hist is not None:
                histories[algo].append(hist)
                history_seeds[algo].append(seed)
            else:
                skipped.append((algo, seed, "eval_history.npz"))

            eulers = read_npz_dict(euler_errors_path(cfg, algo, seed))
            n_found = 0
            for test_seed in active_test_seeds:
                key = f"seed_{test_seed}"
                if key in eulers:
                    euler[algo].append(eulers[key])
                    n_found += 1
                    if deterministic:
                        break
            if n_found == 0:
                skipped.append((algo, seed, "analysis/euler_errors.npz"))

    train_egm_returns = {
        int(k): float(v)
        for k, v in read_json(egm_analysis_dir(cfg) / "train_egm_returns.json", {}).items()
    }

    egm_returns = {
        algo: [train_egm_returns.get(seed, np.nan) for seed in history_seeds[algo]]
        for algo in cfg.algorithms
    }

    n_units = sum(len(v) for v in histories.values()) + sum(len(v) for v in euler.values())

    return {
        "histories": histories,
        "egm_returns": egm_returns,
        "euler": euler,
        "skipped": skipped,
        "n_units": n_units,
    }


def run_combined_figures(cfg: Union[Mapping[str, Any], Any]) -> None:
    """Write <Env>/figures_combined/{euler_errors,convergence}.pdf overlaying mf and pd.

    Reads the per-variant analysis output of both variants; runs no testing and
    loads no checkpoints. Missing units are skipped with a message, a missing
    variant is skipped with a warning.
    """
    labels = " vs ".join(VARIANT_LABELS[v] for v in VARIANTS)
    print(f"\n========== Combined figures: {cfg.model_name} ({labels}) ==========")

    economic_model = cfg.make_economic_model()
    deterministic = is_deterministic_one_state(economic_model)

    egm_returns_path = egm_analysis_dir(cfg) / "train_egm_returns.json"
    if not egm_returns_path.exists():
        print(
            f"[combined] WARNING: {egm_returns_path} is missing; convergence curves will be "
            "empty (run the per-variant analysis first)."
        )

    inputs = {}
    for variant in VARIANTS:
        cfg_v = enforce_variant_constraints(replace(cfg, variant=variant))
        data = load_variant_figure_inputs(cfg_v, economic_model)
        label = VARIANT_LABELS[variant]

        for algo, seed, what in data["skipped"]:
            print(f"[combined] {label}: skipped {algo.upper()} seed={seed} (missing {what})")

        if data["n_units"] == 0:
            print(
                f"[combined] WARNING: no {label} results under {model_results_dir(cfg)} "
                f"({algo_dir_name('<ALGO>', variant)}/seed_*/); drawing the other variant only."
            )
            continue

        inputs[variant] = data

    if not inputs:
        print(
            f"[combined] WARNING: nothing to plot for {cfg.model_name}; "
            "run --variant mf and --variant pd first."
        )
        return

    out = combined_figure_dir(cfg)

    plot_euler_error_distribution_combined(
        economic_model,
        {variant: data["euler"] for variant, data in inputs.items()},
        out / "euler_errors.pdf",
        deterministic,
        algorithms=cfg.algorithms,
    )

    plot_training_convergence_combined(
        {variant: data["histories"] for variant, data in inputs.items()},
        {variant: data["egm_returns"] for variant, data in inputs.items()},
        cfg.max_wallclock_seconds,
        out / "convergence.pdf",
        algorithms=cfg.algorithms,
        grid_step=cfg.eval_interval_seconds,
    )

    print(f"[figures] saved -> {out}")


# =============================================================================
# Entry point
# =============================================================================

def run_analysis(cfg: Union[Mapping[str, Any], Any]) -> None:
    """Run requested analysis stages."""
    if _variant(cfg) == COMBINED_VARIANT:
        run_combined_figures(cfg)
        return

    if cfg.run_testing:
        run_testing(cfg)

    if cfg.make_plots_and_tables:
        run_plots_and_tables(cfg)
