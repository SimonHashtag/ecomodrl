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
import multiprocessing as mp
from pathlib import Path

import numpy as np

from ecomodrl.algorithms.training import ALGO_REGISTRY
from ecomodrl.analysis.policy_interface import EGMPolicy, RLPolicy
from ecomodrl.analysis import accuracy_metrics as metrics
from ecomodrl.analysis.plotting import (
    plot_policy_distribution,
    plot_cev_distribution,
    plot_euler_error_distribution,
    plot_training_convergence,
)
from ecomodrl.analysis.latex_tables import (
    make_latex_table,
    make_combined_latex_table,
)


# =============================================================================
# Paths
# =============================================================================

def model_results_dir(cfg) -> Path:
    """Root directory for one economic model's training and analysis results."""
    return Path(cfg.out_dir) / cfg.model_name


def figure_dir(cfg) -> Path:
    """Directory for figures and tables."""
    return model_results_dir(cfg) / "figures"


def egm_analysis_dir(cfg) -> Path:
    """Directory for benchmark-policy analysis output."""
    return model_results_dir(cfg) / "egm_analysis"


def evaluation_sample_path(cfg, test_seed: int) -> Path:
    """Path to the fixed common-random-number evaluation sample."""
    return egm_analysis_dir(cfg) / f"evaluation_sample_seed_{int(test_seed)}.pkl"


def checkpoint_dir(cfg, algo: str, train_seed: int) -> Path:
    """Directory containing one trained RL checkpoint.

    Training code writes algorithm names in uppercase directories.
    """
    return model_results_dir(cfg) / algo.upper() / f"seed_{int(train_seed)}"


def analysis_result_dir(cfg, algo: str, train_seed: int) -> Path:
    """Directory for one checkpoint's analysis output."""
    return checkpoint_dir(cfg, algo, train_seed) / "analysis"


def value_grid_path(cfg, algo: str, train_seed: int) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "value_grid.npz"


def test_values_path(cfg, algo: str, train_seed: int) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "test_values.json"


def euler_errors_path(cfg, algo: str, train_seed: int) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "euler_errors.npz"


def visited_states_path(cfg, algo: str, train_seed: int) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "visited_m.npz"


def algorithm_summary_path(cfg, algo: str) -> Path:
    """Summary path for one algorithm and one economic model."""
    return model_results_dir(cfg) / algo.upper() / "analysis_summary.json"


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

def solve_egm_model(cfg):
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


def effective_test_seeds(cfg, model) -> list[int]:
    """Return relevant test seeds.

    Deterministic one-state models do not need repeated test seeds because the
    evaluation samples and Euler-error grids are identical across seeds.
    """
    if is_deterministic_one_state(model):
        return [cfg.test_seeds[0]]
    return list(cfg.test_seeds)


def load_rl_policy(cfg, algo: str, train_seed: int, economic_model):
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
        sb3_model = ALGO_REGISTRY[algo].load(
            str(model_path),
            env=vecnormalize,
        )

    else:
        sb3_model = ALGO_REGISTRY[algo].load(str(model_path))

    return RLPolicy(
        sb3_model=sb3_model,
        model=economic_model,
        vecnormalize=vecnormalize,
    )


def check_required_checkpoints(cfg) -> None:
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

def prepare_evaluation_samples(cfg, economic_model, egm_policy) -> None:
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
            sample = metrics.make_evaluation_sample(
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
                metrics.eval_return_grid(economic_model, egm_policy, sample)
            )

    write_json(
        egm_returns_path,
        {str(k): float(v) for k, v in sorted(egm_returns.items())},
    )


def cache_egm_value_grid(cfg, economic_model, egm_policy) -> None:
    """Cache the benchmark value function on the model grid."""
    path = egm_analysis_dir(cfg) / "value_grid.npz"

    if path.exists() and not cfg.overwrite:
        return

    grid, V = metrics.policy_value_on_grid(
        economic_model, 
        egm_policy,
    )
    save_value_grid(path, grid, V)


def cache_egm_euler_errors(cfg, economic_model, egm_policy) -> None:
    """Cache benchmark Euler-error reference distributions."""
    out_dir = egm_analysis_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_deterministic_one_state(economic_model):
        path = out_dir / "euler_errors.npy"

        if path.exists() and not cfg.overwrite:
            return

        ee = metrics.euler_errors(
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

        ee = metrics.euler_errors(
            economic_model,
            egm_policy,
            n_sim=cfg.euler_n_states,
            seed=test_seed,
        )
        np.save(path, np.asarray(ee, dtype=np.float64))


def cache_train_egm_returns(cfg, economic_model, egm_policy) -> None:
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
        sample = metrics.make_evaluation_sample(
            economic_model,
            seed=train_seed + cfg.eval_seed_offset,
            n_states=cfg.eval_m_grid_size,
            horizon=cfg.eval_horizon,
        )
        existing[train_seed] = float(
            metrics.eval_return_grid(economic_model, egm_policy, sample)
        )

    write_json(path, {str(k): float(v) for k, v in sorted(existing.items())})


def load_egm_euler_reference(cfg, economic_model, test_seed: int | None):
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
    deterministic = is_deterministic_one_state(economic_model)
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
            deterministic=deterministic,
        )
        status = "ok"
    except FileNotFoundError:
        status = "missing"
    except Exception as exc:
        status = repr(exc)

    return algo, int(train_seed), status


def analyze_policy_checkpoint(
    *,
    cfg,
    economic_model,
    algo: str,
    train_seed: int,
    test_samples: dict[int, object],
    egm_returns: dict[int, float],
    deterministic: bool,
) -> None:
    """Analyze one trained RL policy checkpoint."""
    policy = load_rl_policy(cfg, algo, train_seed, economic_model)

    out_dir = analysis_result_dir(cfg, algo, train_seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Grid Bellman value for CEV maps.
    vg_path = value_grid_path(cfg, algo, train_seed)
    if cfg.overwrite or not vg_path.exists():
        grid, V = metrics.policy_value_on_grid(
            economic_model, 
            policy,
        )
        save_value_grid(vg_path, grid, V)

    # 2. Test returns, CEV regret, Euler errors, and visited states.
    values_path = test_values_path(cfg, algo, train_seed)
    values = read_json(values_path, {})

    eulers_path = euler_errors_path(cfg, algo, train_seed)
    eulers = read_npz_dict(eulers_path)

    visited_path = visited_states_path(cfg, algo, train_seed)
    visited = read_npz_dict(visited_path) if not deterministic else {}

    changed_values = False
    changed_eulers = False
    changed_visited = False

    for test_seed in test_samples:
        key = str(test_seed)
        array_key = f"seed_{test_seed}"

        if cfg.overwrite or key not in values:
            u_rl = float(
                metrics.eval_return_grid(
                    economic_model,
                    policy,
                    test_samples[test_seed],
                )
            )
            u_egm = float(egm_returns[test_seed])

            if np.isfinite(u_rl) and np.isfinite(u_egm):
                cev_regret = float(metrics.cev(economic_model, u_rl, u_egm))
            else:
                cev_regret = float("nan")

            values[key] = {
                "u_rl": u_rl,
                "u_egm": u_egm,
                "cev_regret": cev_regret,
            }
            changed_values = True

        if cfg.overwrite or array_key not in eulers:
            ee = metrics.euler_errors(
                economic_model,
                policy,
                n_sim=cfg.euler_n_states,
                seed=test_seed,
            )
            eulers[array_key] = np.asarray(ee, dtype=np.float64)
            changed_eulers = True

        if not deterministic and (cfg.overwrite or array_key not in visited):
            states = metrics.simulate_states(
                economic_model,
                policy,
                n_sim=cfg.euler_n_states,
                seed=test_seed,
            )
            visited[array_key] = np.asarray(states[..., 0], dtype=np.float64)
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

def summarize_algorithm(cfg, economic_model, algo: str) -> None:
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
                matrices["wasserstein"][i, j] = metrics.wasserstein_euler(ee, ref)

            n_cells += 1

    summary = {
        "model": cfg.model_name,
        "algorithm": algo,
        "n_cells": int(n_cells),
    }

    for name, matrix in matrices.items():
        summary[name] = summarize_matrix(matrix)

    write_json(algorithm_summary_path(cfg, algo), summary)


def summarize_matrix(matrix: np.ndarray) -> dict:
    """Return descriptive statistics and crossed-bootstrap IQM interval."""
    matrix = np.asarray(matrix, dtype=np.float64)
    finite = matrix[np.isfinite(matrix)]

    if finite.size == 0:
        return {"n": 0}

    iqm, lo, hi = metrics.iqm_ci_crossed(matrix)

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


# =============================================================================
# Main analysis
# =============================================================================

def run_testing(cfg) -> None:
    """Run numerical post-training analysis."""
    print(f"\n========== Analysis: {cfg.model_name} ==========")
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
    """Load training-time evaluation history from current or legacy files."""
    candidates = [
        seed_dir / "eval_history.npz",
        seed_dir / "history.npz",
    ]

    for path in candidates:
        if not path.exists():
            continue

        with np.load(path) as d:
            if "history_t" in d and "history_U" in d:
                return d["history_t"].copy(), d["history_U"].copy()
            if "history_t" in d and "history_V" in d:
                return d["history_t"].copy(), d["history_V"].copy()
            if "t" in d and "V_grid" in d:
                return d["t"].copy(), d["V_grid"].copy()

    return None


def run_plots_and_tables(cfg) -> None:
    """Create figures and LaTeX tables from saved analysis output."""
    print(f"\n========== Figures: {cfg.model_name} ==========")

    economic_model, _ = solve_egm_model(cfg)
    egm_policy = EGMPolicy(economic_model)

    axes_egm, V_egm = load_value_grid(egm_analysis_dir(cfg) / "value_grid.npz")
    deterministic = is_deterministic_one_state(economic_model)
    active_test_seeds = effective_test_seeds(cfg, economic_model)

    rl_policies = {algo: [] for algo in cfg.algorithms}
    rl_value_grids = {algo: [] for algo in cfg.algorithms}
    histories = {algo: [] for algo in cfg.algorithms}
    history_seeds = {algo: [] for algo in cfg.algorithms}
    euler_pools = {algo: [] for algo in cfg.algorithms}
    euler_seedwise = {algo: [] for algo in cfg.algorithms}
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
                    euler_pools[algo].append(eulers[key])

                    if deterministic and not got_deterministic:
                        euler_seedwise[algo].append(eulers[key])
                        got_deterministic = True

                if not deterministic and key in visited:
                    visited_pools[algo].append(visited[key])

    train_egm_returns = {
        int(k): float(v)
        for k, v in read_json(egm_analysis_dir(cfg) / "train_egm_returns.json", {}).items()
    }

    egm_returns_by_algo = {
        algo: [train_egm_returns.get(seed, np.nan) for seed in history_seeds[algo]]
        for algo in cfg.algorithms
    }

    pooled_visited = {
        algo: np.concatenate([np.asarray(x).ravel() for x in arrays])
        for algo, arrays in visited_pools.items()
        if arrays
    }

    out = figure_dir(cfg)

    plot_policy_distribution(
        economic_model,
        egm_policy,
        rl_policies,
        out / "policy.png",
        algorithms=cfg.algorithms,
    )

    plot_cev_distribution(
        economic_model,
        axes_egm,
        V_egm,
        rl_value_grids,
        out / "cev.png",
        algorithms=cfg.algorithms,
        visited_m=pooled_visited,
    )

    plot_euler_error_distribution(
        economic_model,
        euler_pools,
        euler_seedwise,
        out / "euler_errors.png",
        algorithms=cfg.algorithms,
    )

    plot_training_convergence(
        histories,
        egm_returns_by_algo,
        cfg.max_wallclock_seconds,
        out / "convergence.png",
        algorithms=cfg.algorithms,
        grid_step=cfg.eval_interval_seconds,
    )

    summaries = {}
    for algo in cfg.algorithms:
        path = algorithm_summary_path(cfg, algo)
        if path.exists():
            summaries[algo] = read_json(path, {})

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

    print(f"[figures] saved -> {out}")


def run_analysis(cfg) -> None:
    """Run requested analysis stages."""
    if cfg.run_testing:
        run_testing(cfg)

    if cfg.make_plots_and_tables:
        run_plots_and_tables(cfg)