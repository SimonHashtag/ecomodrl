# -----------------------------------------------------------------------------
# Post-training analysis for one economic model.
#
# Responsibilities:
#   1. Solve the EGM benchmark.
#   2. Build fixed evaluation samples for common-random-number comparisons.
#   3. Evaluate trained RL checkpoints by algorithm, training seed, and test seed.
#   4. Compute Euler-error diagnostics.
#   5. Save summaries for plotting and tables.
#
# This file is model-agnostic. A model-specific entry file supplies an
# AnalysisConfig with factories for model, solver, env, and parameters.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from ecomodrl.analysis.latex_tables import make_latex_table, make_combined_latex_table

# =============================================================================
# Paths
# =============================================================================

def model_results_dir(cfg) -> Path:
    return Path(cfg.out_dir) / cfg.model_name


def figure_dir(cfg) -> Path:
    return model_results_dir(cfg) / "figures"


# EGM
def egm_analysis_dir(cfg) -> Path:
    return model_results_dir(cfg) / "egm_analysis"


def evaluation_sample_path(cfg, test_seed: int) -> Path:
    return egm_analysis_dir(cfg) / f"eval_sample_analysis_seed_{int(test_seed)}.pkl"


# RL
def checkpoint_dir(cfg, algo: str, train_seed: int) -> Path:
    return model_results_dir(cfg) / algo / f"seed_{int(train_seed)}"


def analysis_result_dir(cfg, algo: str, train_seed: int) -> Path:
    return checkpoint_dir(cfg, algo, train_seed) / "analysis"


def value_grid_path(cfg, algo: str, train_seed: int) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "value_grid.npz"


def test_values_path(cfg, algo: str, train_seed: int) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "test_values.json"


def euler_errors_path(cfg, algo: str, train_seed: int) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "euler_errors.npz"


def visited_states_path(cfg, algo: str, train_seed: int) -> Path:
    return analysis_result_dir(cfg, algo, train_seed) / "visited_m.npz"


# =============================================================================
# Small I/O helpers
# =============================================================================

def read_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_npz_dict(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as d:
        return {k: d[k].copy() for k in d.files}


def write_npz_dict(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def save_value_grid(path: Path, grid, V: np.ndarray) -> None:
    """Save value grid with generic axis names and convenient aliases.

    Grid convention:
        axes = (m_grid, x1_grid, ..., xd_grid)

    Saved keys:
        V
        axis_0, axis_1, ...
        m_grid
        z_grid, if exactly one exogenous axis exists
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
    with np.load(path) as d:
        V = d["V"].copy()

        axes = []
        i = 0
        while f"axis_{i}" in d.files:
            axes.append(d[f"axis_{i}"].copy())
            i += 1

        if not axes:
            # Backward-compatible fallback.
            axes = [d["m_grid"].copy()]
            if "z_grid" in d.files:
                axes.append(d["z_grid"].copy())

    return tuple(axes), V


def check_required_checkpoints(cfg):
    missing = []

    for algo in cfg.algorithms:
        for seed in cfg.seeds:
            if not (checkpoint_dir(cfg, algo, seed) / "model.zip").exists():
                missing.append(str(checkpoint_dir(cfg, algo, seed) / "model.zip"))

    if missing:
        raise RuntimeError(
            f"Analysis aborted: {len(missing)} trained checkpoints are missing. "
            "First few missing files:\n  "
            + "\n  ".join(missing[:10])
        )


# =============================================================================
# Model and policies
# =============================================================================

def solve_egm_model(cfg):
    """Construct and solve the economic model with EGM."""
    model = cfg.make_economic_model()
    solver = cfg.make_egm_solver(model)
    solver.solve(verbose=False)
    model.policy = solver.policy
    return model, solver


def is_deterministic_one_state(model) -> bool:
    return model.weights is None and model.exogenous_dim == 0


def load_rl_policy(cfg, algo: str, train_seed: int, economic_model):
    """Load one trained RL checkpoint as an RLPolicy."""
    seed_dir = checkpoint_dir(cfg, algo, train_seed)

    model_path = seed_dir / "model.zip"
    vecnorm_path = seed_dir / "vecnormalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not vecnorm_path.exists():
        raise FileNotFoundError(
            f"Missing VecNormalize statistics: {vecnorm_path}. "
            "Policies trained with VecNormalize must be evaluated with the "
            "same normalization statistics."
        )

    vecnormalize = cfg.load_vecnormalize(vecnorm_path, seed=train_seed)
    sb3_model = ALGO_REGISTRY[algo].load(str(model_path), env=vecnormalize)

    return RLPolicy(
        sb3_model=sb3_model,
        model=economic_model,
        vecnormalize=vecnormalize,
    )


# =============================================================================
# Evaluation samples and EGM references
# =============================================================================

# TODO: For is_deterministic_one_state models, we could just use the single state 
# as the evaluation sample and skip the random sampling. 
# This would make the evaluation deterministic and faster.
def prepare_evaluation_samples(cfg, economic_model, egm_policy):
    """Create fixed evaluation samples and EGM reference returns."""
    out_dir = egm_analysis_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    egm_returns_path = out_dir / "analysis_egm_returns.json"
    egm_returns = {
        int(k): float(v)
        for k, v in read_json(egm_returns_path, {}).items()
    }

    for test_seed in cfg.test_seeds:
        path = evaluation_sample_path(cfg, test_seed)

        if cfg.overwrite or not path.exists():
            sample = metrics.make_evaluation_sample(
                economic_model,
                seed=test_seed,
                n_states=cfg.eval_n_states,
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


def cache_egm_value_grid(cfg, economic_model, egm_policy):
    path = egm_analysis_dir(cfg) / "value_grid.npz"

    if path.exists() and not cfg.overwrite:
        return

    grid, V = metrics.policy_value_on_grid(economic_model, egm_policy)
    save_value_grid(path, grid, V)


def cache_egm_euler_errors(cfg, economic_model, egm_policy):
    """Cache EGM Euler-error reference distributions."""
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

    for test_seed in cfg.test_seeds:
        path = egm_analysis_dir(cfg) / f"egm_euler_errors_{test_seed}.npy"

        if path.exists() and not cfg.overwrite:
            continue

        ee = metrics.euler_errors(
            economic_model,
            egm_policy,
            n_sim=cfg.euler_n_states,
            seed=test_seed,
        )
        np.save(path, np.asarray(ee, dtype=np.float64))


def cache_train_egm_returns(cfg, economic_model, egm_policy):
    """Cache EGM returns on each training-run evaluation sample.

    These are used for convergence plots that show RL regret relative to EGM
    on the same common-random-number sample used during training evaluation.
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


# =============================================================================
# Per-seed analysis
# =============================================================================

def analyze_train_seed(cfg, train_seed: int):
    """Analyze all requested algorithms for one training seed."""
    economic_model, _ = solve_egm_model(cfg)

    deterministic = is_deterministic_one_state(economic_model)

    # Load evaluation samples and EGM returns.
    samples = {}
    for test_seed in cfg.test_seeds:
        with open(evaluation_sample_path(cfg, test_seed), "rb") as f:
            samples[test_seed] = pickle.load(f)

    egm_returns = {
        int(k): float(v)
        for k, v in read_json(egm_analysis_dir(cfg) / "analysis_egm_returns.json", {}).items()
    }

    statuses = []

    for algo in cfg.algorithms:
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
            statuses.append((algo, "ok"))
        except FileNotFoundError:
            statuses.append((algo, "missing"))
        except Exception as exc:
            statuses.append((algo, repr(exc)))

    return int(train_seed), statuses


def analyze_policy_checkpoint(
    *,
    cfg,
    economic_model,
    algo: str,
    train_seed: int,
    test_samples: dict[int, object],
    egm_returns: dict[int, float],
    deterministic: bool,
):
    """Analyze one RL checkpoint."""
    policy = load_rl_policy(cfg, algo, train_seed, economic_model)

    out_dir = analysis_result_dir(cfg, algo, train_seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bellman value grid for CEV maps.
    vg_path = value_grid_path(cfg, algo, train_seed)
    if cfg.overwrite or not vg_path.exists():
        grid, V = metrics.policy_value_on_grid(economic_model, policy)
        save_value_grid(vg_path, grid, V)

    # 2. Test returns and CEV regret on common evaluation samples.
    values_path = test_values_path(cfg, algo, train_seed)
    values = read_json(values_path, {})

    eulers_path = euler_errors_path(cfg, algo, train_seed)
    eulers = read_npz_dict(eulers_path)

    visited_path = visited_states_path(cfg, algo, train_seed)
    visited = read_npz_dict(visited_path) if not deterministic else {}

    changed_values = False
    changed_eulers = False
    changed_visited = False

    for test_seed in cfg.test_seeds:
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

            if np.isfinite(u_rl):
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

def load_egm_euler_reference(cfg, economic_model, test_seed: int | None):
    if is_deterministic_one_state(economic_model):
        path = egm_analysis_dir(cfg) / "euler_errors.npy"
        return np.load(path) if path.exists() else None

    path = egm_analysis_dir(cfg) / f"egm_euler_errors_{int(test_seed)}.npy"
    return np.load(path) if path.exists() else None


def summarize_algorithm(cfg, economic_model, algo: str):
    """Aggregate train-seed x test-seed results for one algorithm."""
    deterministic = is_deterministic_one_state(economic_model)
    active_test_seeds = cfg.test_seeds[:1] if deterministic else cfg.test_seeds

    shape = (len(cfg.seeds), len(active_test_seeds))

    matrices = {
        "u_rl": np.full(shape, np.nan),
        "cev_regret": np.full(shape, np.nan),
        "ee_mean": np.full(shape, np.nan),
        "ee_p95": np.full(shape, np.nan),
        "wasserstein": np.full(shape, np.nan),
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

            ref = load_egm_euler_reference(
                cfg,
                economic_model,
                None if deterministic else test_seed,
            )
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

    path = model_results_dir(cfg) / algo / "analysis_summary.json"
    write_json(path, summary)


def summarize_matrix(matrix: np.ndarray) -> dict:
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
# Main analysis routines
# =============================================================================

def run_testing(cfg):
    print(f"\n========== Analysis: {cfg.model_name} ==========")
    print(f"Algorithms: {', '.join(cfg.algorithms)}")

    check_required_checkpoints(cfg)

    economic_model, _ = solve_egm_model(cfg)
    egm_policy = EGMPolicy(economic_model)

    cache_egm_value_grid(cfg, economic_model, egm_policy)
    prepare_evaluation_samples(cfg, economic_model, egm_policy)
    cache_egm_euler_errors(cfg, economic_model, egm_policy)
    cache_train_egm_returns(cfg, economic_model, egm_policy)

    worker_count = cfg.max_workers
    if worker_count is None:
        worker_count = max(1, (os.cpu_count() or 1) - 1)

    units = list(cfg.seeds)

    print(f"[analysis] {len(units)} training seeds; workers={worker_count}")

    if worker_count == 1:
        for seed in units:
            train_seed, statuses = analyze_train_seed(cfg, seed)
            print_status(train_seed, statuses)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(analyze_train_seed, cfg, seed): seed for seed in units}

            done = 0
            for fut in as_completed(futures):
                seed = futures[fut]
                done += 1

                try:
                    train_seed, statuses = fut.result()
                    print_status(train_seed, statuses, done, len(units))
                except Exception as exc:
                    print(
                        f"[analysis] FAILED seed={seed}: {exc!r} "
                        f"({done}/{len(units)})",
                        flush=True,
                    )

    economic_model, _ = solve_egm_model(cfg)

    for algo in cfg.algorithms:
        summarize_algorithm(cfg, economic_model, algo)
        print(f"[analysis] wrote summary for {cfg.model_name}/{algo}")


def print_status(train_seed, statuses, done=None, total=None):
    tags = " ".join(f"{algo}:{status}" for algo, status in statuses)

    suffix = ""
    if done is not None and total is not None:
        suffix = f" ({done}/{total})"

    print(f"[analysis] seed={train_seed} [{tags}]{suffix}", flush=True)


# =============================================================================
# Plotting
# =============================================================================

def load_training_history(seed_dir: Path):
    """Load callback evaluation history.

    New callback files:
        eval_history.npz with history_t/history_U

    Old fallback:
        history.npz with t/V_grid or history_t/history_V
    """
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


def run_plots_and_tables(cfg):
    print(f"\n========== Figures: {cfg.model_name} ==========")

    economic_model, _ = solve_egm_model(cfg)
    egm_policy = EGMPolicy(economic_model)

    axes_egm, V_egm = load_value_grid(egm_analysis_dir(cfg) / "value_grid.npz")

    deterministic = is_deterministic_one_state(economic_model)

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

            for test_seed in cfg.test_seeds:
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

    pooled_visited = {}
    for algo, arrays in visited_pools.items():
        if arrays:
            pooled_visited[algo] = np.concatenate(
                [np.asarray(x).ravel() for x in arrays]
            )

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
        path = model_results_dir(cfg) / algo / "analysis_summary.json"
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
        Path(cfg.figures_dir) / f"{cfg.model_name}_comparison_table.tex",
        algorithms=cfg.algorithms,
    )

    print(f"[figures] saved -> {out}")


def run_analysis(cfg):
    if cfg.run_testing:
        run_testing(cfg)

    if cfg.make_plots_and_tables:
        run_plots_and_tables(cfg)