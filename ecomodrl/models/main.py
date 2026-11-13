"""Appendix diagnostics for the benchmark policies used by analysis.py.

Run with python -m ecomodrl.models.main. Solve time is a fresh policy solve,
including first-use Numba compilation, but excluding model construction and
Euler evaluation.
"""

import numpy as np

from ecomodrl.analysis import accuracy_metrics
from ecomodrl.analysis.analysis import effective_test_seeds, make_benchmark_model
from ecomodrl.analysis.policy_interface import EGMPolicy
from ecomodrl.configs.analysisconfigs import (
    BrockMirmanAnalysisConfig,
    BufferStockAnalysisConfig,
    CassKoopmansAnalysisConfig,
    PortfolioChoiceAnalysisConfig,
)


def describe_axes(name, axes):
    for label, axis in zip(("m", "z1", "z2"), axes):
        print(f"  {name} {label}: {len(axis)} points, [{axis[0]:.6g}, {axis[-1]:.6g}]")


if __name__ == "__main__":
    for config_type in (
        CassKoopmansAnalysisConfig,
        BrockMirmanAnalysisConfig,
        BufferStockAnalysisConfig,
        PortfolioChoiceAnalysisConfig,
    ):
        cfg = config_type()
        model = cfg.make_economic_model()
        solve_model = make_benchmark_model(cfg, model)
        solver = cfg.make_egm_solver(solve_model)
        portfolio = accuracy_metrics._is_portfolio_choice(model)

        print(f"\n{cfg.model_name} ({type(solver).__name__})", flush=True)
        describe_axes("policy", solve_model.policy_axes)
        asset_label = "rs" if portfolio else "a"
        print(f"  asset {asset_label}: {len(solve_model.a_grid)} points, [{solve_model.a_grid[0]:.6g}, {solve_model.a_grid[-1]:.6g}]")
        if portfolio:
            diagnostic_nodes = len(accuracy_metrics._portfolio_shock_indices(model))
            print(f"  integration: scrambled Sobol with antithetics; "
                  f"{solve_model._M_rqmc} solve nodes, {diagnostic_nodes} diagnostic prefix nodes; "
                  "employment mixture integrated analytically")
        elif model.weights is None:
            print("  integration: deterministic (one future state)")
        else:
            print(f"  integration: Gauss-Hermite; {len(model.weights)} weighted shock nodes")

        print("  method: " + ("nested EGM and simplex KKT root solves; no short sales"
                               if portfolio else "consumption EGM fixed point")
              + "; linear policy interpolation")
        solver.solve(verbose=True)
        policy = EGMPolicy(solve_model)
        seeds = effective_test_seeds(cfg, model)
        errors = np.concatenate([
            accuracy_metrics.euler_errors(
                model, policy, n_sim=cfg.euler_n_states,
                burn_in=cfg.burn_in, seed=seed,
            )
            for seed in seeds
        ])
        finite = errors[np.isfinite(errors)]
        mean = float(np.mean(finite)) if finite.size else float("nan")
        maximum = float(np.max(finite)) if finite.size else float("nan")
        metric = "joint Euler/KKT" if portfolio else "Euler"
        if portfolio:
            print(f"  solve time: {solver.solve_time:.2f} s; {solver.iterations} iterations; "
                  f"outer change tolerance {solver.conv:g}; "
                  f"grid KKT tolerance {max(solver.conv, solver.portfolio_tol):g} "
                  f"(achieved {solver.max_portfolio_residual:.2e}); "
                  f"local root KKT tolerance {solver.portfolio_tol:g}")
        else:
            print(f"  solve time: {solver.solve_time:.2f} s; tolerance {solver.tol:g}")
        print(f"  {metric} log10 absolute error: mean {mean:.4f}, max {maximum:.4f} "
              f"{len(seeds)} seeds, {cfg.euler_n_states} states/seed, "
              f"burn-in {cfg.burn_in})", flush=True)
