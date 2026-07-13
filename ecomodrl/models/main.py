import numpy as np

from ecomodrl.configs.parameters import (
    BrockMirmanParams, 
    CassKoopmansParams,
    BufferStockParams,
    PortfolioChoiceParams,
)
from ecomodrl.analysis import accuracy_metrics
from ecomodrl.analysis.policy_interface import EGMPolicy
from bufferstock import BufferStock
from portfoliochoice import PortfolioChoice
from brockmirman import BrockMirman
from casskoopmans import CassKoopmans
from solvers import BaseEGM, PortfolioChoiceSolver

if __name__ == "__main__":

    specs = [
        ("Cass-Koopmans", CassKoopmans(CassKoopmansParams())),
        ("Brock-Mirman", BrockMirman(BrockMirmanParams())),
        ("Buffer stock", BufferStock(BufferStockParams())),
        ("PortfolioChoice", PortfolioChoice(PortfolioChoiceParams(), m_grid_size=200, rqmc_nodes=2**10, rqmc_seed=1996)),
    ]

    solved = []

    for label, model in specs:
        print(label)
        if label == "PortfolioChoice":
            solver = PortfolioChoiceSolver(model)
        else:
            solver = BaseEGM(model)
        solver.solve(verbose=True)
        egm_policy = EGMPolicy(model)
        print(f"Grid-reference state m_ref for {label}: {solver.model.m_ref:.2f}")
        solved.append((label, model, solver, egm_policy))

    header = f"{'Model':<35} {'Mean log10|Euler error|':>24} {'Mean Return':>24} {'Solve time (s)':>16}"
    print(header)
    print("=" * len(header))

    for label, model, solver, egm_policy in solved:
        ee = accuracy_metrics.euler_errors(model, egm_policy, n_sim=100)
        mean_err = np.mean(ee) if ee.size else np.nan
        sample = accuracy_metrics.make_evaluation_sample(model, seed=1, n_states=1000, horizon=100)
        mean_U = accuracy_metrics.eval_return_grid(model, egm_policy, sample)
        print(f"{label:<35} {mean_err:>24.4f} {mean_U:>24.4f} {solver.solve_time:>16.2f}")

    print("=" * len(header))