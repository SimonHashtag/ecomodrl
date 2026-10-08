# -----------------------------------------------------------------------------
# Configuration objects for post-training analysis.
#
# The analysis config deliberately extends the corresponding run config with:
#   - test seeds;
#   - analysis sample sizes;
#   - algorithm selection;
#   - paths to trained checkpoints and figures;
#   - factories for model, solver, environment, and parameters.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ecomodrl.envs.setup_envs import make_env
from ecomodrl.configs.variants import ANALYSIS_VARIANT_CHOICES
from ecomodrl.configs.runconfigs import (
    CassKoopmansRunConfig,
    BrockMirmanRunConfig,
    BufferStockRunConfig,
    PortfolioChoiceRunConfig,
)
    
from ecomodrl.models import (
    CassKoopmans, 
    BrockMirman, 
    BufferStock,
    PortfolioChoice,
    BaseEGM,
    PortfolioChoiceSolver,
)

ALL_ALGORITHMS = ("ddpg", "td3", "sac", "ppo")

@dataclass
class BaseAnalysisConfig:
    """Base configuration for post-training analysis."""

    # Paths
    out_dir: Path = Path(os.environ.get("RUNRESULTS_PATH", "./runs"))

    # Algorithm selection
    algorithms: tuple[str, ...] = ALL_ALGORITHMS

    # Test seeds
    test_base_seed: int = 300
    n_test_seeds: int = 10
    
    # Evaluation
    euler_n_states: int = 100

    # Parallelism
    max_workers: int | None = None

    # Whether to recompute if files already exist.
    overwrite: bool = False
    
    # Analysis stages
    run_testing: bool = True
    make_plots_and_tables: bool = True
    
    # Metrics
    conv_tol: float = 1e-4      # Convergence tolerance used for time-to-converged metric
    flops_forward_steps: int | None = None    # Env_steps used to determine FLOPs count; automatically determined from hyperparameters if None

    @property
    def test_seeds(self) -> list[int]:
        return list(range(self.test_base_seed, self.test_base_seed + self.n_test_seeds))

    @property
    def model_name(self) -> str:
        """Directory name used for this economic model."""
        return self.env_name

    def make_economic_model(self):
        raise NotImplementedError

    def make_egm_solver(self, model):
        raise NotImplementedError

    def make_unnormalized_vecenv(self, seed: int):
        """Create a dummy VecEnv for loading VecNormalize statistics.

        VecNormalize.load(path, venv) requires an underlying VecEnv. The
        returned VecEnv must match the training observation/action spaces, but
        it should not itself already be wrapped in VecNormalize.
        """
        vec_cls = DummyVecEnv

        return make_vec_env(
            make_env(
                self.env_class(),
                self.make_params(),
                max_episode_steps=self.max_episode_steps,
                reward_floor=100.0,
            ),
            n_envs=1,
            seed=int(seed),
            vec_env_cls=vec_cls,
        )

    def load_vecnormalize(self, path: Path, seed: int):
        """Load saved VecNormalize statistics for offline policy evaluation."""
        dummy_vecenv = self.make_unnormalized_vecenv(seed=seed)
        vecnorm = VecNormalize.load(str(path), dummy_vecenv)
        vecnorm.training = False
        vecnorm.norm_reward = False
        return vecnorm

@dataclass
class CassKoopmansAnalysisConfig(CassKoopmansRunConfig, BaseAnalysisConfig):
    """Post-training analysis configuration for Cass-Koopmans."""

    def make_economic_model(self):
        return CassKoopmans(self.make_params())

    def make_egm_solver(self, model):
        return BaseEGM(model)


@dataclass
class BrockMirmanAnalysisConfig(BrockMirmanRunConfig, BaseAnalysisConfig):
    """Post-training analysis configuration for Brock-Mirman."""

    def make_economic_model(self):
        return BrockMirman(self.make_params())

    def make_egm_solver(self, model):
        return BaseEGM(model)


@dataclass
class BufferStockAnalysisConfig(BufferStockRunConfig, BaseAnalysisConfig):
    """Post-training analysis configuration for Buffer Stock."""

    def make_economic_model(self):
        return BufferStock(self.make_params())

    def make_egm_solver(self, model):
        return BaseEGM(model)
    
@dataclass
class PortfolioChoiceAnalysisConfig(PortfolioChoiceRunConfig, BaseAnalysisConfig):
    """Post-training analysis configuration for Portfolio Choice."""
    
    #max_workers: int | None = 1  # Limit parallelism for Portfolio Choice due to numba

    def make_economic_model(self):
        return PortfolioChoice(self.make_params())

    def make_egm_solver(self, model):
        return PortfolioChoiceSolver(model)


def _parse_algorithm_selection(algo_names: list[str]) -> tuple[str, ...]:
    """Convert CLI algorithm selection to a tuple of algorithm names."""
    if "all" in algo_names:
        if len(algo_names) > 1:
            raise argparse.ArgumentTypeError(
                "Use either '--algo_name all' or a list of algorithms, not both."
            )
        return ALL_ALGORITHMS

    return tuple(algo_names)


def parse_analysis_args(AnalysisConfig: BaseAnalysisConfig, argv=None) -> argparse.Namespace:
    """Parse post-training analysis arguments.

    Examples
    --------
    Run all algorithms:
        python analyze_brockmirman.py

    Run one algorithm:
        python analyze_brockmirman.py --algo_name ppo

    Run two algorithms:
        python analyze_brockmirman.py --algo_name ppo sac

    Only make plots from existing results:
        python analyze_brockmirman.py --skip_testing

    Only run numerical analysis without plotting:
        python analyze_brockmirman.py --skip_plots

    Overlay both variants (from existing per-variant output):
        python analyze_brockmirman.py --variant both
    """
    p = argparse.ArgumentParser()

    p.add_argument(
        "--algo_name",
        "--algo",
        nargs="+",
        choices=["all", *ALL_ALGORITHMS],
        default=argparse.SUPPRESS,
        help="Algorithms to analyze. Default: config value, usually all.",
    )
    p.add_argument(
        "--variant",
        type=str,
        choices=list(ANALYSIS_VARIANT_CHOICES),
        default=argparse.SUPPRESS,
        help=(
            "Algorithm variant to analyze: 'mf' (model-free, default), 'pd' (post-decision), "
            "or 'both' (only writes the combined mf-vs-pd figures from existing per-variant "
            "analysis output; no testing, --skip_testing/--skip_plots are ignored)."
        ),
    )

    p.add_argument("--out_dir", type=Path, default=argparse.SUPPRESS)
    p.add_argument("--max_wallclock_seconds", type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval_interval_seconds", type=int, default=argparse.SUPPRESS)
    p.add_argument(
        "--normalize_obs",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Whether to use VecNormalize observation normalization.",
    )
    p.add_argument("--n_seeds", type=int, default=argparse.SUPPRESS)

    p.add_argument("--n_test_seeds", type=int, default=argparse.SUPPRESS)
    p.add_argument("--test_base_seed", type=int, default=argparse.SUPPRESS)

    p.add_argument("--euler_n_states", type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval_m_grid_size", type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval_horizon", type=int, default=argparse.SUPPRESS)

    p.add_argument("--max_workers", type=int, default=argparse.SUPPRESS)

    p.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Recompute and overwrite existing analysis outputs.",
    )

    p.add_argument(
        "--skip_testing",
        dest="run_testing",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Skip numerical analysis and only make plots from existing results.",
    )

    p.add_argument(
        "--skip_plots",
        dest="make_plots_and_tables",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Run numerical analysis but skip figures and tables.",
    )

    updates = vars(p.parse_args(argv))

    if "algo_name" in updates:
        updates["algorithms"] = _parse_algorithm_selection(
            updates.pop("algo_name")
        )

    return replace(AnalysisConfig, **updates)
