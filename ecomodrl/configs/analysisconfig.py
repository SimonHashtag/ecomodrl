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
from ecomodrl.configs.runconfigs import BrockMirmanRunConfig
from ecomodrl.configs.parameters import BrockMirmanParams
from ecomodrl.models.base_model import BrockMirman, BaseEGM
from ecomodrl.envs.envs import BrockMirmanEnv

ALL_ALGORITHMS: tuple[str, ...] = ("ppo", "ddpg", "td3", "sac")

@dataclass
class BaseAnalysisConfig:
    """Base configuration for post-training analysis."""

    # Paths
    out_dir: Path = Path(os.environ.get("RUNRESULTS_PATH", "./runs"))
    figures_dir: Path = Path("./figures")

    # Algorithm selection
    algorithms: tuple[str, ...] = ALL_ALGORITHMS

    # Test seeds
    test_base_seed: int = 300
    n_test_seeds: int = 10
    
    # Evaluation
    euler_n_states: int = 10_000

    # Parallelism
    max_workers: int | None = None

    # Whether to recompute if files already exist.
    overwrite: bool = False

    @property
    def test_seeds(self) -> list[int]:
        return list(range(self.test_base_seed, self.test_base_seed + self.n_test_seeds))

    @property
    def model_name(self) -> str:
        """Directory name used for this economic model."""
        return self.env_name

    def make_params(self):
        raise NotImplementedError

    def make_economic_model(self):
        raise NotImplementedError

    def make_egm_solver(self, model):
        raise NotImplementedError

    def env_class(self):
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
class BrockMirmanAnalysisConfig(BrockMirmanRunConfig, BaseAnalysisConfig):
    """Post-training analysis configuration for Brock-Mirman."""

    def make_params(self):
        return BrockMirmanParams()

    def make_economic_model(self):
        return BrockMirman(self.make_params())

    def make_egm_solver(self, model):
        return BaseEGM(model)

    def env_class(self):
        return BrockMirmanEnv


def parse_analysis_args(default_config: BaseAnalysisConfig) -> BaseAnalysisConfig:
    """Parse common analysis command-line arguments.

    Examples
    --------
    Run all algorithms:
        python analyze_brockmirman.py

    Run one algorithm:
        python analyze_brockmirman.py --algo ppo

    Run two algorithms:
        python analyze_brockmirman.py --algo ppo sac

    Only make plots from existing results:
        python analyze_brockmirman.py --skip_testing
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo_name",
        nargs="+",
        default=["all"],
        choices=["all", *ALL_ALGORITHMS],
        help="Algorithms to analyze. Default: all.",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--figures_dir", type=Path, default=None)
    parser.add_argument("--n_test_seeds", type=int, default=None)
    parser.add_argument("--test_base_seed", type=int, default=None)
    parser.add_argument("--euler_n_states", type=int, default=None)
    parser.add_argument("--eval_n_states", type=int, default=None)
    parser.add_argument("--eval_horizon", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--skip_testing", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")

    args = parser.parse_args()

    updates = {}

    if args.algo_name == ["all"]:
        updates["algorithms"] = ALL_ALGORITHMS
    else:
        updates["algorithms"] = tuple(args.algo_name)

    for name in (
        "out_dir",
        "figures_dir",
        "n_test_seeds",
        "test_base_seed",
        "euler_n_states",
        "eval_n_states",
        "eval_horizon",
        "max_workers",
    ):
        value = getattr(args, name)
        if value is not None:
            updates[name] = value

    if args.overwrite:
        updates["overwrite"] = True

    cfg = replace(default_config, **updates)
    cfg.skip_testing = bool(args.skip_testing)
    cfg.skip_plots = bool(args.skip_plots)

    return cfg