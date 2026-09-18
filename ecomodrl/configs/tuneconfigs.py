import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path

from ecomodrl.configs.runconfigs import (
    CassKoopmansRunConfig,
    BrockMirmanRunConfig,
    BufferStockRunConfig,
    PortfolioChoiceRunConfig
)
from ecomodrl.configs.variants import VARIANTS

# ----------------------------
# Tune configuration
# ----------------------------
# NOTE: set custom TUNERESULTS_PATH 
#       on your Windows system via setx TUNERESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export TUNERESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/tune_results"

@dataclass
class BaseTuneConfig:
    """
    Static tuning configuration for a sweep.
    Inherits from BaseRunConfig to allow shared parameters and parsing logic between tuning and final runs.
    """
    # Tuning
    out_dir: Path = Path(os.environ.get("TUNERESULTS_PATH", "./tune_results")) 
    total_cost: int = 60 * 60
    eta: int = 2
    rungs: int = 3
    min_fidelity_seconds: int = 30
    crossover_prob: float = 0.5
    mutation_factor: float = 0.5

    # Optional cap; by default tuning uses the CPUs available to this process.
    max_tune_workers: int | None = None
    
    # Seeding different for tuning!
    base_seed: int = 100
    n_seeds: int = 3
    @property
    def seeds(self) -> list[int]:
        """Deterministic seed set (for common random numbers across trials)."""
        return list(range(self.base_seed, self.base_seed + self.n_seeds))
        
        
@dataclass
class CassKoopmansTuneConfig(BaseTuneConfig, CassKoopmansRunConfig):
    pass

@dataclass
class BrockMirmanTuneConfig(BaseTuneConfig, BrockMirmanRunConfig):
    pass

@dataclass
class BufferStockTuneConfig(BaseTuneConfig, BufferStockRunConfig):
    pass

@dataclass
class PortfolioChoiceTuneConfig(BaseTuneConfig, PortfolioChoiceRunConfig):
    pass

def parse_tune_args(TuneConfig: BaseTuneConfig) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo_name", type=str, required=True,
                   choices=["ppo", "ddpg", "td3", "sac"],
                   help="RL algorithm to tune.")
    p.add_argument("--variant", type=str, default=argparse.SUPPRESS,
                   choices=list(VARIANTS),
                   help="Algorithm variant: 'mf' (model-free, default) or 'pd' (post-decision).")
    p.add_argument("--total_cost", type=int, default=argparse.SUPPRESS,
                   help="DEHB total cost budget (seconds).")
    p.add_argument("--hidden_nodes", type=int, default=argparse.SUPPRESS, 
                   help="Number of hidden nodes per layer in the policy/value networks.")
    p.add_argument("--eta", type=int, default=argparse.SUPPRESS, 
                   help="DEHB eta (downsampling rate).")
    p.add_argument("--rungs", type=int, default=argparse.SUPPRESS, 
                   help="Number of rungs in the DEHB hierarchy.")
    p.add_argument("--min_fidelity_seconds", type=int, default=argparse.SUPPRESS, 
                   help="Minimum fidelity in seconds.")
    p.add_argument("--crossover_prob", type=float, default=argparse.SUPPRESS, 
                   help="Probability of crossover in DEHB.")
    p.add_argument("--mutation_factor", type=float, default=argparse.SUPPRESS, 
                   help="Mutation factor in DEHB.")
    p.add_argument("--max_tune_workers", type=int, default=argparse.SUPPRESS,
        help="Optional hard cap on concurrent DEHB worker processes.")
    p.add_argument("--out_dir", type=Path, default=argparse.SUPPRESS)
    p.add_argument("--n_envs", type=int, default=argparse.SUPPRESS)
    p.add_argument("--max_wallclock_seconds", type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval_interval_seconds", type=int, default=argparse.SUPPRESS)
    p.add_argument(
        "--normalize_obs",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Whether to use VecNormalize observation normalization.",
    )
    p.add_argument("--n_seeds", type=int, default=argparse.SUPPRESS)
    
    args = p.parse_args()
    
    return replace(TuneConfig, **vars(args))
