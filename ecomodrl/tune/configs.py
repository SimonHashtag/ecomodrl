import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path

from ecomodrl.experiments.configs import BaseRunConfig
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem as CKPP
from ecomodrl.benchmarks.brockmirman.problemclass import PlanningProblem as BMPP
from ecomodrl.benchmarks.bufferstock.problemclass import PlanningProblem as BSPP
from ecomodrl.benchmarks.portfoliochoice.problemclass import PlanningProblem as PCPP

# ----------------------------
# Tune configuration
# ----------------------------
# NOTE: set custom TUNERESULTS_PATH 
#       on your Windows system via setx TUNERESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export TUNERESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/tune_results"
#
# Run file with e.g.: python ecomodrl/tune/casskoopmans.py --algo_name="ddpg" --total_cost=3600 --hidden_nodes=64

@dataclass
class BaseTuneConfig(BaseRunConfig):
    """
    Static tuning configuration for a sweep.
    Inherits from BaseRunConfig to allow shared parameters and parsing logic between tuning and final runs.
    """
    # Tuning
    out_dir: Path = Path(os.environ.get("TUNERESULTS_PATH", "./tune_results")) 
    total_cost: int = 60 * 60
    eta: int = 2
    rungs: int = 3
    min_fidelity_seconds: int = 60
    crossover_prob: float = 0.5
    mutation_factor: float = 0.5
    
    # Seeding different for tuning!
    base_seed: int = 100
    n_seeds: int = 3
    @property
    def seeds(self) -> list[int]:
        """Deterministic seed set (for common random numbers across trials)."""
        return list(range(self.base_seed, self.base_seed + self.n_seeds))
        
        
@dataclass
class CassKoopmansTuneConfig(BaseTuneConfig):
    """Static CassKoopmans tuning configuration for a sweep."""
    # Environment
    env_name: str = "CassKoopmans"
    max_wallclock_seconds: int = 600
    
    def __post_init__(self):
        """ 
        Use steady-state aware starting boundaries for RL algorithm
        Intention: don't explore outside of the economically relevant state space
        """
        mod = CKPP()
        grid = mod.make_grid()
        self.state_min = grid[0]
        self.state_max = grid[-1]

 
@dataclass
class BrockMirmanTuneConfig(BaseTuneConfig):
    """Static BrockMirman tuning configuration for a sweep."""
    # Environment
    env_name: str = "BrockMirman"
    max_wallclock_seconds: int = 1200
    n_eval_episodes: int = 5  # average over multiple shock sequences per eval (stochastic env)
    
    def __post_init__(self):
        """ 
        Use steady-state aware starting boundaries for RL algorithm
        Intention: don't explore outside of the economically relevant state space
        """
        mod = BMPP()
        grid = mod.make_k_grid()
        self.state_min = grid[0]
        self.state_max = grid[-1]


@dataclass
class BufferStockTuneConfig(BaseTuneConfig):
    """Static BufferStock tuning configuration for a sweep."""
    # Environment
    env_name: str = "BufferStock"
    max_wallclock_seconds: int = 1200
    n_eval_episodes: int = 5  # average over multiple shock sequences per eval (stochastic env)

    def __post_init__(self):
        """ 
        Use steady-state aware starting boundaries for RL algorithm
        Intention: don't explore outside of the economically relevant state space
        """
        mod = BSPP()
        grid = mod.make_grid(100)
        self.state_min = 1.0
        self.state_max = grid[-1]


@dataclass
class PortfolioChoiceTuneConfig(BaseTuneConfig):
    """Static PortfolioChoice tuning configuration for a sweep."""
    # Environment
    env_name: str = "PortfolioChoice"
    max_wallclock_seconds: int = 1200
    n_eval_episodes: int = 5  # average over multiple shock sequences per eval (stochastic env)
    
    def __post_init__(self):
        """ 
        Use steady-state aware starting boundaries for RL algorithm
        Intention: don't explore outside of the economically relevant state space
        """
        mod = PCPP()
        grid = mod.make_m_grid()
        self.state_min = grid[0]
        self.state_max = grid[-1]


def parse_tune_args(TuneConfig: BaseTuneConfig) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo_name", type=str, required=True,
                   choices=["ppo", "ddpg", "td3", "sac"],
                   help="RL algorithm to tune.")
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
    p.add_argument("--out_dir", type=Path, default=argparse.SUPPRESS)
    p.add_argument("--n_envs", type=int, default=argparse.SUPPRESS)
    p.add_argument("--max_wallclock_seconds", type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval_every", type=int, default=argparse.SUPPRESS)
    p.add_argument("--n_seeds", type=int, default=argparse.SUPPRESS)
    
    args = p.parse_args()
    
    return replace(TuneConfig, **vars(args))