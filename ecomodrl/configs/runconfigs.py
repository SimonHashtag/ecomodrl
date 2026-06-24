import argparse
import os
from dataclasses import dataclass, field, replace
from pathlib import Path


from ecomodrl.models.casskoopmans.problemclass import PlanningProblem as CKPP
from ecomodrl.models.brockmirman.problemclass import PlanningProblem as BMPP
from ecomodrl.models.bufferstock.problemclass import PlanningProblem as BSPP
from ecomodrl.models.portfoliochoice.problemclass import PlanningProblem as PCPP

# ----------------------------
# Run configuration
# ----------------------------
# NOTE: set custom RUNRESULTS_PATH 
#       on your Windows system via setx RUNRESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export RUNRESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/runs/ENV"

@dataclass
class BaseRunConfig:
    """Static tuning configuration for a sweep."""
    # Location
    out_dir: Path = Path(os.environ.get("RUNRESULTS_PATH", "./runs"))
    
    # Algorithm
    algo_name: str = "ppo"
    hidden_nodes: int = 64
    total_timesteps: int = 1_000_000_000
    max_wallclock_seconds: int = 600
    base_seed: int = 200
    n_seeds: int = 10
    @property
    def seeds(self) -> list[int]:
        """Deterministic seed set (for common random numbers across trials)."""
        return list(range(self.base_seed, self.base_seed + self.n_seeds))

    # Environment
    n_envs: int = 10
    max_episode_steps: int = 30
    min_delta: float = 1e-3
    state_min: float = field(init=False)
    state_max: float = field(init=False)
    
    # Evaluation
    eval_interval_seconds: int = 5.0
    eval_m_grid_size: int = 1_000
    eval_time_period: int = 100
    
        
        
@dataclass
class CassKoopmansRunConfig(BaseRunConfig):
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
class BrockMirmanRunConfig(BaseRunConfig):
    """Static BrockMirman run configuration for a sweep."""
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
class BufferStockRunConfig(BaseRunConfig):
    """Static BufferStock run configuration for a sweep."""
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
class PortfolioChoiceRunConfig(BaseRunConfig):
    """Static PortfolioChoice run configuration for a sweep."""
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
        grid = mod.make_m_grid(span=(0.2, 30.0))
        self.state_min = grid[0]
        self.state_max = grid[-1]


def parse_run_args(RunConfig: BaseRunConfig) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo_name", type=str, required=True,
                   choices=["ppo", "ddpg", "td3", "sac"],
                   help="RL algorithm to tune.")
    p.add_argument("--hidden_nodes", type=int, default=argparse.SUPPRESS, 
                   help="Number of hidden nodes per layer in the policy/value networks.")
    p.add_argument("--out_dir", type=Path, default=argparse.SUPPRESS)
    p.add_argument("--n_envs", type=int, default=argparse.SUPPRESS)
    p.add_argument("--max_wallclock_seconds", type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval_every", type=int, default=argparse.SUPPRESS)
    p.add_argument("--n_seeds", type=int, default=argparse.SUPPRESS)
    
    args = p.parse_args()
    
    return replace(RunConfig, **vars(args))