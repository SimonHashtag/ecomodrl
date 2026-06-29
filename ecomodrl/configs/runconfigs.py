import argparse
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from ecomodrl.configs.parameters import (
    CassKoopmansParams,
    BrockMirmanParams,
    BufferStockParams,
    PortfolioChoiceParams,
)
from ecomodrl.envs.envs import (
    CassKoopmansEnv,
    BrockMirmanEnv,
    BufferStockEnv,
)

from ecomodrl.envs.portfoliochoice import PortfolioChoiceEnv

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
    eval_seed_offset: int = 1_000
    eval_m_grid_size: int = 1_000
    eval_horizon: int = 100
    
    def make_params(self):
        raise NotImplementedError

    def env_class(self):
        raise NotImplementedError
    
        
        
@dataclass
class CassKoopmansRunConfig(BaseRunConfig):
    """Static CassKoopmans tuning configuration for a sweep."""
    # Environment
    env_name: str = "CassKoopmans"
    max_wallclock_seconds: int = 600
    
    def make_params(self):
        return CassKoopmansParams()

    def env_class(self):
        return CassKoopmansEnv

 
@dataclass
class BrockMirmanRunConfig(BaseRunConfig):
    """Static BrockMirman run configuration for a sweep."""
    # Environment
    env_name: str = "BrockMirman"
    max_wallclock_seconds: int = 1200
    
    def make_params(self):
        return BrockMirmanParams()

    def env_class(self):
        return BrockMirmanEnv


@dataclass
class BufferStockRunConfig(BaseRunConfig):
    """Static BufferStock run configuration for a sweep."""
    # Environment
    env_name: str = "BufferStock"
    max_wallclock_seconds: int = 1200
    
    def make_params(self):
        return BufferStockParams()

    def env_class(self):
        return BufferStockEnv

@dataclass
class PortfolioChoiceRunConfig(BaseRunConfig):
    """Static PortfolioChoice run configuration for a sweep."""
    # Environment
    env_name: str = "PortfolioChoice"
    max_wallclock_seconds: int = 1200
    
    def make_params(self):
        return PortfolioChoiceParams()

    def env_class(self):
        return PortfolioChoiceEnv


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
    p.add_argument("--eval_interval_seconds", type=int, default=argparse.SUPPRESS)
    p.add_argument("--n_seeds", type=int, default=argparse.SUPPRESS)
    
    args = p.parse_args()
    
    return replace(RunConfig, **vars(args))