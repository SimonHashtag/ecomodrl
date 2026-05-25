import argparse
import os

from dataclasses import dataclass, field
from pathlib import Path

from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem

# ----------------------------
# Run configuration
# ----------------------------
# NOTE: set custom RUNRESULTS_PATH 
#       on your Windows system via setx RUNRESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export RUNRESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/runs/CassKoopmans"

@dataclass
class RunConfig:
    # Name
    env_name: str = "CassKoopmans"
    algo_name: str = "ppo"

    # Paths
    out_dir: Path = Path(os.environ.get("RUNRESULTS_PATH", "./runs/CassKoopmans"))

    # Training
    max_episode_steps: int = 30
    max_eval_episode_steps: int = 100
    n_envs: int = 10
    total_timesteps: int = 1_000_000_000
    max_wallclock_seconds: int = 600
    eval_every: int = 10_000
    patience: int = 2000
    min_delta: float = 1e-3

    # Seeds
    base_seed: int = 200
    n_seeds: int = 10

    # Use steady-state aware starting boundaries for RL algorithm
    # Intention: don't explore outside of the economically relevant state space
    state_min: float = field(init=False)
    state_max: float = field(init=False)
    def __post_init__(self):
        mod = PlanningProblem()
        grid = mod.make_grid()
        self.state_min = grid[0]
        self.state_max = grid[-1]

    @property
    def seeds(self) -> list[int]:
        """List of random seeds for multiple runs."""
        return list(range(self.base_seed, self.base_seed + self.n_seeds))
    

def parse_run_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo_name", type=str, default="ppo", choices=["ppo", "ddpg", "td3", "sac"])
    p.add_argument("--out_dir", type=Path, default=Path(os.environ.get("RUNRESULTS_PATH", "./runs/CassKoopmans")))
    p.add_argument("--max_episode_steps", type=int, default=30)
    p.add_argument("--max_eval_episode_steps", type=int, default=100)
    p.add_argument("--n_envs", type=int, default=10)
    p.add_argument("--total_timesteps", type=int, default=1_000_000_000)
    p.add_argument("--max_wallclock_seconds", type=int, default=600)
    p.add_argument("--eval_every", type=int, default=10_000)
    p.add_argument("--patience", type=int, default=2000)
    p.add_argument("--n_seeds", type=int, default=10)
    
    args = p.parse_args()
    
    return RunConfig(**vars(args))