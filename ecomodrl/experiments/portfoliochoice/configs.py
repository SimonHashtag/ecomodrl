import os

from dataclasses import dataclass, field
from pathlib import Path

from ecomodrl.benchmarks.portfoliochoice.problemclass import PlanningProblem

# ----------------------------
# Run configuration
# ----------------------------
# NOTE: set custom RUNRESULTS_PATH 
#       on your Windows system via setx RUNRESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export RUNRESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/runs/PortfolioChoice"

@dataclass
class RunConfig:
    # Paths
    out_dir: Path = Path(os.environ.get("RUNRESULTS_PATH", "./runs/PortfolioChoice"))

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
        grid = mod.make_m_grid()
        self.state_min = grid[0]
        self.state_max = grid[-1]

    @property
    def seeds(self) -> list[int]:
        """List of random seeds for multiple runs."""
        return list(range(self.base_seed, self.base_seed + self.n_seeds))