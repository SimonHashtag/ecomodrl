# ----------------------------
# Run configuration
# ----------------------------
# NOTE: set custom RUNRESULTS_PATH 
#       on your Windows system via setx RUNRESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export RUNRESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/runs/BufferStock"

import os
from dataclasses import dataclass, field
from pathlib import Path

from ecomodrl.benchmarks.bufferstock.problemclass import PlanningProblem

@dataclass
class RunConfig:
    # Paths
    toml_path: Path = Path("./ecomodrl/experiments/model_params/bufferstock.toml")
    out_dir: Path = Path(os.environ.get("RUNRESULTS_PATH", "./runs/BufferStock"))

    # Training
    max_episode_steps: int = 20
    max_eval_episode_steps: int = 100
    n_envs: int = 10
    total_timesteps: int = 1_000_000_000
    max_wallclock_seconds: int = 1200
    eval_every: int = 10_000
    patience: int = 2000
    min_delta: float = 1e-3
    n_eval_episodes: int = 5  # average over multiple shock sequences per eval (stochastic env)

    # Seeds
    base_seed: int = 200
    n_seeds: int = 10

    # Use economically meaningful starting boundaries for RL training.
    # The EGM solver grid starts at m≈0 (borrowing constraint) where u(c)→-∞
    # regardless of policy. For RL, use a lower bound well above the natural
    # borrowing limit (θ_low=0.3) so that rewards are informative.
    state_min: float = field(init=False)
    state_max: float = field(init=False)
    def __post_init__(self):
        mod = PlanningProblem()
        grid = mod.make_grid(100)
        self.state_min = 1.0
        self.state_max = grid[-1]

    @property
    def seeds(self) -> list[int]:
        """List of random seeds for multiple runs."""
        return list(range(self.base_seed, self.base_seed + self.n_seeds))