# ----------------------------
# Run configuration
# ----------------------------
# NOTE: set custom RUNRESULTS_PATH 
#       on your Windows system via setx RUNRESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export RUNRESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/runs/PortfolioChoice"

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig:
    out_dir: Path = Path(os.environ.get("RUNRESULTS_PATH", "./runs/PortfolioChoice"))
    max_episode_steps: int = 10
    max_eval_episode_steps: int = 30
    n_envs: int = 10
    total_timesteps: int = 400_000
    eval_every: int = 10_000
    patience: int = 20
    min_delta: float = 1e-3
    state_min: float = 1.0
    state_max: float = 10.0
    base_seed: int = 42
    n_seeds: int = 1
    @property
    def seeds(self) -> list[int]:
        return list(range(self.base_seed, self.base_seed + self.n_seeds))