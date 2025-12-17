import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch as th

from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.benchmarks.brockmirman.problemclass import PlanningProblem
from ecomodrl.envs.brockmirman import BrockMirmanEnv
from ecomodrl.utils.analysis_BM import analyze_all_seeds
from ecomodrl.utils.params_loader import BrockMirmanParams


# ----------------------------
# Run configuration
# ----------------------------
# NOTE: set custom RUNRESULTS_PATH 
#       on your Windows system via setx RUNRESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export RUNRESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/runs/BrockMirman"
@dataclass
class RunConfig:
    # Paths
    toml_path: Path = Path("./ecomodrl/experiments/model_params/brockmirman.toml")
    out_dir: Path = Path(os.environ.get("RUNRESULTS_PATH", "./runs/BrockMirman"))

    # Training
    max_episode_steps: int = 30
    max_eval_episode_steps: int = 100
    n_envs: int = 10
    total_timesteps: int = 10_000_000
    max_processed_samples: int = 1_000_000
    eval_every: int = 10_000
    patience: int = 10
    min_delta: float = 1e-3

    # Seeds
    base_seed: int = 42
    n_seeds: int = 1

    # Use steady-state aware starting boundaries for RL algorithm
    # Intention: don't explore outside of the economically relevant state space
    state_min: float = field(init=False)
    state_max: float = field(init=False)
    def __post_init__(self):
        mod = PlanningProblem()
        grid = mod.make_k_grid()
        self.state_min = grid[0]
        self.state_max = grid[-1]

    @property
    def seeds(self) -> list[int]:
        """List of random seeds for multiple runs."""
        return list(range(self.base_seed, self.base_seed + self.n_seeds))


if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = RunConfig()
    params = BrockMirmanParams.load_params(cfg.toml_path)

    algo_cfg = AlgoConfig(
        name="ppo",
        policy="MlpPolicy",
        algo_params={
            "gamma": params.beta,
            "n_steps": cfg.max_episode_steps,
            "batch_size": cfg.n_envs * cfg.max_episode_steps // 5,
            "n_epochs": 2,
            "learning_rate": 3.5e-4,
            "ent_coef": 4.4e-5,
            "gae_lambda": 0.925,
            "clip_range": 0.27,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "log_std_init": -2.0,
        },
    )

    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, BrockMirmanEnv, params)
    analyze_all_seeds(cfg, algo_cfg, BrockMirmanEnv, params, n_boot=1000, alpha=0.05, bootstrap_seed=12345)
