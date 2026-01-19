import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch as th

from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem
from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.utils.analysis_CK import analyze_all_seeds
from ecomodrl.utils.params_loader import CassKoopmansParams


# ----------------------------
# Run configuration
# ----------------------------
# NOTE: set custom RUNRESULTS_PATH 
#       on your Windows system via setx RUNRESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export RUNRESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/runs/CassKoopmans"
@dataclass
class RunConfig:
    # Paths
    toml_path: Path = Path("./ecomodrl/experiments/model_params/casskoopmans.toml")
    out_dir: Path = Path(os.environ.get("RUNRESULTS_PATH", "./runs/CassKoopmans"))

    # Training
    max_episode_steps: int = 30
    max_eval_episode_steps: int = 100
    n_envs: int = 10
    total_timesteps: int = 10_000_000
    max_processed_samples: int = 1_000_000
    eval_every: int = 10_000
    patience: int = 20
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


if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    cfg = RunConfig()
    params = CassKoopmansParams.load_params(cfg.toml_path)
    
    algo_cfg = AlgoConfig(
        name="td3",
        policy="MlpPolicy",
        algo_params={
            "learning_rate": 0.0006207652006,
            "buffer_size": 200_000,
            "batch_size": 64,
            "tau": 0.0139904290884,
            "gamma": params.beta,
            "train_freq": (cfg.max_episode_steps * 2, "step"), # "step" is crucial for vectorized envs
            "gradient_steps": 8,
            "learning_starts": 10_000,
            "policy_delay": 2,
            "action_noise_std": 0.0515070057901,
            "target_policy_noise": 0.0784148244779,
            "target_noise_clip": 0.7099592918582,
        },
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": {"pi": [64, 64], "qf": [64, 64]},
        },
    )
    
    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, CassKoopmansEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, CassKoopmansEnv, params, n_boot=1000, alpha=0.05, bootstrap_seed=12345)