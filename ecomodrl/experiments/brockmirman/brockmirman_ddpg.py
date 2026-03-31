import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch as th
import math

from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.benchmarks.brockmirman.problemclass import PlanningProblem
from ecomodrl.envs.brockmirman import BrockMirmanEnv
from ecomodrl.utils.analysis_BM import analyze_all_seeds
from ecomodrl.utils.params_loader import BrockMirmanParams
from ecomodrl.utils.learning_schedule import LinearScheduleByWallClockTime


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
    total_timesteps: int = 1_000_000_000
    max_wallclock_seconds: int = 1200
    eval_every: int = 10_000
    patience: int = 2000
    min_delta: float = 1e-3
    n_eval_episodes: int = 5  # average over multiple shock sequences per eval (stochastic env)

    # Seeds
    base_seed: int = 200
    n_seeds: int = 10

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
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    cfg = RunConfig()
    params = BrockMirmanParams.load_params(cfg.toml_path)
    
    # DDPG Hyperparameters
    INITIAL_LR = 0.0027051726975
    N_EPISODES = 8
    BATCH_SIZE = 32
    UTD_RATIO = 1.5855303302638
    TAU = 0.0175568248496
    ACTION_NOISE_STD = 0.0630440355355  # exploration_sigma

    lr_schedule = LinearScheduleByWallClockTime(
        initial_lr=INITIAL_LR,
        final_lr=3e-5,
        max_wallclock_seconds=cfg.max_wallclock_seconds,
    )

    train_freq = cfg.max_episode_steps * N_EPISODES
    gradient_steps = math.ceil(cfg.n_envs * train_freq * UTD_RATIO / BATCH_SIZE)

    algo_cfg = AlgoConfig(
        name="ddpg",
        policy="MlpPolicy",
        algo_params={
            "learning_rate": lr_schedule,
            "buffer_size": 200_000,
            "batch_size": BATCH_SIZE,
            "tau": TAU,
            "gamma": params.beta,
            "max_grad_norm": 0.5,
            "train_freq": (train_freq, "step"),
            "gradient_steps": gradient_steps,
            "learning_starts": 10_000,
            "action_noise_std": ACTION_NOISE_STD,
        },
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": {"pi": [128, 128], "qf": [128, 128]},
        },
    )
    
    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, BrockMirmanEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, BrockMirmanEnv, params)
