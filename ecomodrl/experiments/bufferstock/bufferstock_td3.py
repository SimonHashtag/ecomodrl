import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch as th
import math

from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.benchmarks.bufferstock.problemclass import PlanningProblem
from ecomodrl.envs.bufferstock import BufferStockSavingsEnv
from ecomodrl.utils.analysis_BS import analyze_all_seeds
from ecomodrl.utils.params_loader import BufferStockParams
from ecomodrl.utils.learning_schedule import LinearScheduleByWallClockTime


# ----------------------------
# Run configuration
# ----------------------------
# NOTE: set custom RUNRESULTS_PATH 
#       on your Windows system via setx RUNRESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export RUNRESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/runs/BufferStock"
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


if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    cfg = RunConfig()
    params = BufferStockParams.load_params(cfg.toml_path)
    
    # TD3 Hyperparameters
    INITIAL_LR = 0.0009771130432
    N_EPISODES = 1
    BATCH_SIZE = 32
    UTD_RATIO = 2.143118403272
    TAU = 0.0136203795913
    ACTION_NOISE_STD = 0.0848123875565  # exploration_sigma
    TARGET_POLICY_NOISE = 0.0597164215216
    # max(hp["target_noise_clip"], hp["target_policy_noise"])
    # max(0.691899339891, 0.0597164215216) = 0.691899339891
    TARGET_NOISE_CLIP = 0.691899339891

    lr_schedule = LinearScheduleByWallClockTime(
        initial_lr=INITIAL_LR,
        final_lr=3e-5,
        max_wallclock_seconds=cfg.max_wallclock_seconds,
    )

    train_freq = cfg.max_episode_steps * N_EPISODES
    gradient_steps = math.ceil(cfg.n_envs * train_freq * UTD_RATIO / BATCH_SIZE)

    algo_cfg = AlgoConfig(
        name="td3",
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
            "policy_delay": 2,
            "action_noise_std": ACTION_NOISE_STD,
            "target_policy_noise": TARGET_POLICY_NOISE,
            "target_noise_clip": TARGET_NOISE_CLIP,
        },
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": {"pi": [128, 128], "qf": [128, 128]},
        },
    )
    
    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_multiple_seeds(cfg, algo_cfg, BufferStockSavingsEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, BufferStockSavingsEnv, params)
