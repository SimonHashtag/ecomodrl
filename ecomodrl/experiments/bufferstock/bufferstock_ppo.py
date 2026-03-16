import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch as th

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


if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    cfg = RunConfig()
    params = BufferStockParams.load_params(cfg.toml_path)

    # PPO Hyperparameters
    INITIAL_LR = 3.5e-4
    N_EPISODES = 2
    N_EPOCHS = 10
    NUM_MINIBATCHES = 4
    CLIP_RANGE = 0.2
    ENT_COEF = 4.4e-5
    GAE_LAMBDA = 0.95
    VF_COEF = 0.5

    lr_schedule = LinearScheduleByWallClockTime(
        initial_lr=INITIAL_LR,
        final_lr=3e-5,
        max_wallclock_seconds=cfg.max_wallclock_seconds,
    )

    algo_cfg = AlgoConfig(
        name="ppo",
        policy="MlpPolicy",
        algo_params={
            "gamma": params.beta,
            "n_steps": cfg.max_episode_steps * N_EPISODES,
            "batch_size": (cfg.n_envs * cfg.max_episode_steps * N_EPISODES) // NUM_MINIBATCHES,
            "n_epochs": N_EPOCHS,
            "learning_rate": lr_schedule,
            "ent_coef": ENT_COEF,
            "gae_lambda": GAE_LAMBDA,
            "clip_range": CLIP_RANGE,
            "vf_coef": VF_COEF,
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

    train_multiple_seeds(cfg, algo_cfg, BufferStockSavingsEnv, params, verbose=1)
    analyze_all_seeds(cfg, algo_cfg, BufferStockSavingsEnv, params)