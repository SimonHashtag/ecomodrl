import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

import multiprocessing as mp
import torch as th

from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.envs.bufferstock import BufferStockSavingsEnv
from ecomodrl.utils.params_loader import BufferStockParams
from ecomodrl.utils.analysis_CK import analyze_all_seeds

# ----------------------------
# Run configuration
# ----------------------------
# NOTE: set custom RUNRESULTS_PATH 
#       on your Windows system via setx RUNRESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export RUNRESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/runs/CassKoopmans"
@dataclass
class RunConfig:
    toml_path: str = "./ecomodrl/experiments/model_params/bufferstock.toml"
    out_dir: Path = Path(os.environ.get("RUNRESULTS_PATH", "./runs/BufferStock"))
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

if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
    cfg = RunConfig()
    params = BufferStockParams.load_params(cfg.toml_path)
    
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
            "net_arch": {"pi": [32, 32], "vf": [32, 32]},
            "log_std_init": -2.0,  # smaller initial exploration to avoid extremes
        },
    )

    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    train_multiple_seeds(cfg, algo_cfg, BufferStockSavingsEnv, params)
    #analyze_all_seeds(cfg, algo_cfg, CassKoopmansEnv, params, n_boot=1000, alpha=0.05, bootstrap_seed=12345)