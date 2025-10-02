import logging
from dataclasses import dataclass
from pathlib import Path

import multiprocessing as mp
import torch as th

from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.utils.params_loader import CassKoopmansParams
from ecomodrl.utils.analysis import analyze_all_seeds

@dataclass
class RunConfig:
    toml_path: str = "./ecomodrl/experiments/model_params/casskoopmans.toml"
    out_dir: Path = Path("D:/ECOMODRL/runs/CassKoopmans/PPO")
    max_episode_steps: int = 30
    max_eval_episode_steps: int = 100
    n_envs: int = 10
    total_timesteps: int = 400_000
    eval_every: int = 10_000
    patience: int = 10
    min_delta: float = 1e-3
    state_min: float = 1.0
    state_max: float = 15.0
    base_seed: int = 42
    n_seeds: int = 10
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
    params = CassKoopmansParams.load_params(cfg.toml_path)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
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
    # algo_cfg = AlgoConfig(
    #     name="td3",
    #     policy="MlpPolicy",
    #     algo_params={
    #         "learning_rate": 3e-4,
    #         "buffer_size": 1_000_000,
    #         "batch_size": cfg.n_envs * cfg.max_episode_steps // 4,
    #         "tau": 0.005,
    #         "gamma": params.beta,
    #         "train_freq": (100, "step"), # "step" is crucial for vectorized envs
    #         "gradient_steps": 1,
    #         "learning_starts": 20_000,
    #         "policy_delay": 2,
    #         "action_noise_std": 0.01,
    #     },
    #     policy_kwargs={
    #         "activation_fn": th.nn.ReLU,
    #         "net_arch": [32, 32],
    #     },
    # )
    
    train_multiple_seeds(cfg, algo_cfg, CassKoopmansEnv, params)
    analyze_all_seeds(cfg, algo_cfg, CassKoopmansEnv, params, n_boot=1000, alpha=0.05, bootstrap_seed=12345)