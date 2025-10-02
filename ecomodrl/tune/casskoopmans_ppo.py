import time
import uuid
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Union, List
import json
import os
import pickle
import math

import torch as th
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from dehb import DEHB

from ecomodrl.algorithms.training import AlgoConfig, train_multiple_seeds
from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.utils.params_loader import CassKoopmansParams
    
# ----------------------------
# Run & algo configuration
# ----------------------------
@dataclass
class RunConfig:
    toml_path: str = "./ecomodrl/experiments/model_params/casskoopmans.toml"
    out_dir: Path = Path("D:/ECOMODRL/tune/CassKoopmans/PPO")
    max_episode_steps: int = 20
    max_eval_episode_steps: int = 100
    n_envs: int = 10
    total_timesteps: int = 200_000
    eval_every: int = 20_000
    patience: int = 5
    min_delta: float = 1e-3
    state_min: float = 1.0
    state_max: float = 15.0
    base_seed: int = 42
    n_seeds: int = 10

    @property
    def seeds(self) -> list[int]:
        return list(range(self.base_seed, self.base_seed + self.n_seeds))


# ----------------------------
# ConfigSpace
# ----------------------------
def divisors(n: int) -> List[int]:
    """Positive divisors of n."""
    ds = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            ds.add(i)
            ds.add(n // i)
    return sorted(ds)

def candidate_minibatches(B: int) -> List[int]:
    """Prefer small set of divisors to keep the search compact."""
    preferred = [2, 4, 5, 8, 10]
    avail = [m for m in preferred if B % m == 0]
    if not avail:
        # fallback: pick up to 5 smallest divisors >= 2
        avail = [m for m in divisors(B) if m >= 2]
        avail = avail[:5] if avail else [2]  # last resort
    return avail

# --- Seed ramping & CRN helpers ---
def seeds_for_fidelity(fidelity: int, min_fidelity: int, max_fidelity: int) -> int:
    phi = (fidelity - min_fidelity) / max(1, (max_fidelity - min_fidelity))
    if phi < 0.05:
        return 1
    elif phi < 0.1:
        return 2
    elif phi < 0.4:
        return 5
    else:
        return 10  # TODO: Let be set by Configs

def get_configspace(cfg: RunConfig, seed=None) -> CS.ConfigurationSpace:
    cs = CS.ConfigurationSpace(seed)

    B = cfg.n_envs * cfg.max_episode_steps
    minibatches = candidate_minibatches(B)

    learning_rate = CSH.UniformFloatHyperparameter("learning_rate", lower=5e-5, upper=1e-3, log=True)
    clip_range    = CSH.UniformFloatHyperparameter("clip_range", lower=0.10, upper=0.30)
    gae_lambda    = CSH.UniformFloatHyperparameter("gae_lambda", lower=0.90, upper=0.98)
    ent_coef      = CSH.UniformFloatHyperparameter("ent_coef", lower=1e-6, upper=1e-2, log=True)
    n_epochs      = CSH.UniformIntegerHyperparameter("n_epochs", lower=2, upper=6)
    num_minibatches = CSH.CategoricalHyperparameter("num_minibatches", minibatches)

    cs.add([learning_rate, clip_range, gae_lambda, ent_coef, n_epochs, num_minibatches])
    return cs

# ----------------------------
# Target function for DEHB
# ----------------------------
def target_function(
    hp: Union[CS.Configuration, Dict, List, np.ndarray],
    fidelity: Union[int, float] = None,
    **kwargs
) -> Dict:
    """
    Returns:
        dict with keys:
          - 'fitness' (lower is better)  -> negative avg best eval return
          - 'cost' (wall time in seconds)
          - 'info' (optional extras)
    """
    base_cfg: RunConfig = kwargs["cfg"]
    algo_cfg: AlgoConfig = kwargs["algo_cfg"]
    params = kwargs["params"]
    EnvClass = kwargs["env"]

    # Convert hp into a plain dict
    if isinstance(hp, CS.Configuration):
        hp = dict(hp)
    elif not isinstance(hp, dict):
        # handle arrays/lists coming from DEHB as a fallback
        hp = {k.name: v for k, v in zip(kwargs["cs"].get_hyperparameters(), hp)}

    # Derive rollout/batch geometry
    n_steps = base_cfg.max_episode_steps
    B = base_cfg.n_envs * n_steps
    M = int(hp["num_minibatches"])
    batch_size = B // M
    n_epochs = int(hp["n_epochs"])

    # Keep updates per iteration U in a healthy band
    U = n_epochs * M
    if U > 10:
        # clamp epochs to keep U <= 10 (never less than 1, but our space is 2..6)
        n_epochs = max(1, min(n_epochs, 10 // M))
        U = n_epochs * M

    # Build a per-trial run config (unique output dir)
    trial_dir = base_cfg.out_dir / f"trial_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    trial_cfg = replace(base_cfg, out_dir=trial_dir)
    if fidelity is not None:
        # set the per-trial training budget
        trial_cfg = replace(trial_cfg, total_timesteps=int(fidelity))
        # Reduce number of training runs for low fertilities
        min_fid = kwargs["min_fidelity"]
        max_fid = kwargs["max_fidelity"]
        n_seeds_for_trial = seeds_for_fidelity(int(fidelity), int(min_fid), int(max_fid))
        trial_cfg = replace(
            trial_cfg,
            n_seeds=int(n_seeds_for_trial),
            # keep base_seed unchanged (e.g., 42) so the first k seeds are identical across configs
            base_seed=base_cfg.base_seed,
        )
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Prepare algo params for this trial
    trial_algo = replace(algo_cfg)
    trial_algo.algo_params = {**(algo_cfg.algo_params or {})}
    trial_algo.algo_params.update({
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "learning_rate": float(hp["learning_rate"]),
        "gae_lambda": float(hp["gae_lambda"]),
        "clip_range": float(hp["clip_range"]),
        "ent_coef": float(hp["ent_coef"]),
    })

    trial_algo.policy_kwargs = {**(algo_cfg.policy_kwargs or {})}

    # Run training
    start = time.time()
    try:
        train_multiple_seeds(
            trial_cfg, trial_algo, EnvClass, params,
            seed_parallel=False, max_workers=1)
    except Exception as e:
        # Fail-safe: return a bad fitness but with measured cost
        cost = time.time() - start
        return {"fitness": 1e9, "cost": cost, "info": {"error": repr(e)}}
    cost = time.time() - start

    # Read results produced by your training harness
    train_cfgs_path = trial_dir / "train_cfgs.json"
    if not train_cfgs_path.exists():
        return {"fitness": 1e9, "cost": cost, "info": {"error": "train_cfgs.json missing"}}

    with open(train_cfgs_path, "r") as f:
        train_cfgs = json.load(f)   # list of dicts with "best_eval_return"

    if not train_cfgs:
        return {"fitness": 1e9, "cost": cost, "info": {"error": "empty train_cfgs.json"}}

    avg_best_return = sum(s["best_eval_return"] for s in train_cfgs) / len(train_cfgs)

    # Minimize
    result = {
        "fitness": -float(avg_best_return),
        "cost": cost,
        "info": {
            "B": B,
            "num_minibatches": M,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "updates_per_iter": U,
            "trial_dir": str(trial_dir),
            "n_seeds": trial_cfg.n_seeds,
            "seeds": trial_cfg.seeds,
        },
    }
    return result

if __name__ == "__main__":
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
            # will be overwritten by target_function per trial
            "gamma": params.beta,
            "n_steps": cfg.max_episode_steps,
            "batch_size": cfg.n_envs * cfg.max_episode_steps // 2,
            "learning_rate": 3e-4,
            "ent_coef": 0.0,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": {"pi": [32, 32], "vf": [32, 32]},
            "log_std_init": -2.0,
        },
    )

    # Compact space tied to your rollout geometry
    cs = get_configspace(cfg, seed=cfg.base_seed)

    eta = 3
    rungs = 3
    min_fidelity = max(cfg.total_timesteps // (eta**(rungs-1)), cfg.eval_every)
    max_fidelity = cfg.total_timesteps
    
    n_workers = max(1, os.cpu_count() - 1 or 1)

    dehb = DEHB(
        f=target_function,
        cs=cs,
        dimensions=len(list(cs.values())),
        min_fidelity=min_fidelity,
        max_fidelity=max_fidelity,
        n_workers=n_workers,
        output_path=cfg.out_dir,
        log_level="INFO",
        eta=eta,
        crossover_prob=0.5,
        mutation_factor=0.5,
    )

    traj, runtime, history = dehb.run(
        total_cost=500,
        cfg=cfg,
        algo_cfg=algo_cfg,
        params=params,
        env=CassKoopmansEnv,
        cs=cs,
        min_fidelity=min_fidelity,   
        max_fidelity=max_fidelity, 
    )

    # Save optimization trace history
    name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(dehb.start))
    with open(os.path.join(cfg.out_dir, f"history_{name}.pkl"), "wb") as f:
        pickle.dump(history, f)