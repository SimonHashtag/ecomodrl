import logging
import os
import pickle
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from dehb import DEHB

from ecomodrl.envs.casskoopmans import CassKoopmansEnv
from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem
from ecomodrl.tune.tuning import get_algo_fixedcfgs, ALGO_REGISTRY, target_function
from ecomodrl.utils.params_loader import CassKoopmansParams

# ----------------------------
# Run configuration
# ----------------------------
# NOTE: set custom TUNERESULTS_PATH 
#       on your Windows system via setx TUNERESULTS_PATH "my_custom_tune_results_path" (close & reopen terminal afterwards)
#       on your mac/linux: export TUNERESULTS_PATH="my_custom_tune_results_path"
#       otherwise defaults to relative path "./ecomodrl/tune_results/CassKoopmans"

@dataclass
class RunConfig:
    """Static run configuration for a sweep."""
    toml_path: str = "./ecomodrl/experiments/model_params/casskoopmans.toml"
    out_dir: Path = Path(os.environ.get("TUNERESULTS_PATH", "./tune_results/CassKoopmans")) 
    max_episode_steps: int = 30
    max_eval_episode_steps: int = 100
    n_envs: int = 10
    total_timesteps: int = 10_000_000
    max_processed_samples: int = 1_000_000
    eval_every: int = 10_000
    patience: int = 10
    min_delta: float = 1e-3
    
    # Use steady-state aware starting boundaries for RL algorithm
    # Intention: don't explore outside of the economically relevant state space
    state_min: float = field(init=False)
    state_max: float = field(init=False)
    def __post_init__(self):
        mod = PlanningProblem()
        grid = mod.make_grid()
        self.state_min = grid[0]
        self.state_max = grid[-1]
    
    # Seeds
    base_seed: int = 42
    n_seeds: int = 10

    @property
    def seeds(self) -> list[int]:
        """Deterministic seed set (for common random numbers across trials)."""
        return list(range(self.base_seed, self.base_seed + self.n_seeds))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    algo_name = "ppo"

    cfg = RunConfig()
    params = CassKoopmansParams.load_params(cfg.toml_path)

    algo_cfg = get_algo_fixedcfgs(algo_name, params)
    
    cfg = replace(cfg, out_dir=cfg.out_dir / algo_cfg.name.upper())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    space_fn, _ = ALGO_REGISTRY[algo_cfg.name.lower()]
    cs = space_fn(cfg, seed=cfg.base_seed)

    eta = 3
    rungs = 4
    min_fidelity = max(cfg.max_processed_samples // (eta**(rungs-1)), cfg.eval_every)
    max_fidelity = cfg.max_processed_samples
    n_workers = max(1, (os.cpu_count() or 2) - 1)

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
    out_file = cfg.out_dir / f"history_{name}.pkl"
    with open(os.path.join(out_file), "wb") as f:
        pickle.dump(history, f)