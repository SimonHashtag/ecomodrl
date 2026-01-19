import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from ecomodrl.benchmarks.casskoopmans.problemclass import PlanningProblem, PlanningSolvers
from ecomodrl.utils.analysis_CK import get_policy_value_functions
from ecomodrl.utils.params_loader import CassKoopmansParams
from ecomodrl.envs.casskoopmans import CassKoopmansEnv

# Setup
ALGOS = ["PPO", "DDPG", "SAC", "TD3"]
BASE_RESULT_PATH = Path(os.environ.get("RUNRESULTS_PATH", "./runs/CassKoopmans"))
TOML_PATH = Path("ecomodrl/experiments/model_params/casskoopmans.toml")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Compare")

def run_comparison():
    # 1. Setup Environment & Grid
    # We need the model structure to generate the grid and calculate errors later.
    params = CassKoopmansParams.load_params(TOML_PATH)
    model = PlanningProblem(gamma=params.beta, beta=params.beta, delta=params.delta, alpha=params.alpha, A=params.A)
    solvers = PlanningSolvers(grid_points=1000, conv=1e-5)
    
    # Generate the state grid (Capital k)
    grid = model.make_grid(solvers.grid_points)

    # Mock config required by 'build_envs' internally
    # FIX: Added 'n_envs' and 'max_eval_episode_steps' to satisfy setup_envs.py
    mock_cfg = SimpleNamespace(
        max_episode_steps=30,      
        max_eval_episode_steps=100, # Required for the eval env creation
        n_envs=1,                   # Required for the train env creation (1 is enough for dummy loading)
        state_min=grid[0], 
        state_max=grid[-1]
    )

    # 2. Load Data & Compute Errors
    algo_data = {}
    global_min, global_max = 0, -np.inf 

    for algo in ALGOS:
        json_path = BASE_RESULT_PATH / algo / "train_cfgs.json"
        
        if not json_path.exists():
            logger.warning(f"Skipping {algo}: No data found at {json_path}")
            continue
        
        logger.info(f"Processing {algo}...")
        
        with open(json_path, "r") as f:
            train_cfgs = json.load(f)
        
        euler_list = []
        
        # Process every seed for this algorithm
        for entry in train_cfgs:
            # Reconstruct policy from saved model
            policy_rl, _ = get_policy_value_functions(
                Path(entry["best_model_path"]), 
                Path(entry["vecnorm_path"]), 
                cfg=mock_cfg,        
                env=CassKoopmansEnv, 
                params=params, 
                grid=grid, 
                algo_name=algo
            )
            
            # Compute Euler Equation Errors
            euler_rl, _, _ = model.get_euler_errors(grid, policy_rl, episodes=100)
            euler_list.append(np.log10(np.abs(euler_rl)))
        
        # Store pooled data
        flat_euler = np.concatenate(euler_list)
        algo_data[algo] = flat_euler
        
        # Track global limits for unified axis
        global_min = min(global_min, np.min(flat_euler))
        global_max = max(global_max, np.max(flat_euler))

    # 3. Plotting (4x1 Vertical Stack)
    # Add buffer to limits for aesthetics
    x_min, x_max = global_min - 0.5, global_max + 0.5

    plt.style.use("seaborn-v0_8-colorblind")
    _, axes = plt.subplots(len(ALGOS), 1, figsize=(8, 10), sharex=True, sharey=False)
    
    if len(ALGOS) == 1: axes = [axes] # Safety for single algo testing

    for i, algo in enumerate(ALGOS):
        ax = axes[i]
        if algo in algo_data:
            data = algo_data[algo]
            ax.hist(data, bins=50, density=True, alpha=0.7, color="darkblue")
            ax.set_ylabel(f"Density")
            ax.set_title(f"{algo}")
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)

        # Unification magic
        ax.set_xlim(x_min, x_max)

    axes[-1].set_xlabel("log10(|Euler Error|)")
    plt.tight_layout()

    out_path = BASE_RESULT_PATH / "COMPARISON_euler_errors.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    logger.info(f"Comparison plot saved to: {out_path}")

if __name__ == "__main__":
    run_comparison()