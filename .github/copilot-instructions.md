# EcoModRL - AI Coding Guidelines

## Project Overview
Deep reinforcement learning framework for solving economic models. Trains RL agents (PPO, TD3, DDPG, SAC via Stable-Baselines3) to learn optimal policies in economic environments, then compares against analytical/numerical benchmarks.

## Architecture

### Core Directory Structure
- **`ecomodrl/envs/`** - Gymnasium environments for economic models (state=capital/wealth, action=consumption share ∈(0,1))
- **`ecomodrl/benchmarks/{model}/`** - Economic theory: `problemclass.py` defines model primitives (utility, production, Euler equation) and `PlanningSolvers` for analytical solutions (VFI, EGM)
- **`ecomodrl/experiments/{model}/`** - Training scripts with tuned hyperparameters
- **`ecomodrl/tune/`** - DEHB hyperparameter optimization (run via CLI with `--algo_name` flag)
- **`ecomodrl/utils/`** - Shared utilities: early stopping, learning schedules, metrics, analysis

### Key Patterns

**Parameter Loading**: Economic model parameters live in TOML files (`experiments/model_params/*.toml`) and are loaded via typed dataclasses in [params_loader.py](ecomodrl/utils/params_loader.py):
```python
params = CassKoopmansParams.load_params("./ecomodrl/experiments/model_params/casskoopmans.toml")
```

**Environment Creation**: Use `build_envs()` from [setup_envs.py](ecomodrl/envs/setup_envs.py) which handles VecNormalize, TimeLimit wrapping, and reward flooring:
```python
train_env, eval_env = build_envs(CassKoopmansEnv, params=params, cfg=cfg, seed=42)
```

**Training**: `train_multiple_seeds()` parallelizes across seeds using `ProcessPoolExecutor`. Each seed outputs to `{out_dir}/seed_{N}/`.

**Analysis workflow** (two-stage):
1. Each experiment script calls `analyze_all_seeds()` after training → saves `metrics_analysis.json` with raw data and aggregated statistics
2. Run `casskoopmans_compare.py` after all algorithms finish → loads JSON files and generates comparison plots + LaTeX table

## Economic Model Conventions

| Model | State | Action | Benchmark Solver |
|-------|-------|--------|------------------|
| Cass-Koopmans | log(capital) | consumption share | EGM, VFI |
| Buffer-Stock | normalized market resources | consumption share | EGM |
| Brock-Mirman | (capital, productivity) | consumption share | VFI |
| Portfolio Choice | wealth | (consumption share, risky share) | VFI |

**Discount factor**: Environments use `beta` (economic convention), RL algos use `gamma` (ML convention) - these must match: `algo_params={"gamma": params.beta}`.

## Running Experiments

**Train a single algorithm** (from project root):
```bash
python ecomodrl/experiments/casskoopmans/casskoopmans_ppo.py
```

**Train all algorithms and compare** (SLURM cluster):
```bash
sbatch ecomodrl/experiments/casskoopmans/casskoopmans_training.sh
```

**Generate comparison plots** (after training all algorithms):
```bash
python ecomodrl/experiments/casskoopmans/casskoopmans_compare.py
```
This loads `metrics_analysis.json` from each algorithm's output folder and produces:
- `1_learning_curves.pdf` - Regret curves (log scale)
- `2_policy_functions.pdf` - 2x2 policy comparison
- `3_euler_errors.pdf` - Euler error histograms
- `4_cev_analysis.pdf` - CEV with state density
- `comparison_table.tex` - LaTeX summary table

**Hyperparameter tuning**:
```bash
python ecomodrl/tune/casskoopmans.py --algo_name=ppo --total_cost=3600
```

**Output paths**: Configurable via `RUNRESULTS_PATH` / `TUNERESULTS_PATH` env vars, default to `./runs/` and `./tune_results/`.

**TensorBoard**: `tensorboard --logdir=runs/CassKoopmans/PPO`

## Key Implementation Details

- **State representation**: Environments return log-transformed states for numerical stability (`np.log(capital)`)
- **Early stopping**: `EarlyStoppingCallback` monitors discounted returns, saves best model and VecNormalize stats
- **Learning rate schedule**: `LinearScheduleByProcessedSamples` decays LR based on total samples processed
- **Off-policy gradient clipping**: TD3/DDPG/SAC use custom `add_grad_norm_rescaling()` wrapper since SB3 doesn't support it natively
- **Steady-state aware bounds**: `RunConfig.__post_init__` computes economically relevant state bounds from `PlanningProblem().make_grid()`

## Adding a New Economic Model

1. Create `ecomodrl/envs/{model}.py` - Gymnasium env with `model_params`, `fixed_start`, `reset_state`, `state_min/max` interface
2. Create `ecomodrl/benchmarks/{model}/problemclass.py` - Define `PlanningProblem` with utility, production, Euler methods
3. Create `ecomodrl/experiments/model_params/{model}.toml` - Default parameters
4. Add `{Model}Params` dataclass to [params_loader.py](ecomodrl/utils/params_loader.py)
5. Create experiment script following pattern in [casskoopmans_ppo.py](ecomodrl/experiments/casskoopmans/casskoopmans_ppo.py)

## Dependencies
- Python 3.11+ (required for `tomllib`)
- Use `conda-lock install -n ENV requirements.conda-lock.yml` for reproducible env
- Core: `stable-baselines3`, `gymnasium`, `torch`, `dehb` (tuning)
