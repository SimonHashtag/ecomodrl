# EcoModRL
This repository accompanies the working paper “Deep Reinforcement Learning in Dynamic Economic Models: An Empirical Study”. The project evaluates deep reinforcement learning (RL) as a numerical solution method for dynamic economic models, with systematic algorithm comparisons and benchmark solutions.

> Working paper: *link forthcoming*

## Models and benchmarks

Four dynamic economic models are solved with deep RL and evaluated against high-accuracy numerical benchmarks:

| Model | CLI name | Economics | Benchmark solver |
|---|---|---|---|
| Cass–Koopmans | `casskoopmans` | deterministic optimal growth | EGM |
| Brock–Mirman | `brockmirman` | stochastic growth, AR(1) technology | EGM |
| Buffer-stock saving | `bufferstock` | income risk, occasionally binding borrowing constraint | EGM |
| Portfolio choice | `portfoliochoice` | buffer-stock saving + risky/safe asset allocation | nested EGM + KKT root-solving |

Each model is exposed as a Gymnasium environment ([ecomodrl/envs/envs.py](ecomodrl/envs/envs.py)) and solved with four algorithms: PPO, DDPG, TD3, and SAC (stable-baselines3). Benchmark solutions come from endogenous grid method (EGM) solvers in [ecomodrl/models/solvers.py](ecomodrl/models/solvers.py). Accuracy and welfare are measured by Euler-equation errors (mean log10 absolute error), the Wasserstein-1 distance between the RL and benchmark Euler-error distributions, and the consumption-equivalent welfare cost (CEV) of the RL policy. All results are aggregated across seeds using interquartile means with bootstrap confidence intervals. Model derivations are documented in the notebooks under [ecomodrl/models/theory/](ecomodrl/models/theory/).

## Repository structure

```
ecomodrl/
  algorithms/    # SB3 training loop, multi-seed runner, evaluation callbacks
  analysis/      # accuracy metrics, plotting, LaTeX tables
  cluster/       # SLURM scripts: full tune -> train -> analyze pipeline per model
  configs/       # model calibrations + run/tune/analysis configs (CLI)
  envs/          # Gymnasium environments + vectorized env setup
  experiments/   # training entry points (one per model) + analyze_experiments.py
  models/        # economic models, EGM benchmark solvers, theory notebooks
  tune/          # DEHB hyperparameter tuning entry points
  utils/         # tuned-hyperparameter loading
```

The package is not installed via pip. Run all commands from the repository root using `python -m`, as shown below.

## Installation

Requires Python 3.11.

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

The pinned versions in [requirements.txt](requirements.txt) reproduce the paper environment (Python 3.11.8, torch 2.8.0, numpy 1.26.4).

## Quickstart

A fast example: PPO on Cass–Koopmans with 4 seeds. Training runs on a wall-clock budget (300 s per seed for this model); seeds train in parallel across CPU cores:

```
python -m ecomodrl.experiments.casskoopmans --algo_name ppo --n_seeds 4
python -m ecomodrl.experiments.analyze_experiments --env casskoopmans --algo_name ppo --n_seeds 4
```

`--n_seeds` must match between the two calls. Outputs land under `./runs` (override with the `RUNRESULTS_PATH` environment variable):

- **Checkpoints:** `runs/CassKoopmans/PPO/seed_<n>/` — best model (`model.zip`), normalization stats (`vecnormalize.pkl`), evaluation history (`eval_history.npz`), TensorBoard event files — plus run metadata in `train_cfgs.json`.
- **Metrics:** `runs/CassKoopmans/PPO/analysis_summary.json` — aggregated accuracy and welfare statistics.
- **Figures and tables:** `runs/CassKoopmans/figures/` — `policy.png`, `cev.png`, `euler_errors.png`, `convergence.png`, `table.tex`, `CassKoopmans_comparison_table.tex`.

## Hyperparameter tuning (optional)

Hyperparameters are tuned with DEHB, using wall-clock training time as the multi-fidelity budget:

```
python -m ecomodrl.tune.casskoopmans --algo_name ppo --total_cost 3600
```

This writes `tune_results/<Model>/<ALGO>/incumbent.json` (override the root with `TUNERESULTS_PATH`), which training picks up automatically. Without tuning results, training falls back to default hyperparameters (the quickstart works either way).

## Full replication

The CLI defaults are the paper settings: `--n_seeds 100`, `--hidden_nodes 128`, `--n_envs 10`, with wall-clock budgets of 300 s (Cass–Koopmans), 600 s (Brock–Mirman, buffer-stock), and 1200 s (portfolio choice) per seed. To replicate a model's results, run tune, train, and analyze for each algorithm (`ppo`, `ddpg`, `td3`, `sac`), substituting the model's CLI name in the commands above.

The SLURM scripts in [ecomodrl/cluster/](ecomodrl/cluster/) run the entire pipeline: DEHB tuning (4 h per algorithm), training, and analysis for all four algorithms as a single job per model on an HPC cluster.

## Extending to new models

1. Implement the model as an `EconomicModel` subclass in [ecomodrl/models/](ecomodrl/models/) (see [base_model.py](ecomodrl/models/base_model.py)) and add a Gymnasium environment in [ecomodrl/envs/envs.py](ecomodrl/envs/envs.py).
2. Add a calibration dataclass in [ecomodrl/configs/parameters.py](ecomodrl/configs/parameters.py) and run/tune/analysis configs in [ecomodrl/configs/](ecomodrl/configs/).
3. Add entry points mirroring [experiments/casskoopmans.py](ecomodrl/experiments/casskoopmans.py) and [tune/casskoopmans.py](ecomodrl/tune/casskoopmans.py) (each ~7 lines), and register the model in [analyze_experiments.py](ecomodrl/experiments/analyze_experiments.py).

## Citation

If you use this code, please cite the working paper:

```bibtex
@techreport{Haastert.2026,
  title       = {Deep Reinforcement Learning in Dynamic Economic Models: An Empirical Study},
  author      = {Haastert, Simon and Micheel, Ruven},
  year        = {2026},
  institution = {Working paper},
}
```

## License

MIT — see [LICENSE](LICENSE).
