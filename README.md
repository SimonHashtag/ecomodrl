# EcoModRL
This repository accompanies the working paper “Deep Reinforcement Learning in Dynamic Economic Models: An Empirical Study”. The project evaluates deep reinforcement learning (RL) as a numerical solution method for dynamic economic models, with systematic algorithm comparisons and benchmark solutions.

> Working paper: *link forthcoming*

## Models and benchmarks

Four dynamic economic models are solved with deep RL and evaluated against high-accuracy numerical benchmarks:

| Model | CLI name | Economics | Benchmark solver |
|---|---|---|---|
| Cass–Koopmans | `casskoopmans` | deterministic growth | EGM |
| Brock–Mirman | `brockmirman` | stochastic growth, AR(1) technology | EGM |
| Buffer-stock saving | `bufferstock` | income risk, occasionally binding borrowing constraint | EGM |
| Portfolio choice | `portfoliochoice` | buffer-stock saving + asset allocation | nested EGM + KKT root-solving |

Each model in [ecomodrl/models](ecomodrl/models/) is translated into a [Gymnasium](https://gymnasium.farama.org/) environment ([ecomodrl/envs/envs.py](ecomodrl/envs/envs.py)) - the API standard for reinforcement learning. All models are solved in [ecomodrl/experiments](ecomodrl/experiments) with four algorithms: PPO, DDPG, TD3, and SAC (the implementation of the core algorithms is based on the [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/#) package). Benchmark solutions come from (nested) endogenous grid method (EGM) solvers in [ecomodrl/models/solvers.py](ecomodrl/models/solvers.py).

### Structure-exploiting critics

All four algorithms use a *post-decision* critic ([ecomodrl/algorithms/post_decision](ecomodrl/algorithms/post_decision/)) instead of the generic state-action critic: the critic knows the utility function and the deterministic part of the budget identity and learns only the continuation value over the post-decision state,

```
Q(s, a) = u(c(s, a)) + W(a_pd),        a_pd = m - c   (plus exogenous states)
```

Its target integrates over the model's own shock nodes (Gauss-Hermite for Brock–Mirman and buffer-stock, a subset of the scrambled-Sobol nodes for portfolio choice) instead of using the single sampled next state. DDPG, TD3 and SAC keep their training loops with the critic target replaced; PPO uses the same value head and the one-step post-decision advantage in place of GAE (so there is no `gae_lambda`). Observations are not normalized, because the critic inverts the observation transform.

## Repository structure

```
ecomodrl/
  algorithms/    # training loop, multi-seed runner, evaluation callbacks, hyperparameter adapters, algo wrappers
  analysis/      # metrics, plotting, LaTeX tables
  cluster/       # SLURM scripts: full tune -> train -> analyze pipeline per model
  configs/       # model calibrations + run/tune/analysis configs (CLI)
  envs/          # Gymnasium environments + vectorized env setup
  experiments/   # training entry points (one per model) + analyze_experiments.py
  models/        # economic models, benchmark solvers, theory notebooks
  tune/          # DEHB hyperparameter tuning CLI, target function, entry points, tuned-hyperparameter loading
```

The package is not installed via pip. Run all commands from the repository root using `python -m`, as shown below.

## Installation

**Platform note.** The conda environment definition in [requirements.yml](requirements.yml) targets `win-64` and Python 3.11.

We use conda-lock to install the requirements of this project. This ensures full reproducibility of the code.

First install conda-lock via condax (safer):
```
conda install condax
condax install conda-lock
```

Create a new conda environment "YOURENV" or install the packages into your preexisting conda env "YOURENV" by
```
conda-lock install -n YOURENV requirements.conda-lock.yml
```

To update a single package to the latest version compatible with the version constraints in the source:
```
conda-lock lock  --lockfile requirements.conda-lock.yml --update PACKAGE
```

To re-solve the entire environment (produce a new requirements.conda-lock.yml), e.g. after changing a version constraint or adding a package in the source file:
```
conda-lock -f requirements.yml --lockfile requirements.conda-lock.yml
```

You can also use pip to reproduce our results. Simply run:
```
pip install -r requirements.txt
```

## Quickstart

A fast example: PPO on Cass–Koopmans with 4 seeds. Training runs on a wall-clock budget (300 s per seed for this model); seeds train in parallel across CPU cores:

```
python -m ecomodrl.experiments.casskoopmans --algo_name ppo --n_seeds 4
python -m ecomodrl.experiments.analyze_experiments --env casskoopmans --algo_name ppo --n_seeds 4
```

`--n_seeds` must match between the two calls. Outputs land under `./runs` (override with the `RUNRESULTS_PATH` environment variable, see instructions in `/configs/runconfigs.py`):

- **Checkpoints:** `./runs/CassKoopmans/PPO/seed_<n>/` — best model (`model.zip`), evaluation history (`eval_history.npz`), TensorBoard event files — plus run metadata in `train_cfgs.json`.
- **Metrics:** `runs/CassKoopmans/PPO/analysis_summary.json` — aggregated accuracy and welfare statistics.
- **Figures and tables:** `runs/CassKoopmans/figures/` — `policy.png`, `cev.png`, `euler_errors.png`, `convergence.png`, `table.tex`, `CassKoopmans_comparison_table.tex`.

## Hyperparameter tuning (optional)

Hyperparameters are tuned with DEHB, using wall-clock training time as the multi-fidelity budget (in seconds):

```
python -m ecomodrl.tune.casskoopmans --algo_name ppo --total_cost 3600
```

This writes `./tune_results/<Model>/<ALGO>/incumbent.json` (override the root with `TUNERESULTS_PATH`), which training picks up automatically. Without tuning results, training falls back to default hyperparameters (the quickstart works either way).

## Full replication

The CLI defaults are the paper settings: `--n_seeds 100`, `--hidden_nodes 128`, `--n_envs 10`, with wall-clock budgets of 300 s (Cass–Koopmans), 600 s (Brock–Mirman, buffer-stock), and 1200 s (portfolio choice) per seed. To replicate a model's results, run tune, train, and analyze for each algorithm (`ppo`, `ddpg`, `td3`, `sac`), substituting the model's CLI name in the commands above.

The SLURM scripts in [ecomodrl/cluster/](ecomodrl/cluster/) run the entire pipeline: DEHB tuning (4 h per algorithm), training, and analysis for all four algorithms as a single job per model on a HPC cluster.

## Extending to new models

1. Implement the model as an `EconomicModel` subclass in [ecomodrl/models/](ecomodrl/models/) (see [base_model.py](ecomodrl/models/base_model.py)) and add a Gymnasium environment in [ecomodrl/envs/envs.py](ecomodrl/envs/envs.py).
2. Implement `post_decision_spec(reward_floor)` on the model: plain floats/lists with the action map, utility parameters and the quadrature nodes of the RL transition (see the four existing models). If the model does not fit an existing family in [post_decision/economics.py](ecomodrl/algorithms/post_decision/economics.py), add a torch `PDEconomics` subclass there (`controls` and `next_obs`) and register it in `FAMILIES`.
3. Add a calibration dataclass in [ecomodrl/configs/parameters.py](ecomodrl/configs/parameters.py) and run/tune/analysis configs in [ecomodrl/configs/](ecomodrl/configs/).
4. Add entry points mirroring [experiments/casskoopmans.py](ecomodrl/experiments/casskoopmans.py) and [tune/casskoopmans.py](ecomodrl/tune/casskoopmans.py) (each ~7 lines), and register the model in [analyze_experiments.py](ecomodrl/experiments/analyze_experiments.py).

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
