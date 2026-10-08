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

### Model-free and post-decision variants

Every algorithm comes in two variants, selected with `--variant` on the tune, train, and analysis CLIs:

- `mf` (default): the standard, model-free SB3 algorithm.
- `pd`: the model-informed *post-decision* version ([ecomodrl/algorithms/post_decision](ecomodrl/algorithms/post_decision/)). All four algorithms integrate over the model's own shock nodes (Gauss-Hermite for Brock–Mirman and buffer-stock, a subset of the scrambled-Sobol nodes for portfolio choice) instead of using the single sampled next state.

  DDPG, TD3 and SAC use a *post-decision* critic instead of the generic state-action critic: the critic knows the utility function and the deterministic part of the budget identity and learns only the continuation value over the post-decision state,

  ```
  Q(s, a) = u(c(s, a)) + W(a_pd),        a_pd = m - c   (plus exogenous states)
  ```

  with the critic target computed by quadrature over the shock nodes; their training loops are otherwise unchanged. PPO keeps a single standard value network V(s) and standard GAE, but the one-step bootstrap V(s') in its TD error is replaced by the quadrature expectation of V at the visited post-decision state,

  ```
  delta_t = r_t + beta * sum_k w_k V(s'_k(s_t, a_t)) - V(s_t)
  ```

  computed once at the end of each rollout with the detached value network.

Observation normalization (`--normalize_obs`) is switched off for `pd`, because the critics invert the observation transform.

## Repository structure

```
ecomodrl/
  algorithms/    # training loop, multi-seed runner, evaluation callbacks, hyperparameter adapters, algo wrappers
                 #   post_decision/: model-informed (pd) variants of PPO/DDPG/TD3/SAC
  analysis/      # metrics, plotting, LaTeX tables
  cluster/       # SLURM scripts: full tune -> train -> analyze pipeline per model (both variants)
  configs/       # model calibrations + run/tune/analysis configs (CLI), variants.py (mf/pd naming)
  envs/          # Gymnasium environments + vectorized env setup
  experiments/   # training entry points (one per model) + analyze_experiments.py
  models/        # economic models, NumPy/Torch array_ops.py, ModelSpec serialization, benchmark solvers
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

- **Checkpoints:** `./runs/CassKoopmans/PPO/seed_<n>/` — best model (`model.zip`), normalization stats (`vecnormalize.pkl`), evaluation history (`eval_history.npz`), TensorBoard event files — plus run metadata in `train_cfgs.json`.
- **Metrics:** `runs/CassKoopmans/PPO/analysis_summary.json` — aggregated accuracy and welfare statistics.
- **Figures and tables:** `runs/CassKoopmans/figures/` — `policy_mf.pdf`, `cev_mf.pdf`, `euler_errors_mf.pdf`, `convergence_mf.pdf`, `table.tex`, `CassKoopmans_comparison_table.tex`.

The post-decision variant is run and analyzed separately with `--variant pd` on both commands; its outputs get a `_PD` suffix (`runs/CassKoopmans/PPO_PD/`, `runs/CassKoopmans/figures_PD/`), while the benchmark-policy cache `runs/CassKoopmans/egm_analysis/` is shared:

```
python -m ecomodrl.experiments.casskoopmans --algo_name ppo --variant pd --n_seeds 4
python -m ecomodrl.experiments.analyze_experiments --env casskoopmans --algo_name ppo --variant pd --n_seeds 4
```

To compare both variants in one figure, run the analysis a third time with `--variant both`. This stage runs no testing and loads no checkpoints; it reads existing per-variant analysis output and the shared `egm_analysis/` cache. It writes `runs/CassKoopmans/figures_combined/euler_errors.pdf`, `convergence.pdf`, and `cev.pdf`.

```
python -m ecomodrl.experiments.analyze_experiments --env casskoopmans --algo_name ppo --variant both --n_seeds 4
```

## Hyperparameter tuning (optional)

Hyperparameters are tuned with DEHB, using wall-clock training time as the multi-fidelity budget (in seconds):

```
python -m ecomodrl.tune.casskoopmans --algo_name ppo --total_cost 3600
```

This writes `./tune_results/<Model>/<ALGO>/incumbent.json` (override the root with `TUNERESULTS_PATH`), which training picks up automatically. Without tuning results, training falls back to default hyperparameters (the quickstart works either way). The post-decision variant is tuned separately (`--variant pd`) and writes `./tune_results/<Model>/<ALGO>_PD/incumbent.json`; its search space only differs for PPO, where `gae_lambda` may go down to 0 (pure one-step quadrature advantage).

## Full replication

The CLI defaults are the paper settings: `--n_seeds 100`, `--hidden_nodes 128`, `--n_envs 10`, with wall-clock budgets of 300 s (Cass–Koopmans), 600 s (Brock–Mirman, Buffer Stock), and 1200 s (Portfolio Choice) per seed. To replicate a model's results, run tune, train, and analyze for each algorithm (`ppo`, `ddpg`, `td3`, `sac`) and each variant (`mf`, `pd`), substituting the model's CLI name in the commands above.

The SLURM scripts in [ecomodrl/cluster/](ecomodrl/cluster/) run the entire pipeline: DEHB tuning (up to 4 h per algorithm and variant), training, and analysis for all four algorithms and both variants, plus the combined mf-vs-pd figures, as a single job per model on a HPC cluster.

## Extending to new models

1. Implement the model as an `EconomicModel` subclass in [ecomodrl/models/](ecomodrl/models/) (see [base_model.py](ecomodrl/models/base_model.py)) and add a Gymnasium environment in [ecomodrl/envs/envs.py](ecomodrl/envs/envs.py). For the post-decision variants: Write `utility` and `controls_from_action` against the op table in [models/array_ops.py](ecomodrl/models/array_ops.py) (`xp = ops(x)`), so they accept NumPy arrays and Torch tensors alike. Override both `obs_from_state` and `state_from_obs` when the default log transform is unsuitable.
2. For the post-decision variant, implement `post_decision_dim`, `post_decision_features(state, controls)` (the inputs of the continuation network W) and `quadrature_shocks()` (nodes and weights in the same shock layout that `sample_shocks()` produces, so the base class can reuse `transition()`). Implement `model_spec_kwargs()` for behavior-affecting constructor options that are not part of the calibration dataclass, and register the class in [models/model_spec.py](ecomodrl/models/model_spec.py).
3. Add a economic parameter dataclass in [ecomodrl/configs/parameters.py](ecomodrl/configs/parameters.py) and run/tune/analysis configs in [ecomodrl/configs/](ecomodrl/configs/).
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
