# EcoModRL
This repository accompanies the working paper “Deep Reinforcement Learning in Dynamic Economic Models: An Empirical Study”. The project evaluates deep reinforcement learning (RL) as a numerical solution method for dynamic economic models, with systematic algorithm comparisons and benchmark solutions.

## Table of contents
- [What this repository contains](#what-this-repository-contains)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Run the baseline replication](#run-the-baseline-replication)
- [Create your own experiments](#create-your-own-experiments)
- [Cite this work](#cite-this-work)

## What this repository contains
- **Model environments (Gymnasium):** economic MDPs in [ecomodrl/envs](ecomodrl/envs) for Brock–Mirman growth, Cass–Koopmans growth, buffer‑stock saving, and portfolio choice.
- **Benchmark solvers + theory notes:** EGM-based reference solutions and notebooks in [ecomodrl/benchmarks](ecomodrl/benchmarks).
- **Experiment entry points:** training scripts in [ecomodrl/experiments](ecomodrl/experiments).
- **Model parameter calibrations:** TOML files in [ecomodrl/experiments/model_params](ecomodrl/experiments/model_params).
- **Hyperparameter tuning:** DEHB sweeps in [ecomodrl/tune](ecomodrl/tune).
- **Analysis outputs:** aggregated plots and metrics (learning curves, policy/value functions, Euler errors, CEV) saved under `runs/` by [ecomodrl/utils](ecomodrl/utils).

## Repository structure
```
ecomodrl/
  algorithm/               # Model- and algorithm-agnostic training file
  envs/                    # Economic environments/models
  benchmarks/              # Benchmark solvers (EGM) + theory notebooks
  experiments/
    model_params/          # TOML calibrations for each model
    brockmirman/           # Training + analysis entry point 
    bufferstock/           # Training + analysis entry point 
    casskoopmans/          # Training + analysis entry point 
    portfoliochoice/       # Training + analysis entry point 
  tune/                    # DEHB tuning entry points for multiple algorithms
  utils/                   # Analysis, plotting, metrics
```

## Getting started

**Platform note.** The conda environment definition in [requirements.yml](requirements.yml) targets `win-64` and Python 3.11.

### 1. Project setup

#### 1.1 conda-lock
We use conda-lock to install the requirements of this project. 
This ensures full reproducibility of the code.

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

#### 1.2 pip-compile
You can also use pip to reproduce our results.
Simply run:
```
pip install -r requirements.txt
```

If you want to make your work reproducible as well, follow these steps:
1. First install pip-tools
2. If you would like to add a new module to the project add it to requirements.in.
3. Then you can update requirements.txt with the given requirements.in file.

The code:
```
pip install pip-tools
pip-compile requirements.in
pip install --upgrade pip
pip install -r requirements.txt
```

To update all modules in your requirements.txt file:
```
pip-compile --upgrade
```

#### 1.2 pre-commit
We use pre-commit for code style and preventing large files in our project.
Install pre-commit in git hooks to check if our code and files
are in line with our definitions in pre-commit-config.yaml.
```
pre-commit install
```
Now at each commit pre-commit checks our files

#### 1.3 TensorBoard
To enable hosting, live tracking and sharing of your experiments you can use TensorBoard. <br>
###### 1.3.1 TensorBoard Dev (Image display not supported!)
To upload your experiment on *Tensorboard Dev* enter your terminal and type:

```
tensorboard dev upload --logdir=\relative\path\to\logs (default: 'runs\..') \
      --name "(optional) Cool experiment" \
      --description "(optional) The agent learns her optimal intertemporal consumption."
```

After accepting to the condition that your data will be visible to everyone (who has the link), you can
enter the supplied link to the search bar of your browser and view your experiment online.

###### 1.3.2 Local TensorBoard

In order to enable display of images and figures you can use Tensorboard locally. To do so type into the terminal:
```
tensorboard --logdir=\relative\path\to\logs (default: 'runs\..') --window_title 'Name of tab in browser'
```
After hitting the supplied link you can see the training progress in your browser. <br>
Sharing of the results is only possible with TensorBoard Dev.

TensorBoard automatically downsamples logged summaries to prevent out-of-memory errors. If you wish to obtain
control over downsampling, add the following flag to the terminal command with the category of the uploaded summary
and the number of frames you wish to keep: <br>
e.g. ```--samples_per_plugin "scalars=10000, images=100"```

## Run the baseline replication

### Tuning
DEHB-based hyperparameter optimization is implemented in [ecomodrl/tune](ecomodrl/tune) and shared logic in [ecomodrl/tune/tuning.py](ecomodrl/tune/tuning.py). Outputs are written to `tune_results/<Model>/<ALGO>/` by default.

Entry points:
- [ecomodrl/tune/brockmirman.py](ecomodrl/tune/brockmirman.py)
- [ecomodrl/tune/bufferstock.py](ecomodrl/tune/bufferstock.py)
- [ecomodrl/tune/casskoopmans.py](ecomodrl/tune/casskoopmans.py)
- [ecomodrl/tune/portfoliochoice.py](ecomodrl/tune/portfoliochoice.py)

Example:
```
python ecomodrl/tune/casskoopmans.py --algo_name="ddpg" --total_cost=3600 --hidden_nodes=64
```

### Training
Training scripts in [ecomodrl/experiments](ecomodrl/experiments) solve the economic models using deep RL. They log to TensorBoard, save best models by seed, and store metadata in `train_cfgs.json`.
Entry points:
- [ecomodrl/experiments/brockmirman/brockmirman_ppo.py](ecomodrl/experiments/brockmirman/brockmirman_ppo.py)
- [ecomodrl/experiments/bufferstock/bufferstock_ppo.py](ecomodrl/experiments/bufferstock/bufferstock_ppo.py)
- [ecomodrl/experiments/casskoopmans/casskoopmans_ppo.py](ecomodrl/experiments/casskoopmans/casskoopmans_ppo.py)
- [ecomodrl/experiments/casskoopmans/casskoopmans_ddpg.py](ecomodrl/experiments/casskoopmans/casskoopmans_ddpg.py)
- [ecomodrl/experiments/casskoopmans/casskoopmans_sac.py](ecomodrl/experiments/casskoopmans/casskoopmans_sac.py)
- [ecomodrl/experiments/casskoopmans/casskoopmans_td3.py](ecomodrl/experiments/casskoopmans/casskoopmans_td3.py)
- [ecomodrl/experiments/portfoliochoice/portfoliochoice_ppo.py](ecomodrl/experiments/portfoliochoice/portfoliochoice_ppo.py)

### Analysis
Post‑training evaluation in [ecomodrl/utils](ecomodrl/utils) generates plots and summary statistics:
- `learning_curve_agg.png`
- `policy_value_agg.png`
- `euler_errors_agg.png`
- `metrics_analysis.json` (welfare/accuracy summaries)
- model‑specific welfare plots (e.g., `cev_agg.png`)
 
Modules:
- [ecomodrl/utils/analysis_BM.py](ecomodrl/utils/analysis_BM.py)
- [ecomodrl/utils/analysis_BS.py](ecomodrl/utils/analysis_BS.py)
- [ecomodrl/utils/analysis_CK.py](ecomodrl/utils/analysis_CK.py)

Cross‑algorithm comparison (Cass–Koopmans):
- [ecomodrl/experiments/casskoopmans/casskoopmans_compare.py](ecomodrl/experiments/casskoopmans/casskoopmans_compare.py)

## Create your own experiments

1. Create a gym-style environment in [ecomodrl/envs](ecomodrl/envs) featuring the economic model.
2. Add a calibration in [ecomodrl/experiments/model_params](ecomodrl/experiments/model_params) as a `TOML`-file and add a dataclass in [ecomodrl/utils/params_loader.py](ecomodrl/utils/params_loader.py).
3. Copy a training script in [ecomodrl/experiments](ecomodrl/experiments), update:
   - `environment`
   - `toml_path`
   - `params_loader`
   - `out_dir` or `RUNRESULTS_PATH`
   - `n_seeds`, `total_timesteps`, and evaluation settings
   - algorithm hyperparameters in `AlgoConfig` (should be tuned via custom file in [ecomodrl/tune](ecomodrl/tune))
3. Run the script and inspect `runs/<Model>/<ALGO>/` outputs.

## Cite this work

```bibtex
@techreport{Haastert.2026,
  title       = {Deep Reinforcement Learning in Dynamic Economic Models: An Empirical Study},
  author      = {Haastert, Simon and Micheel, Ruven},
  year        = {2026},
  institution = {Working paper},
}
```

## Contributing and maintenance
...
