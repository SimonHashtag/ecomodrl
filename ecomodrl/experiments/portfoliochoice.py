from ecomodrl.configs.runconfigs import PortfolioChoiceRunConfig
from ecomodrl.experiments.run_experiments import run_experiment

# E.g., run with: 
# `python ecomodrl/experiments/portfoliochoice.py --algo_name ppo --hidden_nodes 64 --n_seeds 20`

if __name__ == "__main__":
    run_experiment(PortfolioChoiceRunConfig)