from ecomodrl.configs.tuneconfigs import CassKoopmansTuneConfig
from ecomodrl.tune.tuning import run_tuning

# Run file with e.g.: python ecomodrl/tune/casskoopmans.py --algo_name="ddpg" --total_cost=3600 --hidden_nodes=64

if __name__ == "__main__":
    run_tuning(CassKoopmansTuneConfig)