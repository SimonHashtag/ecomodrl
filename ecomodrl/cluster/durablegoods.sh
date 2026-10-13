#!/bin/bash

# Run via:
# > cd ecomodrl/ecomodrl/cluster
# > sed -i 's/\r$//' durablegoods.sh
# > chmod +x durablegoods.sh
# > mkdir -p logs
# > sbatch durablegoods.sh
# logs will be in logs/ below the directory sbatch is called from
#
# Durable goods (Fella 2014): discrete durable choice + continuous savings.
# Benchmark: Fella's VFI1 (numba, parallel over the CPUs of this job).
# Budget estimate: tuning 8 x 2 h = 16 h, training 8 x 3 batches x ~22 min = ~9 h,
# analysis < 1 h  ->  ~26 h of the 48 h limit. Worst case, if every DEHB run
# overshoots by one full-fidelity trial (3 seeds x 1200 s), ~34 h.

#SBATCH --job-name=DG_ALL
#SBATCH --partition=d0ow
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=92G
#SBATCH --time=48:00:00
#SBATCH --export=NONE

#SBATCH --output=logs/DG_ALL_%j.out
#SBATCH --error=logs/DG_ALL_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simonhaastert@uni-muenster.de

set -euo pipefail

echo "================================================"
echo "Job $SLURM_JOBID on $(hostname)"
echo "Starting Pipeline at: $(date)"
echo "================================================"

##########################
# Load the module stack #
##########################

module purge
module load palma/2023a
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load Python/3.11.3

echo
echo "Loaded modules:"
module list
echo

#############################
# Scratch working directory #
#############################

WORKDIR="$WORK/batchednets-$SLURM_JOBID"
mkdir -p "$WORKDIR"
echo "Working directory: $WORKDIR"

# Copy repo to scratch
cp -r "$HOME/ecomodrl" "$WORKDIR/"
cd "$WORKDIR/ecomodrl"
export PYTHONPATH=$(pwd):$PYTHONPATH

############################
#   Configuration Setup    #
############################

# 1. Define list of algorithms
ALGOS=("ppo" "ddpg" "sac" "td3")

# 1b. Define list of algorithm variants: mf (model-free), pd (post-decision)
VARIANTS=("mf" "pd")

# 2. Define and Export Paths
# (Exporting makes them available to Python via os.environ)
export TUNERESULTS_PATH="$WORKDIR/tune_results"
mkdir -p "$TUNERESULTS_PATH"

export RUNRESULTS_PATH="$WORKDIR/training_results"
mkdir -p "$RUNRESULTS_PATH"

echo "Tuning results will be stored in: $TUNERESULTS_PATH"
echo "Training results will be stored in: $RUNRESULTS_PATH"
echo

############################
# Phase 1: Hyperparam Tune #
############################

echo "================================================"
echo " PHASE 1: TUNING "
echo "================================================"

BUDGET_PER_ALGO=7200 # 2 hours

for variant in "${VARIANTS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "--- Tuning: $algo ($variant) ---"

        python ecomodrl/tune/durablegoods.py \
            --algo_name="$algo" \
            --variant="$variant" \
            --total_cost=$BUDGET_PER_ALGO \
            --hidden_nodes=128

        echo "Finished tuning for $algo ($variant)"
    done
done

echo
echo "All tuning jobs finished."
echo "Tune results are located in: $TUNERESULTS_PATH"
echo "Finished at: $(date)"

############################
# Phase 2: Run Training    #
############################

echo "================================================"
echo " PHASE 2: TRAINING "
echo "================================================"
# Note: Because TUNERESULTS_PATH is exported, your python training script
# can fetch the incumbent hyperparameters directly from that folder!
# Training wall-clock budget per seed: 1200 s (DurableGoodsRunConfig).

for variant in "${VARIANTS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "--- Training: $algo ($variant) ---"

        python ecomodrl/experiments/durablegoods.py \
            --algo_name="$algo" \
            --variant="$variant" \
            --hidden_nodes=128

        echo "Finished training for $algo ($variant)"
    done
done

echo
echo "All training complete."
echo "Finished at: $(date)"
echo

############################
# Phase 3: Unified Analysis#
############################

echo "================================================"
echo " PHASE 3: ANALYSIS "
echo "================================================"

echo "Generating unified comparison plots and tables..."
for variant in "${VARIANTS[@]}"; do
    python ecomodrl/experiments/analyze_experiments.py --env durablegoods --variant "$variant"
done

# Combined figures overlaying both variants (reads the per-variant analysis output above).
python ecomodrl/experiments/analyze_experiments.py --env durablegoods --variant both

############################
#       Finalize           #
############################

echo
echo "================================================"
echo " ALL PHASES COMPLETE "
echo "================================================"
echo "Tuning  Results: $TUNERESULTS_PATH"
echo "Training Results/Plots: $RUNRESULTS_PATH"
echo "Finished at: $(date)"
