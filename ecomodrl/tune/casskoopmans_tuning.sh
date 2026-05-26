#!/bin/bash

# Run via:
# > cd ecomodrl/ecomodrl/tune
# > sed -i 's/\r$//' casskoopmans_tuning.sh
# > chmod +x casskoopmans_tuning.sh
# > sbatch casskoopmans_tuning.sh
# logs will be in ecomodrl/ecomodrl/tuning/logs/
# Incumbent in scratch in tuning_results/CassKoopmans/

#SBATCH --job-name=CK_TUNE
#SBATCH --partition=d0ow
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=92G
#SBATCH --time=04:30:00
#SBATCH --export=NONE

#SBATCH --output=logs/CK_TUNE_%j.out
#SBATCH --error=logs/CK_TUNE_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruven.micheel@uni-muenster.de

set -euo pipefail

echo "Job $SLURM_JOBID on $(hostname)"
echo "Starting at: $(date)"

##########################
# Load the PyTorch stack #
##########################

module purge
module load palma/2023a
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load Python/3.11.3
module load PyTorch/2.1.2-CUDA-12.1.1

echo
echo "Loaded modules:"
module list

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
# Run Hyperparam Tuning    #
############################

# 1. Define list of algorithms
ALGOS=("ppo" "ddpg" "sac" "td3")

# 2. Define where Python should write results in scratch
export TUNERESULTS_PATH="$WORKDIR/tune_results"
mkdir -p "$TUNERESULTS_PATH"

# 3. Budget per algorithm
BUDGET_PER_ALGO=3600

for algo in "${ALGOS[@]}"; do
    echo "------------------------------------------------"
    echo "Starting tuning for: $algo"
    echo "------------------------------------------------"
    
    python ecomodrl/tune/casskoopmans.py \
        --algo_name="$algo" \
        --total_cost=$BUDGET_PER_ALGO \
        --hidden_nodes=64

    echo "Finished tuning for $algo"
done

echo
echo "All tuning jobs finished."
echo "Results are located in: $TUNERESULTS_PATH" 
echo "Finished at: $(date)"