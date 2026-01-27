#!/bin/bash

# Run via:
# > cd ecomodrl/ecomodrl/experiments/casskoopmans
# > sed -i 's/\r$//' casskoopmans_training.sh
# > chmod +x casskoopmans_training.sh
# > sbatch casskoopmans_training.sh
# logs will be in ecomodrl/ecomodrl/experiments/logs/
# Plots in scratch in training_results/CassKoopmans/

#SBATCH --job-name=CK_TRAINING
#SBATCH --partition=d0ow
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=92G
#SBATCH --time=02:00:00
#SBATCH --export=NONE

#SBATCH --output=logs/CK_TRAINING_%j.out
#SBATCH --error=logs/CK_TRAINING_%j.err
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
#   Configuration Setup    #
############################

# Define one common place in scratch for the plots/results.
export RUNRESULTS_PATH="$WORKDIR/training_results/CassKoopmans"
mkdir -p "$RUNRESULTS_PATH"

echo "Results will be stored in: $RUNRESULTS_PATH"

############################
#   Run Training Scripts   #
############################

# 1. PPO
echo "------------------------------------------------"
echo "Starting PPO..."
python ecomodrl/experiments/casskoopmans/casskoopmans_ppo.py

# 2. DDPG
echo "------------------------------------------------"
echo "Starting DDPG..."
python ecomodrl/experiments/casskoopmans/casskoopmans_ddpg.py

# 3. SAC
echo "------------------------------------------------"
echo "Starting SAC..."
python ecomodrl/experiments/casskoopmans/casskoopmans_sac.py

# 4. TD3
echo "------------------------------------------------"
echo "Starting TD3..."
python ecomodrl/experiments/casskoopmans/casskoopmans_td3.py

############################
#   Unified Analysis       #
############################

echo "------------------------------------------------"
echo "Generating unified comparison plots..."
python ecomodrl/experiments/casskoopmans/casskoopmans_compare.py

################
# Finish       #
################

echo "------------------------------------------------"
echo "All jobs finished."
echo "Final results are located in scratch at: $RUNRESULTS_PATH"
echo "Finished at: $(date)"