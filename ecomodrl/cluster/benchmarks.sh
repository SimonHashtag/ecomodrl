#!/bin/bash

# Submit from the repository root with: sbatch ecomodrl/cluster/benchmarks.sh
# The Python 3.11 environment must have the project's dependencies installed.
# Logs are written only to $WORK/ecomodrl-benchmarks-<job ID>/:
# benchmarks_<job ID>.out (prints), benchmarks_<job ID>.err (errors).

#SBATCH --job-name=BENCHMARKS
#SBATCH --partition=d0ow
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=92G
#SBATCH --time=1:00:00
#SBATCH --export=NONE
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simonhaastert@uni-muenster.de

set -euo pipefail

# SBATCH directives cannot expand $WORK; set up log paths in the shell instead.
WORKDIR="$WORK/ecomodrl-benchmarks-$SLURM_JOB_ID"
mkdir -p "$WORKDIR"
exec > "$WORKDIR/benchmarks_$SLURM_JOB_ID.out" \
     2> "$WORKDIR/benchmarks_$SLURM_JOB_ID.err"

echo "Job $SLURM_JOB_ID on $(hostname), starting at $(date)"
echo "Working directory and logs: $WORKDIR"

echo "[$(date)] Loading cluster modules"
module purge
module load palma/2023a
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load Python/3.11.3
module list

# Copy the repository to the job's working directory.
echo "[$(date)] Copying repository to $WORKDIR"
cp -r "$HOME/ecomodrl" "$WORKDIR/"
echo "[$(date)] Repository copy complete"
cd "$WORKDIR/ecomodrl"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "[$(date)] Starting Python benchmark diagnostics"
python -u -m ecomodrl.models.main

echo "Finished at $(date)"
