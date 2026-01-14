#!/usr/bin/env bash
#SBATCH --account=gisr113267
#SBATCH --partition=genoa       # Explicitly target CPU nodes (commonly 'genoa', 'rome', or 'batch')
#SBATCH -t 30:00:00
#SBATCH --nodes=1               # Force all tasks/cpus onto a single node
#SBATCH --ntasks=1              # Run 1 main task (the python script)
#SBATCH --cpus-per-task=16      # Give that task 16 cores for multiprocessing
#SBATCH --mem=64G               # Request explicit RAM (BM25 can be memory hungry)
#SBATCH --job-name=test_bm25
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=daniel.van.oosteroom@student.uva.nl
#SBATCH --output=/gpfs/work5/0/prjs1828/DSI-QG/logs/%j_training.log

exec > >(ts '[%Y-%m-%d %H:%M:%S]') 2>&1

# Verify we are on a CPU node
echo "Running on node: $(hostname)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"

# Load modules (adjust based on your environment)
module purge
module load 2023
module load Java/11.0.20

# 3. CRITICAL: Set JAVA_HOME using the cluster's internal variable
export JAVA_HOME=$EBROOTJAVA

echo "Starting testing"

export PYTHONUNBUFFERED=1 

/gpfs/work5/0/prjs1828/DSI-QG/.venv/bin/python3 -m CE.utils.test.evaluation_BM25 "N10k"
