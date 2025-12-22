#!/bin/bash
#SBATCH -J test_log
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 00:05:00
#SBATCH --output=/gpfs/work5/0/prjs1828/DSI-QG/logs/test_%j.log

# Trap errors to see what's happening
trap 'echo "Error on line $LINENO"; exit 1' ERR

echo "Job started at $(date)"
ENV_PATH="/gpfs/work5/0/prjs1828/DSI-QG"

echo "Modules loaded. Checking Python..."
which python

source "$ENV_PATH/.venv/bin/activate"
echo "Venv activated."

python -c "import torch; print('Torch version:', torch.__version__)"
