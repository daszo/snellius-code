#!/bin/bash
#SBATCH -J enron_dsi_100k_thread
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH -G 2
#SBATCH --cpus-per-task=18
#SBATCH -t 0:30:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=daniel.van.oosteroom@student.uva.nl
#SBATCH --output=/gpfs/work5/0/prjs1828/DSI-QG/logs/enron_dsi_100k_thread.log


exec > >(ts '[%Y-%m-%d %H:%M:%S]') 2>&1

ENV_PATH="/gpfs/work5/0/prjs1828/DSI-QG"
source "$ENV_PATH/.venv/bin/activate"
export PYTHONUNBUFFERED=1

torchrun --nproc_per_node=1 test_100k_thread_dsi.py\
