#!/usr/bin/env bash
#SBATCH --account=gisr113267
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH -t 60:00:00
#SBATCH --cpus-per-task=16
#SBATCH --job-name=test_writing_files
#SBATCH --output=/gpfs/work5/0/prjs1828/DSI-QG/logs/%j_training.log

/gpfs/work5/0/prjs1828/DSI-QG/.venv/bin/python3 -m pipelines.prep_datasets "v_CleanMessages"
