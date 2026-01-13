#!/usr/bin/env bash
#SBATCH --account=gisr113267
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --job-name=test_cleaning
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=daniel.van.oosteroom@student.uva.nl
#SBATCH --output=/gpfs/work5/0/prjs1828/DSI-QG/logs/%j_training.log

exec > >(ts '[%Y-%m-%d %H:%M:%S]') 2>&1

export PYTHONUNBUFFERED=1 
#/gpfs/work5/0/prjs1828/DSI-QG/.venv/bin/python3 -m pipelines.run_stage1
/gpfs/work5/0/prjs1828/DSI-QG/.venv/bin/python3 -m pipelines.prep_datasets "full_text_rank_d2q_q1"

