#!/usr/bin/env bash
#SBATCH --account=gisr113267
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH -t 3:00:00
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pre_pipeline
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=daniel.van.oosteroom@student.uva.nl
#SBATCH --output=/gpfs/work5/0/prjs1828/DSI-QG/logs/pre_pipeline_same_mid.log

exec > >(ts '[%Y-%m-%d %H:%M:%S]') 2>&1

echo "Starting pipeline"
echo "1. data cleaning and textrank"
export PYTHONUNBUFFERED=1

PYTHONUNBUFFERED=1 /gpfs/work5/0/prjs1828/DSI-QG/.venv/bin/python3 -m pipelines.run_stage1 --thread --table_from "N10k" --table_to "N10k_thread_same_mid" --destination_table "N10k_thread_same_mid_tr" --text_rank_only

RETURN_SEQ=1

echo "2. Title generation"

PYTHONUNBUFFERED=1 /gpfs/work5/0/prjs1828/DSI-QG/.venv/bin/torchrun --nproc_per_node=1 run.py \
    --do_predict \
    --task generation \
    --table_name N10k_thread_same_mid_tr \
    --model_name /gpfs/work5/0/prjs1828/DSI-QG/local_models/t5-headline \
    --per_device_eval_batch_size 32 \
    --run_name docTquery-full-generation \
    --max_length 256 \
    --valid_file /data/enron.db \
    --output_dir temp \
    --dataloader_num_workers 10 \
    --report_to wandb \
    --logging_steps 100 \
    --num_return_sequences $RETURN_SEQ \
    --thread 1 \
    --same_mid N10k_thread_same_mid_full

echo "3. splitsing the data"

# PYTHONUNBUFFERED=1 /gpfs/work5/0/prjs1828/DSI-QG/.venv/bin/python3 -m pipelines.prep_datasets --thread "N10k_thread_same_mid"
