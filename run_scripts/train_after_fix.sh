#!/bin/bash
#SBATCH -J enron_dsi
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH -G 2
#SBATCH --cpus-per-task=18
#SBATCH -t 9:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=daniel.van.oosteroom@student.uva.nl
#SBATCH --output=/gpfs/work5/0/prjs1828/DSI-QG/logs/%j_training.log

#export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=0
#export NCCL_NET_GDR_LEVEL=2
#export PYTHONFAULTHANDLER=1 # This prevents "silent" exits by printing tracebacks on crashes

exec > >(ts '[%Y-%m-%d %H:%M:%S]') 2>&1
# 1. Environment Setup
ENV_PATH="/gpfs/work5/0/prjs1828/DSI-QG"
# Load the appropriate toolchain and python module for Snellius
#module purge
#module load 2022  # Or the specific version your venv was built with
#Python/3.9.5-GCCcore-10.3.0

# 2. I/O Optimization: Use Local Scratch ($TMPDIR)
# Home and Project directories are network-mounted (slow).
# SQLite/DB reads should happen on local SSD scratch.
echo "Copying database to local scratch: $TMPDIR"
cp "$ENV_PATH/data/enron.db" "$TMPDIR/enron.db"

# 3. Execution
# Activation is cleaner than manual PYTHONPATH manipulation
source "$ENV_PATH/.venv/bin/activate"
export PYTHONUNBUFFERED=1

torchrun --nproc_per_node=2 run.py \
	--task "DSI" \
	--model_name "$ENV_PATH/local_models/google/mt5-base" \
	--run_name "enron-10k-mt5-base-DSI-Q-classicv1.2" \
        --max_length 32 \
        --output_dir "$ENV_PATH/models/enron-10k-mt5-base-DSI-Q-classic2" \
        --learning_rate 0.0005 \
        --warmup_steps 100000 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --evaluation_strategy "steps" \
	--eval_steps 1000 \
        --max_steps 100000 \
        --save_strategy "steps" \
        --dataloader_num_workers 12 \
        --save_steps 1000 \
        --save_total_limit 5 \
        --load_best_model_at_end True \
        --gradient_accumulation_steps 1 \
        --report_to "wandb" \
        --logging_steps 100 \
        --dataloader_drop_last False \
        --metric_for_best_model "Hits@10" \
        --greater_is_better True \
        --remove_prompt True \
        --db_name "$TMPDIR/enron.db" \
	--table_name "N10k" \
	--save_size "10K" \
	--save_experiment_type "base" \
	--save_version "v1.1" \

###BATCH --qos=short
