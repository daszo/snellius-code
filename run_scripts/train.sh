#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/.venv/lib/python3.9/site-packages
PYTHONUNBUFFERED=1 /gpfs/work5/0/prjs1828/DSI-QG/.venv/bin/torchrun --nproc_per_node=1 run.py \
        --task "DSI" \
        --model_name "./local_models/google/mt5-base" \
        --run_name "enron-10k-mt5-base-DSI-Q-classic" \
        --max_length 32 \
        --output_dir "models/enron-10k-mt5-base-DSI-Q-classic" \
        --learning_rate 0.0005 \
        --warmup_steps 100000 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --max_steps 1000000 \
        --save_strategy steps \
        --dataloader_num_workers 10 \
        --save_steps 1000 \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 1 \
        --report_to wandb \
        --logging_steps 100 \
        --dataloader_drop_last False \
        --metric_for_best_model Hits@10 \
        --greater_is_better True \
        --remove_prompt True \
	--db_name "data/enron.db" \
       	--table_name "N10k_text_rank_d2q_q1" 2>&1 | tee training_log.txt
	
