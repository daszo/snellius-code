#!/usr/bin/env bash

 
PYTHONUNBUFFERED=1 /gpfs/work5/0/prjs1828/DSI-QG/.venv/bin/torchrun --nproc_per_node=1 run.py \
    --do_predict \
    --task generation \
    --table_name N10k_text_rank \
    --model_name /gpfs/work5/0/prjs1828/DSI-QG/local_models/t5-headline \
    --per_device_eval_batch_size 32 \
    --run_name docTquery-N10k-generation \
    --max_length 256 \
    --valid_file /data/enron.db \
    --output_dir temp \
    --dataloader_num_workers 10 \
    --report_to wandb \
    --logging_steps 100 \
    --num_return_sequences 1 2>&1 | tee make_queries_db.log
