#!/bin/env/usr bash

#TODO add batch parameters.
.venv/bin/python3 -m pipelines.run_stage1

RETURN_SEQ=1

PYTHONUNBUFFERED=1 /gpfs/work5/0/prjs1828/DSI-QG/.venv/bin/torchrun --nproc_per_node=1 run.py \
    --do_predict \
    --task generation \
    --table_name full_text_rank \
    --model_name /gpfs/work5/0/prjs1828/DSI-QG/local_models/t5-headline \
    --per_device_eval_batch_size 32 \
    --run_name docTquery-full-generation \
    --max_length 256 \
    --valid_file /data/enron.db \
    --output_dir temp \
    --dataloader_num_workers 10 \
    --report_to wandb \
    --logging_steps 100 \
    --num_return_sequences $RETURN_SEQ

.venv/bin/python3 -m pipelines.prep_datasets "full_text_rank_d2q_q$RETURN_SEQ"
