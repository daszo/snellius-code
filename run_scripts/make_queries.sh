#!/usr/bin/env bash

python3 -m torch.distributed.launch --nproc_per_node=1 run.py \
    --do_predict \
    --task generation \
    --model_name /gpfs/work5/0/prjs1828/DSI-QG/local_models/t5-headline \
    --per_device_eval_batch_size 32 \
    --run_name docTquery-N10k-generation \
    --max_length 256 \
    --valid_file /projects/prjs1828/data/N10k_text_rank_and_subject_msmarco.tsv \
    --output_dir temp \
    --dataloader_num_workers 10 \
    --report_to wandb \
    --logging_steps 100 \
    --num_return_sequences 1
