from data import (
    IndexingTrainDataset,
    GenerateDataset,
    IndexingCollator,
    QueryEvalCollator,
)
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    set_seed,
    AutoTokenizer
)
from trainer import DSITrainer, DocTqueryTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import json
from tqdm import tqdm
import pandas as pd
import os

from CE.utils.database import load_db, write_to_db

from CE.tries import generate_trie_dict

from sklearn.model_selection import train_test_split

import time
import datetime
from textwrap import dedent

set_seed(313)


@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=32)
    id_max_length: Optional[int] = field(default=20)
    remove_prompt: Optional[bool] = field(default=False)
    train_file: Optional[str] = field(default=None)
    valid_file: Optional[str] = field(default=None)
    task: Optional[str] = field(default=None, metadata={"help": "DSI, docTquery, generation"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)
    table_name: Optional[str] = field(default=None)
    db_name: Optional[str] = field(default=None)

    # --- settings for train validate split. ---
    train_size: Optional[float] = 0.8
    validate_size: Optional[float] = 0.1
    test_size: Optional[float] = 0.1
    query_type: Optional[str] = "mixed"
    split_by: Optional[str] = "query" # "email"

    # --- settings for saving ---
    save_size: str =field(default="N10k"),
    save_experiment_type: str =field(default="base"),
    save_version: str =field(default="v1.0")


def make_compute_metrics(tokenizer, valid_ids):

    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)

            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            if len(hits) != 0:
                hit_at_10 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
        return {
            "Hits@1": hit_at_1 / len(eval_preds.predictions),
            "Hits@10": hit_at_10 / len(eval_preds.predictions),
        }

    return compute_metrics

@dataclass
class SplitArgs:
    train_size: float = 0.8
    validate_size: float = 0.1
    test_size: float = 0.1
    query_type: str = "mixed"
    split_by: str = "query" # "email"

def verify_distributed_setup():
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = os.environ.get("LOCAL_RANK", "N/A")
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
                                                    
        print(f"--- [Rank {rank}/{world_size}] Local Rank: {local_rank} | Device: {gpu_name} ---")
    else:
        print("Distributed environment not initialized.")

def split_train_validate_test(run_args, local_rank: int) -> Tuple[Dict, bool, pd.DataFrame]:
    table_name = run_args.table_name
    df = load_db(run_args.table_name, run_args.db_name)

    semantic_ids = df['elaborative_description'].map(type).eq(str).all()
    if semantic_ids is True:    
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        def count_t5_tokens(text):
            return len(tokenizer.encode(text))
        df['token_count'] = df['text'].apply(count_t5_tokens)
        run_args.id_max_length = df['token_count'].max()

    train_size = run_args.train_size
    validate_size = run_args.validate_size
    test_size = run_args.test_size

    total_proportion = train_size + test_size + validate_size
    if total_proportion > 1 or total_proportion < 0.98:
        raise ValueError(f"Total {total_proportion} is invalid. Check sizes.")

    # 1. Isolate the static bodies (kept separate as per your original code)
    bodies = df[["body_clean_and_subject", "elaborative_description"]].copy()
    
    # 2. Prepare the main data for splitting
    # We create a unique ID to track the original rows
    df['orig_idx'] = df.index 
    
    # "Melt" the dataframe: Turn 1 row with 2 cols into 2 rows with 1 col
    # This creates a dataframe with cols: [orig_idx, elaborative_description, text]
    melted = df.melt(
        id_vars=['orig_idx', 'elaborative_description'], 
        value_vars=['text_rank_query', 'doctoquery'],
        value_name='text'
    )

    # Clean: Drop entries where text is NaN or empty string
    melted = melted[melted['text'].notna() & (melted['text'] != "")]

    # Shuffle everything once to ensure randomness
    melted = melted.sample(frac=1, random_state=42).reset_index(drop=True)

    # Group by the original row ID and take the first available item for the 'guaranteed' set
    guaranteed_train = melted.groupby('orig_idx').head(1)
    
    # The remainder is everything else (the second column of the rows, if it existed)
    remainder = melted.drop(guaranteed_train.index)

    # Calculate exact counts for Test and Validation based on the TOTAL dataset size
    total_samples = len(melted)
    n_val = int(total_samples * validate_size)
    n_test = int(total_samples * test_size)

    # Ensure we have enough remainder to cover Test + Validate
    # (You said Train > Test+Val, so mathematically this should always pass)
    if len(remainder) < (n_val + n_test):
        raise ValueError("Not enough 'spare' columns to fill Test/Val sets without violating the 'one-per-row-in-train' rule.")

    # Slice the remainder to fill Test and Validate
    df_validate = remainder.iloc[:n_val]
    df_test = remainder.iloc[n_val : n_val + n_test]
    
    # All leftover remainder goes to Train
    extra_train = remainder.iloc[n_val + n_test:]
    
    # Combine the guaranteed set with the extra set
    df_train = pd.concat([guaranteed_train, extra_train]).sample(frac=1, random_state=42)

    # ---------------------------------------------------------
    # Writing to files
    # ---------------------------------------------------------

    file_names = {}
    
    # Helper to clean up the write loop (since data is now already "exploded")
    datasets = [("train", df_train), ("validate", df_validate), ("test", df_test)]

    for name, target_df in datasets:
        output_filename = f"data/{name}.{table_name}.docTquery"
        file_names[name] = output_filename

        if local_rank == 0:
            print(f"Writing DataFrame to {output_filename}...")
            with open(output_filename, "w") as f:
                
                # Standard Data Writing
                # Note: target_df now has 'text' and 'elaborative_description' directly.
                # No need to loop over column names anymore.
                for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc=f"Writing {name}"):
                    item_dict = {}
                    item_dict['text'] = str(row['text'])
                    item_dict['text_id'] = str(row['elaborative_description'])
                    
                    f.write(json.dumps(item_dict) + "\n")

                # "Bodies" Writing (Only for Train)
                if name == "train":
                    for _, row in tqdm(bodies.iterrows(), total=len(bodies), desc="Writing bodies"):
                        # Safety check for empty bodies
                        if pd.isna(row["body_clean_and_subject"]):
                            continue

                        item_dict = {}
                        item_dict['text'] = str(row["body_clean_and_subject"])
                        item_dict['text_id'] = str(row['elaborative_description'])
                        
                        f.write(json.dumps(item_dict) + "\n")

    if dist.is_initialized():
        dist.barrier()

    print("File writing complete.")
    return file_names, semantic_ids, df



def main():

    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

    verify_distributed_setup()

    # We use wandb logger: https://wandb.ai/site.
    local_rank = training_args.local_rank
    if local_rank == 0:  # only on main process
        # Initialize wandb run
        wandb.login()
        wandb.init(project="DSI", name=training_args.run_name)

    if "mt5" in run_args.model_name:
        tokenizer = MT5Tokenizer.from_pretrained(run_args.model_name, cache_dir="cache")
        fast_tokenizer = MT5TokenizerFast.from_pretrained(
            run_args.model_name, cache_dir="cache"
        )
        if run_args.model_path:
            model = MT5ForConditionalGeneration.from_pretrained(
                run_args.model_path, cache_dir="cache"
            )
        else:
            model = MT5ForConditionalGeneration.from_pretrained(
                run_args.model_name, cache_dir="cache"
            )
    else:
        tokenizer = T5Tokenizer.from_pretrained(run_args.model_name, cache_dir="cache")
        fast_tokenizer = T5TokenizerFast.from_pretrained(
            run_args.model_name, cache_dir="cache"
        )
        if run_args.model_path:
            model = T5ForConditionalGeneration.from_pretrained(
                run_args.model_path, cache_dir="cache"
            )
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                run_args.model_name, cache_dir="cache"
            )
    table_name = run_args.table_name
    db_name = run_args.db_name

    if run_args.task == "docTquery":
        train_dataset = IndexingTrainDataset(
            path_to_data=run_args.train_file,
            max_length=run_args.max_length,
            cache_dir="cache",
            tokenizer=tokenizer,
        )

        valid_dataset = IndexingTrainDataset(
            path_to_data=run_args.valid_file,
            max_length=run_args.max_length,
            cache_dir="cache",
            remove_prompt=run_args.remove_prompt,
            tokenizer=tokenizer,
        )
        trainer = DocTqueryTrainer(
            do_generation=False,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding="longest",
            ),
        )
        trainer.train()

    elif run_args.task == "DSI":

        run_semantic = True

        if table_name is not None and db_name is not None:

            file_names, run_semantic, df = split_train_validate_test(run_args, local_rank)

            run_args.train_file = file_names["train"]
            run_args.valid_file = file_names["validate"]
            run_args.test_file = file_names["test"]


        if table_name is not None and db_name is None:
            raise ValueError(f"forgot the specify a -db_name with table_name {table_name}")

        if table_name is None and db_name is not None:
            raise ValueError(f"forgot the specify a -table_name with db_name {db_name}")

        train_dataset = IndexingTrainDataset(
            path_to_data=run_args.train_file,
            max_length=run_args.max_length,
            cache_dir="cache",
            tokenizer=tokenizer,
        )

        valid_dataset = IndexingTrainDataset(
            path_to_data=run_args.valid_file,
            max_length=run_args.max_length,
            cache_dir="cache",
            remove_prompt=run_args.remove_prompt,
            tokenizer=tokenizer,
            )


        # TODO: check if docids are semantic:
        # TODO: implement trie
        # TODO: implement beam first search
        ################################################################
        # docid generation constrain, we only generate integer docids.
        if not run_semantic:
            SPIECE_UNDERLINE = "▁"
            INT_TOKEN_IDS = []
            for token, id in tokenizer.get_vocab().items():
                if token[0] == SPIECE_UNDERLINE:
                    if token[1:].isdigit():
                        INT_TOKEN_IDS.append(id)
                if token == SPIECE_UNDERLINE:
                    INT_TOKEN_IDS.append(id)
                elif token.isdigit():
                    INT_TOKEN_IDS.append(id)
            INT_TOKEN_IDS.append(tokenizer.eos_token_id)

            def restrict_decode_vocab(batch_idx, prefix_beam):
                return INT_TOKEN_IDS
        ################################################################
        else:
            decoder_trie = generate_trie_dict(df, tokenizer)

            def restrict_decode_vocab(batch_idx, prefix_beam):
                return decoder_trie.get(prefix_beam.tolist())

        model.resize_token_embeddings(len(tokenizer))

        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            run_semantic=run_semantic,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding="longest",
            ),
            compute_metrics=make_compute_metrics(
                fast_tokenizer, train_dataset.valid_ids
            ),
            restrict_decode_vocab=restrict_decode_vocab ,
            id_max_length=run_args.id_max_length,
        )

        start_time = time.time()
        train_result = trainer.train()
        end_time = time.time()

        duration_seconds = end_time - start_time

        td = datetime.timedelta(seconds=duration_seconds)

        total_batches = train_result.global_step

        metrics = train_result.metrics
        final_epoch = metrics.get("epoch")

        write= dedent(f"""
        total_time = {td}
        batches = {total_batches}
        final_epoch = {final_epoch}
        """)

        with open(f"logs/{training_args.run_name}_{datetime.datetime.now()}.txt", 'w') as f:
            f.write(write)
        
        from CE.utils.test.gr_evaluation import DSIEmailSearchEvaluator


        evaluator = DSIEmailSearchEvaluator(
            trainer=trainer, run_args=run_args, input_file=run_args.test_file, df=df
        )

        print("Preparing Data...")
        evaluator.prepare_data()

        print("Running Retrieval Phase (Inference)...")
        evaluator.run_retrieval_phase()

        print("Computing Metrics...")
        evaluator.compute_metrics()

        print("Saving Results...")
        evaluator.save_results(size=run_args.save_size, experiment_type=run_args.save_experiment_type, version=run_args.save_version)


    elif run_args.task == "generation":
        generate_dataset = GenerateDataset(
            path_to_data=run_args.valid_file,
            max_length=run_args.max_length,
            cache_dir="cache",
            tokenizer=tokenizer,
            table_name=table_name,
        )

        trainer = DocTqueryTrainer(
            do_generation=True,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=QueryEvalCollator(
                tokenizer,
                padding="longest",
            ),
        )
        predict_results = trainer.predict(
            generate_dataset,
            top_k=run_args.top_k,
            num_return_sequences=run_args.num_return_sequences,
            max_length=run_args.q_max_length,
        )

        if table_name is not None:
            df_db = generate_dataset.db_df
            data = []

            for batch_tokens, batch_ids in tqdm(
                zip(predict_results.predictions, predict_results.label_ids),
                desc="Processing",
            ):
                for tokens, docid in zip(batch_tokens, batch_ids):
                    query = fast_tokenizer.decode(tokens, skip_special_tokens=True)
                    data.append({"mid": docid.item(), "doctoquery": query})

            df_pred = pd.DataFrame(data)

            df_result = df_db.merge(df_pred, left_index=True, right_on="mid", how="left")

            destination_table_name = (
                f"{table_name}_d2q_q{run_args.num_return_sequences}"
            )

            write_to_db(df_result, destination_table_name)

        else:
            with open(
                f"{run_args.valid_file}.q{run_args.num_return_sequences}.docTquery", "w"
            ) as f:
                for batch_tokens, batch_ids in tqdm(
                    zip(predict_results.predictions, predict_results.label_ids),
                    desc="Writing file",
                ):
                    for tokens, docid in zip(batch_tokens, batch_ids):
                        query = fast_tokenizer.decode(tokens, skip_special_tokens=True)
                        jitem = json.dumps({"text_id": docid.item(), "text": query})
                        f.write(jitem + "\n")


    elif run_args.task == "test":

            from CE.utils.test.evaluation_BM25 import BM25EmailSearchEvaluator
            file_names, run_semantic, df = split_train_validate_test(run_args, local_rank)

            run_args.test_file = file_names["test"]
        
            decoder_trie = generate_trie_dict(df, tokenizer)

            def restrict_decode_vocab(batch_idx, prefix_beam):
                return decoder_trie.get(prefix_beam.tolist())

#        model,
#        tokenizer,
#        device,
#        restrict_decode_vocab,
#        id_max_length=20,
#        batch_size=8,
#
#        trainer = DSITrainer(
#            model=model,
#            tokenizer=tokenizer,
#            device=None,
#            args=training_args,
#            train_dataset=train_dataset,
#            eval_dataset=valid_dataset,
#            data_collator=IndexingCollator(
#                tokenizer,
#                padding="longest",
#            ),
#            compute_metrics=make_compute_metrics(
#                fast_tokenizer, train_dataset.valid_ids
#            ),
#            restrict_decode_vocab=restrict_decode_vocab ,
#            id_max_length=run_args.id_max_length,
#        )


    else:
        raise NotImplementedError(
            "--task should be in 'DSI' or 'docTquery' or 'generation'"
        )


if __name__ == "__main__":
    main()
