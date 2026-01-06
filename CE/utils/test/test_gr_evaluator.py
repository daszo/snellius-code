import json
import os
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Union, Tuple
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding

# --- MOCKS FOR MISSING DEPENDENCIES ---
# Replacing 'database', 'data', 'general', 'run' imports with placeholders
# so this script runs standalone.

class RunArguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class BaseMetricCalculator:
    def calculate_mrr(self, ranks, k):
        return np.mean([1.0 / r if r <= k else 0.0 for r in ranks])

    def calculate_hits(self, ranks, k):
        return np.mean([1.0 if r <= k else 0.0 for r in ranks])

def save_result(data):
    print(f"--> [MOCK DB SAVE] Saved results: {data}")

class IndexingCollator:
    """Mock Collator that pads input_ids."""
    def __init__(self, tokenizer, padding):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        
        # Pad inputs
        inputs_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # Pad labels (targets)
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        attention_mask = (inputs_padded != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": inputs_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded
        }

# --- END MOCKS ---

class DSIEvalDataset(Dataset):
    """Simple dataset wrapper for inference."""

    def __init__(self, data_path, tokenizer, max_length):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Tokenize Query
        inputs = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False, 
        )
        # Tokenize Target DocID for ground truth comparison
        # We assume text_id is a string. If it's an int, convert it.
        labels = self.tokenizer(
            str(item["text_id"]), truncation=True, max_length=64, padding=False
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"],
        }


class DSIPredictor:
    """Lightweight inference wrapper replacing the Trainer."""

    def __init__(
        self,
        model,
        tokenizer,
        device,
        restrict_decode_vocab,
        id_max_length=20,
        batch_size=8,
        num_return_sequences=20, # Added this to ensure consistency
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length
        self.batch_size = batch_size
        self.num_return_sequences = num_return_sequences

    def predict(self, dataset, collator):
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=collator, shuffle=False
        )

        self.model.eval()
        self.model.to(self.device)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Running Inference"):
                input_ids = batch["input_ids"].to(self.device)
                
                # Beam Search Generation
                # Returns (Batch * Num_Return_Sequences, Seq_Len)
                outputs = self.model.generate(
                    input_ids,
                    max_length=self.id_max_length,
                    num_beams=self.num_return_sequences, # Usually beams == return_seqs for retrieval
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=self.num_return_sequences,
                    early_stopping=True,
                )

                # Reshape to (Batch, Num_Beams, Seq_Len)
                # Important: The view must match the batch size of the current iteration
                current_batch_size = input_ids.shape[0]
                batch_preds = outputs.view(current_batch_size, self.num_return_sequences, -1)

                all_preds.extend(batch_preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())

        return all_preds, all_labels


class DSIEmailSearchEvaluator(BaseMetricCalculator):
    def __init__(
        self,
        model,
        tokenizer,
        run_args,
        input_file: str,
        restrict_decode_vocab,
        eval_dir: str = "dsi_eval_cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.run_args = run_args
        self.input_file = input_file
        self.eval_dir = eval_dir
        self.device = device

        # DSI specific components
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = getattr(run_args, "id_max_length", 20)
        self.batch_size = getattr(run_args, "per_device_eval_batch_size", 8)

        self.dataset_map = []  # Tracks order of query types
        self.execution_results = {"textrank": [], "d2q": []}
        self.final_metrics = []
        self.valid_file_path = None

    def prepare_data(self):
        """Prepares the validation JSONL file."""
        if os.path.exists(self.eval_dir):
            shutil.rmtree(self.eval_dir)
        os.makedirs(self.eval_dir)

        valid_file_path = os.path.join(self.eval_dir, "valid.jsonl")

        with open(self.input_file, "r", encoding="utf-8") as fin, open(
            valid_file_path, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                data = json.loads(line)
                text_id = data.get("text_id")

                # We create separate entries for different query generation methods
                if data.get("text_rank_query"):
                    entry = {"text": data["text_rank_query"], "text_id": text_id}
                    fout.write(json.dumps(entry) + "\n")
                    self.dataset_map.append("textrank")

                if data.get("doctoquery"):
                    entry = {"text": data["doctoquery"], "text_id": text_id}
                    fout.write(json.dumps(entry) + "\n")
                    self.dataset_map.append("d2q")

        self.valid_file_path = valid_file_path

    def run_retrieval_phase(self):
        """Runs the DSIPredictor and calculates ranks."""
        if not self.valid_file_path:
            raise ValueError("Run prepare_data() before run_retrieval_phase()")

        # 1. Setup Data
        dataset = DSIEvalDataset(
            data_path=self.valid_file_path,
            tokenizer=self.tokenizer,
            max_length=getattr(self.run_args, "max_length", 128),
        )

        collator = IndexingCollator(self.tokenizer, padding="longest")

        # 2. Setup Predictor
        predictor = DSIPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            restrict_decode_vocab=self.restrict_decode_vocab,
            id_max_length=self.id_max_length,
            batch_size=self.batch_size,
        )

        # 3. Run Inference
        predictions, labels = predictor.predict(dataset, collator)

        # 4. Calculate Ranks
        # predictions is list of shape (Batch, 20 beams, Seq_Len)
        for i, beam_tokens_list in enumerate(predictions):
            target_tokens = labels[i]

            # Decode Target 
            target_doc_id = self.tokenizer.decode(
                [t for t in target_tokens if t != -100], skip_special_tokens=True
            ).strip()

            rank = float("inf")

            # Check beams
            for r, beam_toks in enumerate(beam_tokens_list):
                pred_doc_id = self.tokenizer.decode(
                    beam_toks, skip_special_tokens=True
                ).strip()

                if pred_doc_id == target_doc_id:
                    rank = r + 1
                    break

            # Record Rank
            # Map index i back to the query type
            if i < len(self.dataset_map):
                query_type = self.dataset_map[i]
                self.execution_results[query_type].append(rank)

    def compute_metrics(self):
        """Computes MRR and Hits metrics."""
        all_type_metrics = []
        for key in ["textrank", "d2q"]:
            ranks = self.execution_results[key]
            if not ranks:
                print(f"No results for {key}")
                # Append zeros to keep indexing consistent if one missing
                all_type_metrics.append([0.0, 0.0, 0.0, 0.0])
                continue

            mrr3 = self.calculate_mrr(ranks, 3)
            mrr20 = self.calculate_mrr(ranks, 20)
            hits1 = self.calculate_hits(ranks, 1)
            hits10 = self.calculate_hits(ranks, 10)

            all_type_metrics.append([mrr3, mrr20, hits1, hits10])
            print(f"\nResults for {key}: MRR@3: {mrr3:.4f}, Hits@1: {hits1:.4f}")

        # Average across the two query types if both exist
        if len(all_type_metrics) >= 1:
             # Calculate average across the types we found
            sums = [sum(x) for x in zip(*all_type_metrics)]
            count = len(all_type_metrics)
            self.final_metrics = [f"{s/count:.4f}" for s in sums]

    def save_results(self, size: str, experiment_type: str, version: str = "v1.0"):
        data = ["DSI-base", size, experiment_type, version] + self.final_metrics
        save_result(tuple(data))


if __name__ == "__main__":
    # --- 1. SETUP DUMMY ARGS ---
    run_args = RunArguments(
        model_name="local_models/google/mt5-base",
        task="DSI",
        db_name="data/enron.db",
        train_size=0.8,
        validate_size=0.1,
        test_size=0.1,
        id_max_length=10,
        max_length=64,
        per_device_eval_batch_size=2
    )

    # --- 2. SETUP DUMMY DATA ---
    # Create a dummy dataframe and save it to jsonl for the evaluator to read
    print("Generating dummy data...")
    dummy_data = [
        {"text_id": "1001", "text_rank_query": "budget meeting", "doctoquery": "finance report"},
        {"text_id": "1002", "text_rank_query": "project launch", "doctoquery": "marketing plan"},
        {"text_id": "1003", "text_rank_query": "holiday party", "doctoquery": "hr announcement"}
    ]
    input_file = "dummy_eval_input.jsonl"
    with open(input_file, "w") as f:
        for item in dummy_data:
            f.write(json.dumps(item) + "\n")

    # --- 3. LOAD MODEL & TOKENIZER ---
    print(f"Loading model {run_args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(run_args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(run_args.model_name)

    # --- 4. DEFINE RESTRICTION FUNCTION ---
    # For this test, we allow all tokens. In production, this uses your Trie.
    def dummy_restrict_decode_vocab(batch_idx, prefix_input_ids):
        return None # None means all tokens allowed
    
    # --- 5. RUN EVALUATOR ---
    print("Initializing Evaluator...")
    evaluator = DSIEmailSearchEvaluator(
        model=model,
        tokenizer=tokenizer,
        run_args=run_args,
        input_file=input_file,
        restrict_decode_vocab=dummy_restrict_decode_vocab
    )

    print("Preparing Data...")
    evaluator.prepare_data()
    
    print("Running Retrieval Phase (Inference)...")
    evaluator.run_retrieval_phase()
    
    print("Computing Metrics...")
    evaluator.compute_metrics()
    
    print("Saving Results...")
    evaluator.save_results(size="small", experiment_type="test_run")
    
    # Cleanup
    if os.path.exists(input_file):
        os.remove(input_file)
    print("Done.")
