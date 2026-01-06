import json
import os
import shutil
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Union
from tqdm import tqdm
from database import save_result
from data import IndexingCollator  # Assuming this exists in your project
from general import BaseMetricCalculator


class DSIEvalDataset(Dataset):
    """Simple dataset wrapper for inference."""

    def __init__(self, data_path, tokenizer, max_length):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
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
            padding=False,  # Collator handles padding
        )
        # Tokenize Target DocID for ground truth comparison
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length
        self.batch_size = batch_size

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
                # Returns (Batch * Num_Beams, Seq_Len)
                outputs = self.model.generate(
                    input_ids,
                    max_length=self.id_max_length,
                    num_beams=20,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=20,
                    early_stopping=True,
                )

                # Reshape to (Batch, Num_Beams, Seq_Len)
                # We assume 20 return sequences as per user config
                batch_preds = outputs.reshape(input_ids.shape[0], 20, -1)

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

        # 1. Setup Data
        dataset = DSIEvalDataset(
            data_path=self.valid_file_path,
            tokenizer=self.tokenizer,
            max_length=self.run_args.max_length,
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

            # Decode Target (clean up -100 if present from collator, though simple dataset likely just has pads)
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
            query_type = self.dataset_map[i]
            self.execution_results[query_type].append(rank)

    def compute_metrics(self):
        """Computes MRR and Hits metrics."""
        all_type_metrics = []
        for key in ["textrank", "d2q"]:
            ranks = self.execution_results[key]
            if not ranks:
                continue

            mrr3 = self.calculate_mrr(ranks, 3)
            mrr20 = self.calculate_mrr(ranks, 20)
            hits1 = self.calculate_hits(ranks, 1)
            hits10 = self.calculate_hits(ranks, 10)

            all_type_metrics.append([mrr3, mrr20, hits1, hits10])
            print(f"\nResults for {key}: MRR@3: {mrr3:.4f}, Hits@1: {hits1:.4f}")

        if len(all_type_metrics) == 2:
            self.final_metrics = [
                f"{(all_type_metrics[0][i] + all_type_metrics[1][i]) / 2:.4f}"
                for i in range(4)
            ]

    def save_results(self, size: str, experiment_type: str, version: str = "v1.0"):
        data = ["DSI-base", size, experiment_type, version] + self.final_metrics
        save_result(tuple(data))

if "__main__" == __name__:

table_name = run_args.table_name

df = load_db(run_args.table_name, run_args.db_name)

semantic_ids = df['elaborative_description'].map(type).eq(str).all()
if semantic_ids is True:

tokenizer = AutoTokenizer.from_pretrained("t5-base")
def count_t5_tokens(text):
    return len(tokenizer.encode(text))
df['token_count'] = df['text'].apply(count_t5_tokens)
longest_count = df['token_count'].max()

run_args.id_max_length = longest_count

evaluator = DSIEmailSearchEvaluator(
        model = "/enron-10k-mt5-base-DSI-Q-classic/checkpoint-15000/"
        )
