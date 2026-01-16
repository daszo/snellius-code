import json
import os
import shutil
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Union
from tqdm import tqdm
from data import IndexingCollator  # Assuming this exists in your project
from run import RunArguments
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from CE.utils.test.general import BaseMetricCalculator
from CE.utils.database import load_db, write_to_db, save_result
from CE.tries import generate_trie_dict


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
        return (inputs["input_ids"], str(item["text_id"])) 

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
        num_return_sequences=20,  # Added this to ensure consistency
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

                attention_mask = batch["attention_mask"].to(self.device)
                # Beam Search Generation
                # Returns (Batch * Num_Return_Sequences, Seq_Len)
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=self.id_max_length,
                    num_beams=self.num_return_sequences,  # Usually beams == return_seqs for retrieval
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=self.num_return_sequences,
                    early_stopping=True,
                )

                # Reshape to (Batch, Num_Beams, Seq_Len)
                # Important: The view must match the batch size of the current iteration
                current_batch_size = input_ids.shape[0]
                batch_preds = outputs.view(
                    current_batch_size, self.num_return_sequences, -1
                )

                all_preds.extend(batch_preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())

        return all_preds, all_labels


class DSIEmailSearchEvaluator(BaseMetricCalculator):
    def __init__(
        self,
        run_args,
        input_file: str,
        trainer=None,
        restrict_decode_vocab=None,
        df=None,
        model=None,
        tokenizer=None,
        eval_dir: str = "dsi_eval_cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        if trainer is None:
            tokenizer = AutoTokenizer.from_pretrained(run_args.model_name)
            if isinstance(model, str) or (model is None and run_args.model_name != ""):
                model = AutoModelForSeq2SeqLM.from_pretrained(model or run_args.model_name)
                tokenizer = AutoTokenizer.from_pretrained(run_args.model_name)

                if df is None:
                    raise ValueError(
                        "no df provided to build the tree for the restric decode vocab fn"
                    )

                if restrict_decode_vocab is None:
                    del restrict_decode_vocab
                    decoder_trie = generate_trie_dict(df, tokenizer)

                    def restrict_decode_vocab(batch_idx, prefix_beam):
                        return decoder_trie.get(prefix_beam.tolist())

            if model is None and (run_args is None or run_args.model_name == ""):
                raise ValueError("no model provided")

            self.model = model
            self.tokenizer = tokenizer
            if restrict_decode_vocab is None:
                raise ValueError("cant build the restrict_decode_vocab_fn, no model path is given. Provide the function or use the model path method")
            self.restrict_decode_vocab = restrict_decode_vocab
            self.trainer = None
        else:
            self.trainer = trainer
            self.model = None
            if tokenizer is None:
                raise ValueError("Tokenizer should be set if trainer is given.")
            self.tokenizer = tokenizer
            self.restrict_decode_vocab = None

        self.run_args = run_args
        self.input_file = input_file
        self.eval_dir = eval_dir
        self.device = device

        # DSI specific components
        self.id_max_length = getattr(run_args, "id_max_length", 20)
        self.batch_size = getattr(run_args, "per_device_eval_batch_size", 8)

        self.execution_results = []
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

                if data.get("text"):
                    entry = {"text": data["text"], "text_id": text_id}
                    fout.write(json.dumps(entry) + "\n")

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

        collator = IndexingCollator(
            tokenizer=self.tokenizer,
            padding="longest",  
        )

        if trainer := self.trainer:
            predictor = trainer 
        else:
        
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

            self.execution_results.append(rank)

    def compute_metrics(self):
        """Computes MRR and Hits metrics."""
        # Simplified to handle single query type
        ranks = self.execution_results
        
        if not ranks:
            print("No results for query")
            self.final_metrics = ["0.0000", "0.0000", "0.0000", "0.0000"]
            return

        mrr3 = self.calculate_mrr(ranks, 3)
        mrr20 = self.calculate_mrr(ranks, 20)
        hits1 = self.calculate_hits(ranks, 1)
        hits10 = self.calculate_hits(ranks, 10)

        print(f"\nResults: MRR@3: {mrr3:.4f}, Hits@1: {hits1:.4f}")
        
        # Save directly (no averaging needed)
        self.final_metrics = [f"{mrr3:.4f}", f"{mrr20:.4f}", f"{hits1:.4f}", f"{hits10:.4f}"]

    def save_results(self, size: str, experiment_type: str, version: str = "v1.0"):
        data = ["DSI-base", size, experiment_type, version] + self.final_metrics
        save_result(tuple(data))


# if __name__ == "__main__":
#     # --- 1. SETUP DUMMY ARGS ---
#     run_args = RunArguments(
#         model_name="local_models/google/mt5-base",
#         task="DSI",
#         db_name="data/enron.db",
#         train_size=0.8,
#         validate_size=0.1,
#         test_size=0.1,
#         id_max_length=10,
#         max_length=64,
#     )
#
#     # --- 2. SETUP DUMMY DATA ---
#     # Create a dummy dataframe and save it to jsonl for the evaluator to read
#     print("Generating dummy data...")
#     dummy_data = [
#         {
#             "text_id": "1001",
#             "text_rank_query": "budget meeting",
#             "doctoquery": "finance report",
#         },
#         {
#             "text_id": "1002",
#             "text_rank_query": "project launch",
#             "doctoquery": "marketing plan",
#         },
#         {
#             "text_id": "1003",
#             "text_rank_query": "holiday party",
#             "doctoquery": "hr announcement",
#         },
#     ]
#     input_file = "dummy_eval_input.jsonl"
#     with open(input_file, "w") as f:
#         for item in dummy_data:
#             f.write(json.dumps(item) + "\n")
#
#     # --- 4. DEFINE RESTRICTION FUNCTION ---
#     # For this test, we allow all tokens. In production, this uses your Trie.
#     def dummy_restrict_decode_vocab(batch_idx, prefix_input_ids):
#         return None  # None means all tokens allowed
#
#     # --- 5. RUN EVALUATOR ---
#     print("Initializing Evaluator...")
#     evaluator = DSIEmailSearchEvaluator(
#         model=run_args.model_name,
#         run_args=run_args,
#         input_file=input_file,
#         restrict_decode_vocab=dummy_restrict_decode_vocab,
#     )
#
#     print("Preparing Data...")
#     evaluator.prepare_data()
#
#     print("Running Retrieval Phase (Inference)...")
#     evaluator.run_retrieval_phase()
#
#     print("Computing Metrics...")
#     evaluator.compute_metrics()
#
#     print("Saving Results...")
#     evaluator.save_results(size="10k", experiment_type="base")
#
#     # Cleanup
#     if os.path.exists(input_file):
#         os.remove(input_file)

if "__main__" == __name__:
    run_args = RunArguments(
        model_name="t5-base",  # Use a small model for testing
        task="DSI",
        table_name="test_emails",  # Dummy table name
        db_name="data/enron.dc",  # Dummy DB
        train_size=0.8,
        validate_size=0.1,
        test_size=0.1,
        id_max_length=10,  # Short length for speed
    )

    table_name = run_args.table_name

    df = load_db(run_args.table_name, run_args.db_name)

    semantic_ids = df["elaborative_description"].map(type).eq(str).all()

    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    def count_t5_tokens(text):
        return len(tokenizer.encode(text))

    df["token_count"] = df["text"].apply(count_t5_tokens)
    longest_count = df["token_count"].max()

    run_args.id_max_length = longest_count

    evaluator = DSIEmailSearchEvaluator(
        model="/gpfs/work5/0/prjs1828/DSI-QG/models/enron-10k-mt5-base-DSI-Q-classicv1.2/checkpoint-44000",
        input_file="/gpfs/work5/0/prjs1828/DSI-QG/data/test.N10k.docTquery",
        run_args=run_args
    )

    print("Preparing Data...")
    evaluator.prepare_data()

    print("Running Retrieval Phase (Inference)...")
    evaluator.run_retrieval_phase()

    print("Computing Metrics...")
    evaluator.compute_metrics()

    print("Saving Results...")
    evaluator.save_results(size="10k", experiment_type="no_thread")
