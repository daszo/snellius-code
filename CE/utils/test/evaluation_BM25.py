import json
import os
import shutil
import subprocess
import csv
from typing import List, Dict
from pyserini.search.lucene import LuceneSearcher
from CE.utils.database import save_result
from CE.utils.test.general import BaseMetricCalculator
from CE.utils.database import load_db
from tqdm import tqdm
import argparse
import sys


class BM25EmailSearchEvaluator(BaseMetricCalculator):
    def __init__(
        self,
        input_file,
        table_name,
        corpus_dir="corpus_data",
        index_dir="indexes/enron_index",
        threads=4,
    ):
        self.table_name = table_name
        self.input_file = input_file
        self.corpus_dir = corpus_dir + "_" + table_name
        self.index_dir = index_dir + "_" + table_name

        self.threads = threads

        self.mid_to_textid = {}
        self.queries = []

        # Storage
        self.execution_results = []
        self.final_metrics = []
        self.detailed_logs = []  # Added for CSV logging

    def prepare_data(self):
        """Prepares corpus and query list."""
        if os.path.exists(self.corpus_dir):
            shutil.rmtree(self.corpus_dir)
        os.makedirs(self.corpus_dir)

        # 1. Load Queries from Input File
        with open(self.input_file, "r", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                text_id = data.get("text_id", "").strip()

                if data.get("text"):
                    self.queries.append(
                        {"target_text_id": text_id, "query": data["text"]}
                    )

        # 2. Build Corpus from Database (for indexing)
        with open(
            os.path.join(self.corpus_dir, "docs.jsonl"), "w", encoding="utf-8"
        ) as fout:
            df = load_db(self.table_name)
            for _, row in df.iterrows():
                mid = str(row["mid"])
                body = row["body_clean_and_subject"]

                # Write for Pyserini
                fout.write(json.dumps({"id": mid, "contents": body}) + "\n")

                # Store mapping: mid (internal ID) -> text_id (Target ID)
                self.mid_to_textid[mid] = row["elaborative_description"].strip()

    def build_index(self):
        """Builds the Pyserini Lucene index."""
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir)

        cmd = [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            self.corpus_dir,
            "--index",
            self.index_dir,
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(self.threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
        subprocess.run(cmd, check=True)

    def run_retrieval_phase(self, k1=0.9, b=0.4):
        """Phase 1: Run all datapoints, extract ranks, and save detailed logs."""
        searcher = LuceneSearcher(self.index_dir)
        searcher.set_bm25(k1=k1, b=b)

        self.execution_results = []
        self.detailed_logs = []

        for q in tqdm(self.queries, desc="Running BM25 Retrieval"):
            query_text = q["query"]
            target_id = q["target_text_id"]

            # Get Top 20 Hits
            hits = searcher.search(query_text, k=20)

            # Convert Lucene Internal IDs (mid) to your Text IDs
            retrieved_text_ids = []
            for hit in hits:
                # Map mid -> text_id. Default to hit.docid if mapping fails
                mapped_id = self.mid_to_textid.get(hit.docid, hit.docid)
                retrieved_text_ids.append(mapped_id)

            # Calculate Rank
            rank = float("inf")
            if target_id in retrieved_text_ids:
                # +1 because index is 0-based
                rank = retrieved_text_ids.index(target_id) + 1

            self.execution_results.append(rank)

            # Store Log
            self.detailed_logs.append(
                {
                    "query": query_text,
                    "target_text_id": target_id,
                    "rank": rank if rank != float("inf") else -1,
                    # Join top 20 predictions with pipe for CSV
                    "model_predictions": " | ".join(retrieved_text_ids),
                }
            )

        # Save the logs immediately
        self.save_debug_logs()

    def save_debug_logs(self):
        """Saves the detailed retrieval logs to a CSV file."""
        # Save in the index directory or current dir, depending on preference.
        # Here I save it alongside the index dir for organization.

        name = self.input_file.split("/")[-1]
        log_path = os.path.join(self.index_dir, name + ".bm25_debug_logs.csv")
        print(f"Saving detailed inference logs to {log_path}...")

        headers = ["query", "target_text_id", "rank", "model_predictions"]

        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.detailed_logs)

    def compute_metrics(self):
        """Phase 2: Calculate metrics from stored ranks."""
        ranks = self.execution_results

        mrr3 = self.calculate_mrr(ranks, 3)
        mrr20 = self.calculate_mrr(ranks, 20)
        hits1 = self.calculate_hits(ranks, 1)
        hits10 = self.calculate_hits(ranks, 10)

        self.final_metrics = [f"{value:.4f}" for value in [mrr3, mrr20, hits1, hits10]]
        print(f"\nResults for: MRR@3: {mrr3:.4f}, Hits@1: {hits1:.4f}")

    def save_results(self, size: str, experiment_type: str, version: str = "v1.0"):
        data = ["BM25-base", size, experiment_type, version] + self.final_metrics
        save_result(tuple(data))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple argument parser")

    # Positional argument (Required)
    parser.add_argument("table_name", help="table name")
    args = parser.parse_args()
    table_name = args.table_name

    table_name1 = table_name

    evaluator = BM25EmailSearchEvaluator(
        input_file=f"data/test.{table_name1}.docTquery", table_name=table_name1
    )
    evaluator.prepare_data()
    evaluator.build_index()
    evaluator.run_retrieval_phase()
    evaluator.compute_metrics()
    evaluator.save_results("10k", "thread_same_mid")

    table_name2 = "N100k_thread"

    evaluator_ = BM25EmailSearchEvaluator(
        input_file=f"data/test.{table_name2}.docTquery", table_name=table_name2
    )
    evaluator_.prepare_data()
    evaluator_.build_index()
    evaluator_.run_retrieval_phase()
    evaluator_.compute_metrics()
    evaluator_.save_results("100k", "thread")
