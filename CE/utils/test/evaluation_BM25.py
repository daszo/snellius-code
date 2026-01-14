import json
import os
import shutil
import subprocess
from typing import List, Dict
from pyserini.search.lucene import LuceneSearcher
from CE.utils.database import save_result
from CE.utils.test.general import BaseMetricCalculator
from CE.utils.database import load_db
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
        self.corpus_dir = corpus_dir
        self.index_dir = index_dir
        self.threads = threads

        self.mid_to_textid = {}
        self.queries = []
        # self.queries_textrank = []
        # self.queries_d2q = []

        # Phase 1 storage: Raw Ranks
        self.execution_results = []
        self.final_metrics = []

    def prepare_data(self):
        if os.path.exists(self.corpus_dir):
            shutil.rmtree(self.corpus_dir)
        os.makedirs(self.corpus_dir)

        with open(self.input_file, "r", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                text_id, body = (
                    data.get("text_id"),
                    data.get("body_clean_and_subject", ""),
                )

                # fout.write(json.dumps({"id": mid, "contents": body}) + "\n")
                # self.mid_to_textid[mid] = text_id
                if data.get("text"):
                    self.queries.append(
                        {"target_text_id": text_id, "query": data["text"]}
                    )

                # if data.get("text_rank_query"):
                #     self.queries_textrank.append(
                #         {"target_text_id": text_id, "query": data["text_rank_query"]}
                #     )
                # if data.get("doctoquery"):
                #     self.queries_d2q.append(
                #         {"target_text_id": text_id, "query": data["doctoquery"]}
                # )
        with open(
            os.path.join(self.corpus_dir, "docs.jsonl"), "w", encoding="utf-8"
        ) as fout:

            df = load_db(self.table_name)
            for _, row in df.iterrows():
                mid = row["mid"]
                body = row["body_clean_and_subject"]
                fout.write(json.dumps({"id": mid, "contents": body}) + "\n")
                self.mid_to_textid[mid] = row["elaborative_description"]

    def build_index(self):
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
        """Phase 1: Run all datapoints and extract ranks."""
        searcher = LuceneSearcher(self.index_dir)
        searcher.set_bm25(k1=k1, b=b)

        ranks = []
        for q in self.queries:
            hits = searcher.search(q["query"], k=20)
            rank = float("inf")
            for i, hit in enumerate(hits):
                if self.mid_to_textid.get(hit.docid) == q["target_text_id"]:
                    rank = i + 1
                    break
            ranks.append(rank)
        self.execution_results = ranks

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

    evaluator = BM25EmailSearchEvaluator(
        input_file=f"data/test.{table_name}.docTquery", table_name=table_name
    )
    evaluator.prepare_data()
    evaluator.build_index()
    evaluator.run_retrieval_phase()
    evaluator.compute_metrics()
    evaluator.save_results("10k", "no_thread")
