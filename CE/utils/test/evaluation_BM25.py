import json
import os
import shutil
import subprocess
from typing import List, Dict
from pyserini.search.lucene import LuceneSearcher
from database import save_result
from general import BaseMetricCalculator


class BM25EmailSearchEvaluator(BaseMetricCalculator):
    def __init__(
        self,
        input_file,
        corpus_dir="corpus_data",
        index_dir="indexes/enron_index",
        threads=4,
    ):
        self.input_file = input_file
        self.corpus_dir = corpus_dir
        self.index_dir = index_dir
        self.threads = threads

        self.mid_to_textid = {}
        self.queries_textrank = []
        self.queries_d2q = []

        # Phase 1 storage: Raw Ranks
        self.execution_results = {"textrank": [], "d2q": []}
        self.final_metrics = []

    def prepare_data(self):
        if os.path.exists(self.corpus_dir):
            shutil.rmtree(self.corpus_dir)
        os.makedirs(self.corpus_dir)

        seen_mids = set()
        with open(self.input_file, "r", encoding="utf-8") as fin, open(
            os.path.join(self.corpus_dir, "docs.jsonl"), "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                data = json.loads(line)
                mid, text_id, body = (
                    str(data.get("mid")),
                    data.get("text_id"),
                    data.get("subject") + "\n" + data.get("body_clean", ""),
                )

                if mid not in seen_mids:
                    fout.write(json.dumps({"id": mid, "contents": body}) + "\n")
                    seen_mids.add(mid)
                    self.mid_to_textid[mid] = text_id

                if data.get("text_rank_query"):
                    self.queries_textrank.append(
                        {"target_text_id": text_id, "query": data["text_rank_query"]}
                    )
                if data.get("doctoquery"):
                    self.queries_d2q.append(
                        {"target_text_id": text_id, "query": data["doctoquery"]}
                    )

    def build_index(self):
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir)
        cmd = [
            "python",
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

        for key, query_set in [
            ("textrank", self.queries_textrank),
            ("d2q", self.queries_d2q),
        ]:
            ranks = []
            for q in query_set:
                hits = searcher.search(q["query"], k=20)
                rank = float("inf")
                for i, hit in enumerate(hits):
                    if self.mid_to_textid.get(hit.docid) == q["target_text_id"]:
                        rank = i + 1
                        break
                ranks.append(rank)
            self.execution_results[key] = ranks

    def compute_metrics(self):
        """Phase 2: Calculate metrics from stored ranks."""
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

        # Aggregate across both query types (averaging the sets)
        if len(all_type_metrics) == 2:
            self.final_metrics = [
                f"{(all_type_metrics[0][i] + all_type_metrics[1][i]) / 2:.4f}"
                for i in range(4)
            ]

    def save_results(self, size: str, experiment_type: str, version: str = "v1.0"):
        data = ["BM25-base", size, experiment_type, version] + self.final_metrics
        save_result(tuple(data))


if __name__ == "__main__":
    evaluator = BM2EmailSearchEvaluator(input_file="test.jsonl")
    evaluator.prepare_data()
    evaluator.build_index()
    evaluator.run_retrieval_phase()
    evaluator.compute_metrics()
    evaluator.save_results("10k", "base")
