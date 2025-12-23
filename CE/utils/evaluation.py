import json
import os
import shutil
import subprocess
from pyserini.search.lucene import LuceneSearcher


class EmailSearchEvaluator:
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

        # State storage
        self.mid_to_textid = {}
        self.queries_textrank = []
        self.queries_d2q = []

    def prepare_data(self):
        """
        Parses the input file, creates the Pyserini corpus,
        and extracts queries and truth mappings.
        """
        print(f"--- Step 1: Preparing Data from {self.input_file} ---")

        # Reset corpus directory
        if os.path.exists(self.corpus_dir):
            shutil.rmtree(self.corpus_dir)
        os.makedirs(self.corpus_dir)

        seen_mids = set()

        with open(self.input_file, "r", encoding="utf-8") as fin, open(
            os.path.join(self.corpus_dir, "docs.jsonl"), "w", encoding="utf-8"
        ) as fout:

            for line in fin:
                try:
                    data = json.loads(line)

                    # Identifiers
                    mid = str(data.get("mid"))
                    text_id = data.get("text_id")
                    body = data.get("body", "")

                    # 1. Build Corpus (One entry per MID)
                    # We index by 'mid' because it is the unique key per row
                    if mid not in seen_mids:
                        entry = {"id": mid, "contents": body}
                        fout.write(json.dumps(entry) + "\n")
                        seen_mids.add(mid)

                        # Store mapping: MID -> Text ID (for flexible relevance matching)
                        self.mid_to_textid[mid] = text_id

                    # 2. Extract Queries
                    if data.get("text_rank_query"):
                        self.queries_textrank.append(
                            {
                                "mid": mid,
                                "target_text_id": text_id,
                                "query": data["text_rank_query"],
                            }
                        )

                    if data.get("doctoquery"):
                        self.queries_d2q.append(
                            {
                                "mid": mid,
                                "target_text_id": text_id,
                                "query": data["doctoquery"],
                            }
                        )

                except json.JSONDecodeError:
                    continue

        print(f"Processed {len(seen_mids)} unique documents.")
        print(
            f"Found {len(self.queries_textrank)} TextRank queries and {len(self.queries_d2q)} Doc2Query queries."
        )

    def build_index(self):
        """
        Runs the Pyserini (Lucene) indexer via subprocess.
        """
        print("\n--- Step 2: Building Index ---")

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
        print("Indexing complete.")

    def _calculate_metrics(self, searcher, queries, label):
        """
        Internal method to run search and calculate MRR and Hits stats.
        """
        print(f"\nEvaluating: {label}")
        print("-" * 40)

        if not queries:
            print("No queries to evaluate.")
            return

        metrics = {
            "mrr_3_sum": 0.0,
            "mrr_20_sum": 0.0,
            "hits_1_count": 0,
            "hits_10_count": 0,
        }

        count = len(queries)

        for q in queries:
            target_text_id = q["target_text_id"]
            query_text = q["query"]

            try:
                # Retrieve top 20 documents
                hits = searcher.search(query_text, k=20)
            except Exception:
                continue

            # Determine rank of the first relevant document
            first_relevant_rank = float("inf")

            for i, hit in enumerate(hits):
                # We retrieved a document ID (mid).
                # Is it relevant? Only if its 'text_id' matches our target.
                retrieved_mid = hit.docid
                retrieved_text_id = self.mid_to_textid.get(retrieved_mid)

                if retrieved_text_id == target_text_id:
                    first_relevant_rank = i + 1  # 1-based rank
                    break

            # Update Metrics
            if first_relevant_rank <= 1:
                metrics["hits_1_count"] += 1
            if first_relevant_rank <= 10:
                metrics["hits_10_count"] += 1
            if first_relevant_rank <= 3:
                metrics["mrr_3_sum"] += 1.0 / first_relevant_rank
            if first_relevant_rank <= 20:
                metrics["mrr_20_sum"] += 1.0 / first_relevant_rank

        # Print Results
        print(f"MRR@3:   {metrics['mrr_3_sum'] / count:.4f}")
        print(f"MRR@20:  {metrics['mrr_20_sum'] / count:.4f}")
        print(f"Hits@1:  {metrics['hits_1_count'] / count:.4f}")
        print(f"Hits@10: {metrics['hits_10_count'] / count:.4f}")

    def run_evaluation(self, k1=0.9, b=0.4):
        """
        Loads the index and runs the evaluation logic for both query types.
        """
        print("\n--- Step 3: Evaluation ---")
        searcher = LuceneSearcher(self.index_dir)
        searcher.set_bm25(k1=k1, b=b)

        self._calculate_metrics(searcher, self.queries_textrank, "TextRank Queries")
        self._calculate_metrics(searcher, self.queries_d2q, "Doc2Query Queries")

    def run_pipeline(self):
        """
        Orchestrates the full flow.
        """
        self.prepare_data()
        self.build_index()
        self.run_evaluation()


# --- Usage ---
if __name__ == "__main__":
    evaluator = EmailSearchEvaluator(input_file="test.N10k_text_rank_d2q_q1.docTquery")
    evaluator.run_pipeline()
