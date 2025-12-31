import torch
import os
import json
import shutil
from unittest.mock import MagicMock
from types import SimpleNamespace

from gr_evaluation import DSIEmailSearchEvaluator

# Make sure to import your class:
# from your_module import DSIEmailSearchEvaluator


def test_dsi_evaluator():
    print("--- Starting DSI Evaluator Test ---")

    # 1. Setup Mock Components
    # -------------------------

    # Mock Tokenizer: Basic encode/decode logic
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    def mock_call(text, **kwargs):
        # Returns dummy tensors.
        # For labels (ids), let's pretend the token ID is the integer value of the char (e.g., "1" -> 1)
        if text.isdigit():
            ids = [int(c) for c in text]
        else:
            ids = [10, 11, 12]  # dummy query tokens
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids))}

    tokenizer.side_effect = mock_call

    def mock_decode(token_ids, skip_special_tokens=True):
        # Decodes back to string. Assumes input is list or tensor.
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        # Filter out 0 (pad) and 1 (eos) and join
        cleaned = [str(t) for t in token_ids if t > 1]
        return "".join(cleaned)

    tokenizer.decode.side_effect = mock_decode

    # Mock Model: Generates specific beams to test ranking
    model = MagicMock()
    model.config.pad_token_id = 0

    def mock_generate(input_ids, **kwargs):
        # We need to return shape (Batch * Num_Beams, Seq_Len)
        # We simulate a batch size of 1 for simplicity in this test loop
        num_beams = kwargs.get("num_beams", 20)
        bs = input_ids.shape[0]

        # Create a tensor filled with "wrong" answers (e.g., id "999")
        # [9, 9, 9, 1 (eos), 0 (pad)]
        wrong_seq = torch.tensor([9, 9, 9, 1, 0])
        out = wrong_seq.repeat(bs * num_beams, 1)

        # Inject the "correct" answer ("123") into the FIRST beam (Rank 1)
        # "123" -> tokens [1, 2, 3] + eos [1]
        correct_seq = torch.tensor([1, 2, 3, 1, 0])

        # We only inject the correct answer for the FIRST item in the batch
        # This allows us to test a "success" case (Rank 1) and a "fail" case (Rank inf)
        # depending on what the ground truth is in the JSON file.
        out[0] = correct_seq

        return out

    model.generate.side_effect = mock_generate

    # Mock Arguments
    run_args = SimpleNamespace(
        max_length=32,
        id_max_length=10,
        per_device_eval_batch_size=1,
        top_k=10,
        num_return_sequences=20,
    )

    # Mock Restrict Vocab (dummy)
    def restrict_decode_vocab(batch_idx, prefix_beam):
        return [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 2. Setup Dummy Data
    # -------------------
    input_file = "test_input_temp.jsonl"
    with open(input_file, "w") as f:
        # Case 1: TextRank Query. Ground Truth = 123.
        # Our mock model puts "123" at Rank 1. -> Expected Rank: 1
        f.write(
            json.dumps(
                {
                    "text_id": 123,
                    "text_rank_query": "query_success",
                    "body_clean": "content",
                }
            )
            + "\n"
        )

        # Case 2: Doc2Query. Ground Truth = 456.
        # Our mock model puts "123" at Rank 1 and "999" elsewhere. -> Expected Rank: inf
        f.write(
            json.dumps(
                {"text_id": 456, "doctoquery": "query_fail", "body_clean": "content"}
            )
            + "\n"
        )

    # 3. Run Pipeline
    # ---------------
    try:
        evaluator = DSIEmailSearchEvaluator(
            model=model,
            tokenizer=tokenizer,
            run_args=run_args,
            input_file=input_file,
            restrict_decode_vocab=restrict_decode_vocab,
            eval_dir="test_eval_cache_temp",
            device="cpu",
        )

        print("1. Testing prepare_data()...")
        evaluator.prepare_data()
        if os.path.exists("test_eval_cache_temp/valid.jsonl"):
            print("   [Pass] Valid file created.")
        else:
            print("   [Fail] Valid file missing.")

        print("2. Testing run_retrieval_phase()...")
        evaluator.run_retrieval_phase()

        # Check internal state
        tr_ranks = evaluator.execution_results["textrank"]
        d2q_ranks = evaluator.execution_results["d2q"]

        print(f"   TextRank Ranks (Expect [1]): {tr_ranks}")
        print(f"   D2Q Ranks (Expect [inf]): {d2q_ranks}")

        assert tr_ranks == [1], "TextRank retrieval failed or ranked incorrectly."
        assert d2q_ranks == [float("inf")], "D2Q should have failed retrieval."

        print("3. Testing compute_metrics()...")
        evaluator.compute_metrics()

        # TextRank: MRR@3 = 1.0
        # D2Q: MRR@3 = 0.0
        # Avg MRR@3 should be 0.5
        final_mrr3 = float(evaluator.final_metrics[0])
        print(f"   Final Aggregated MRR@3 (Expect 0.5): {final_mrr3}")

        assert final_mrr3 == 0.5, f"Metrics calculation incorrect. Got {final_mrr3}"
        print("   [Pass] Metrics calculated correctly.")

    except Exception as e:
        print(f"\n[ERROR] Test Failed: {e}")
        raise e
    finally:
        # Cleanup
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists("test_eval_cache_temp"):
            shutil.rmtree("test_eval_cache_temp")
        print("--- Test Cleanup Complete ---")


if __name__ == "__main__":
    test_dsi_evaluator()
