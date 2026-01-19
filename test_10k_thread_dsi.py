from CE.utils.test.gr_evaluation import DSIEmailSearchEvaluator
from run import RunArguments
from CE.utils.database import load_db
from transformers import AutoTokenizer


def main():

    table_name = "N10k_thread"

    train_name = "enron-10k-t5-base-DSI-Q-thread"

    top_model = "checkpoint-38000"

    run_args = RunArguments(
        model_name=f"/gpfs/work5/0/prjs1828/DSI-QG/models/{train_name}/{top_model}",
        task="DSI",
        db_name="data/enron.db",
        train_size=0.8,
        validate_size=0.1,
        test_size=0.1,
        id_max_length=10,
        max_length=64,
        table_name=table_name,
    )

    df = load_db(run_args.table_name, run_args.db_name)

    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    def count_t5_tokens(text):
        return len(tokenizer.encode(text))

    df["token_count"] = df["elaborative_description"].apply(count_t5_tokens)
    run_args.id_max_length = df["token_count"].max()

    input_file = f"data/test.{table_name}.docTquery"

    evaluator = DSIEmailSearchEvaluator(
        model=run_args.model_name, run_args=run_args, input_file=input_file, df=df
    )

    print("Preparing Data...")
    evaluator.prepare_data()

    print("Running Retrieval Phase (Inference)...")
    evaluator.run_retrieval_phase()

    print("Computing Metrics...")
    evaluator.compute_metrics()

    print("Saving Results...")
    evaluator.save_results(size="10k", experiment_type="thread", version="v1.0")


main()
