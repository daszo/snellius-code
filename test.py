from CE.utils.test.gr_evaluation import DSIEmailSearchEvaluator
from run import RunArguments
from CE.utils.database import load_db


def main():

    run_args = RunArguments(
        model_name="models/enron-10k-mt5-base-DSI-Q-classic/checkpoint-15000",
        task="DSI",
        db_name="data/enron.db",
        train_size=0.8,
        validate_size=0.1,
        test_size=0.1,
        id_max_length=10,
        max_length=64,
        table_name="N10k_text_rank_d2q_q1",
    )

    df = load_db(run_args.table_name, run_args.db_name)

    input_file = "data/test.N10k_text_rank_d2q_q1.docTquery"

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
    evaluator.save_results(size="10k", experiment_type="base")


main()
