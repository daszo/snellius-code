from pipelines.clean_files import clean_email_bodies_pipeline
from pipelines.run_textrank import run_text_rank
import argparse


def stage1():

    parser = argparse.ArgumentParser(description="A simple argument parser")

    # Positional argument (Required)
    parser.add_argument("--thread", help="thread", action="store_true")
    args = parser.parse_args()
    thread = args.thread
    clean_email_bodies_pipeline(keep_full_history=thread)

    if thread:
        run_text_rank(
            table="v_CleanMessages_thread",
            destination_table="text_rank_thread",
            thread=thread,
        )
    else:
        run_text_rank(table="v_CleanMessages", destination_table="full_text_rank")


if __name__ == "__main__":
    stage1()
