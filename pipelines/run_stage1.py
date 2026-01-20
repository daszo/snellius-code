from pipelines.clean_files import clean_email_bodies_pipeline
from pipelines.run_textrank import run_text_rank
import argparse


def stage1():

    parser = argparse.ArgumentParser(description="A simple argument parser")

    # Positional argument (Required)
    parser.add_argument("--thread", help="thread", action="store_true")
    parser.add_argument("--table_from", help="Source table name", type=str)
    parser.add_argument("--table_to", help="Destination table name", type=str)
    parser.add_argument("--destination_table", help="Destination table name", type=str)
    args = parser.parse_args()
    thread = args.thread

    table_from = args.table_from
    table_to = args.table_to
    destination_table=args.destination_table

    if table_from and table_to:
        clean_email_bodies_pipeline(keep_full_history=thread, table_from=table_from, table_to=table_to)
    elif table_from:
        clean_email_bodies_pipeline(keep_full_history=thread, table_from=table_from)
    elif table_to:
        clean_email_bodies_pipeline(keep_full_history=thread, table_to=table_to)
    else:
        clean_email_bodies_pipeline(keep_full_history=thread)

    if table_to:
        if destination_table is None:
            raise ValueError("no destination table given")
        if thread:
            run_text_rank(
                table=table_to,
                destination_table=destination_table,
                thread=thread,
            )
        else:
            run_text_rank(
                table=table_to,
                destination_table=destination_table,
            )

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
