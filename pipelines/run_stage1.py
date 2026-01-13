from pipelines.clean_files import clean_email_bodies_pipeline
from pipelines.run_textrank import run_text_rank


def stage1():
#   clean_email_bodies_pipeline()
    run_text_rank(table="v_CleanMessages", destination_table="full_text_rank")


if __name__ == "__main__":
    stage1()
