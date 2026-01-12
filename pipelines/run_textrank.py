import sqlite3
import pandas as pd
from CE.utils.database import load_db, write_to_db
from summa import summarizer
from summa import keywords


def get_keywords_safe(text, num_words=8):
    """
    Wrapper for summa.keywords to handle IndexError on short texts.
    Summa throws an IndexError if the text has fewer tokens than 'words'.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        # Summa returns a newline-separated string
        kw_str = keywords.keywords(text, words=num_words)
        return " ".join(kw_str.split("\n")) if kw_str else ""
    except IndexError:
        kw_str = keywords.keywords(text)
        return " ".join(kw_str.split("\n")) if kw_str else ""
    except Exception:
        return " ".join(text.split())


def calculate_query_and_ed(row):

    text = f"{row['subject']} \n {row['body_clean']}"

    row["text_rank_query"] = summarizer.summarize(text, words=15)

    # --- 2. Generate "Elaborative Description" (Keyword Extraction) ---
    # We want a list of salient terms that describe the document content.
    # This creates a "bag of entities" representation useful for GR.
    # row['elaborative_description'] = keywords.keywords(text, words=8)
    row["elaborative_description"] = get_keywords_safe(text, 8)

    return row


def run_text_rank(
    table="N10k",
    destination_table="N10k_text_rank",
):

    df = load_db(table)

    df_queries = df.apply(lambda x: calculate_query_and_ed(x), axis=1)

    print("Finished text rank query generation")

    write_to_db(df_queries, destination_table)
