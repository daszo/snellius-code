import sqlite3
import pandas as pd

DB_PATH = "data/enron.db"

experiment = True

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

if experiment:
    sql_query = "SELECT * FROM N10K"
else:
    sql_query = "SELECT * FROM N100K"

df = pd.read_sql_query(sql_query, conn)

conn.commit()
conn.close()

df.columns
from summa import summarizer
from summa import keywords

print("loaded_table")


def get_keywords_safe(text, num_words=8):
    """
    Wrapper for summa.keywords to handle IndexError on short texts.
    Summa throws an IndexError if the text has fewer tokens than 'words'.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        # Summa returns a newline-separated string
        kw_str = keywords.keywords(text, words=num_words)
        return kw_str.split("\n") if kw_str else []
    except IndexError:
        kw_str = keywords.keywords(text)
        return kw_str.split("\n") if kw_str else []
    except Exception:
        return text.split()


def calculate_query_and_ed(row):

    # try:
    text = f"{row['subject']} \n {row['body_clean']}"

    row["text_rank_query"] = summarizer.summarize(text, words=15)

    # --- 2. Generate "Elaborative Description" (Keyword Extraction) ---
    # We want a list of salient terms that describe the document content.
    # This creates a "bag of entities" representation useful for GR.
    # row['elaborative_description'] = keywords.keywords(text, words=8)
    row["elaborative_description"] = get_keywords_safe(text, 8)

    return row



print("started generation")
df_queries = df.apply(lambda x: calculate_query_and_ed(x), axis=1)
print("ended generation")

df["char_count"] = df["body_clean"].str.len()

# Sort from short to long
df_sorted = df.sort_values(by="char_count", ascending=True)

# Display result
print(df_sorted.iloc[3]["body_clean"])

df_table_name = "body_clean_and_subject"
# 1. Clean the body column first (Vectorized)
cleaned_body = df["body_clean"].astype(str).str.replace(r"[\n\r\t]", " ", regex=True)

# 2. Concatenate strings element-wise (Vectorized)
df[df_table_name] = "subject: " + df["subject"].astype(str) + " body: " + cleaned_body


conn = sqlite3.connect(DB_PATH)

# Write to new table 'similarities'
# if_exists='replace' drops the table if it exists and creates a new one
# if_exists='append' adds to it
df.to_sql(
    name="N10k_text_rank",
    con=conn,
    if_exists="replace",
    index=False,
    chunksize=10000,  # Write in batches to save memory
)

cursor = conn.cursor()
cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_mid ON Message (mid)")
conn.commit()

conn.close()
