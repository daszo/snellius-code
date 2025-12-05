import sqlite3
import pandas as pd


def load_db(table: str, DB_PATH: str = "data/enron.db") -> pd.DataFrame:

    print(f"Started, Loading DB {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    sql_query = f"SELECT * FROM {table}"

    df = pd.read_sql_query(sql_query, conn)

    conn.commit()
    conn.close()

    print(f"Finished, Loading Dataframe of size {df.columns}")

    return df


def write_to_db(df: pd.DataFrame, table: str, DB_PATH: str = "data/enron.db"):

    print(f"Started, Writing {table} to database {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # Write to new table 'similarities'
    # if_exists='replace' drops the table if it exists and creates a new one
    # if_exists='append' adds to it
    df.to_sql(
        name=table,
        con=conn,
        if_exists="replace",
        index=False,
        chunksize=10000,  # Write in batches to save memory
    )

    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_mid ON Message (mid)")
    conn.commit()

    conn.close()

    print(f"Finished, Writing {table} to database {DB_PATH}")
