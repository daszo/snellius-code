import sqlite3
import pandas as pd

DB_PATH = "data/enron.db"


def load_db(table: str, db_path: str = DB_PATH) -> pd.DataFrame:

    print(f"Started, Loading DB {db_path}")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    sql_query = f"SELECT * FROM {table}"

    df = pd.read_sql_query(sql_query, conn)

    conn.commit()
    conn.close()

    print(f"Finished, Loading Dataframe of size {df.columns}")

    return df


def write_to_db(df: pd.DataFrame, table: str, db_path: str = DB_PATH):

    print(f"Started, Writing {table} to database {db_path}")

    conn = sqlite3.connect(db_path)

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

    print(f"Finished, Writing {table} to database {db_path}")


def save_result(data_tuple, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    sql = """
    INSERT INTO experiment_results (
        system, size, experiment_type, version, 
        mrr_3, mrr_20, hits_1, hits_10
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(system, size, experiment_type, version) 
    DO UPDATE SET 
        mrr_3=excluded.mrr_3,
        mrr_20=excluded.mrr_20,
        hits_1=excluded.hits_1,
        hits_10=excluded.hits_10;
    """

    cursor.execute(sql, data_tuple)
    conn.commit()
    conn.close()


# Example usage:
# data = ('BERT-Base', '110M', 'retrieval', 'v1.2', 0.45, 0.62, 0.31, 0.88)
# save_result('experiments.db', data)
