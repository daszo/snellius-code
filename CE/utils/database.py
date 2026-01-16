import sqlite3
import pandas as pd

DB_PATH = "data/enron.db"


def load_db(table: str, db_path: str = DB_PATH) -> pd.DataFrame:

    print(f"Started, Loading DB {db_path}")

    conn = sqlite3.connect(db_path)

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
def finish_clean_message_and_drop_folders(
    keep_full_history=False, db_path: str = DB_PATH
):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Define the SQL for dropping/creating the base folder view (Run this FIRST)
    sql_drop_folders = """
    DROP VIEW IF EXISTS v_DroppedFolders;

    CREATE VIEW v_DroppedFolders AS
        SELECT
        m.*,
        s.similarities
        FROM Message m
        LEFT JOIN similarities s ON m.mid = s.mid
        WHERE folder NOT IN (
            'deleted_items',
            'calendar',
            'contacts',
            'notes',
            'drafts',
            'outbox',
            'discussion_threads',
            'all_documents',
            'notes_inbox'
        );
    """
    if keep_full_history:
        view_name = "v_CleanMessages_thread"
    else:
        view_name = "v_CleanMessages"

    # 2. Define the SQL for the cleaned messages view (Run this SECOND)
    sql_clean_messages = f"""
    DROP VIEW IF EXISTS {view_name};

    CREATE VIEW {view_name} AS
    SELECT * FROM (
        SELECT *
        FROM v_DroppedFolders
        WHERE clean_length_word > 30
        GROUP BY subject, body
    );
    """

    # 3. Execution (Assuming 'conn' is your database connection)
    # conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()

    # Execute drop_folders first as requested (and because v_CleanMessages depends on it)
    cursor.executescript(sql_drop_folders)

    # Execute cleaned_view second
    cursor.executescript(sql_clean_messages)

    conn.commit()


def combine_views(table_name):
    """
    v_CleanMessages_thread
    full_thread_d2q_q1
    text_rank_thread
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # SQL to create the view
    # Uses LEFT JOIN to keep all records from v_CleanMessages_thread
    # Aliases are used to select all columns from each table
    sql_create_view = f"""
    CREATE VIEW IF NOT EXISTS {table_name} AS
    SELECT 
        t1.*, 
        t2.*, 
        t3.*
    FROM v_CleanMessages_thread t1
    LEFT JOIN full_thread_d2q_q1 t2 ON t1.mid = t2.mid
    LEFT JOIN text_rank_thread t3 ON t1.mid = t3.mid;
    """

    try:
        cursor.execute(sql_create_view)
        conn.commit()
        print("View '{table_name}' created successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()
