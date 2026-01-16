from sklearn.model_selection import train_test_split
from CE.utils.database import load_db, write_to_db, combine_views
import argparse


def create_10k_100k_dataset(table_name: str = "full_text_rank_d2q_q1", thread=False):

    if thread:
        combine_views(table_name)

    df = load_db(table_name)
    df.drop(columns=["mid_x", "mid_y"], inplace=True)

    df["strata_key"] = df["sender"].astype(str) + "_" + df["folder"].astype(str)

    # 2. Identify keys with at least 2 instances
    key_counts = df["strata_key"].value_counts()
    valid_keys = key_counts[key_counts >= 2].index

    # 3. Filter the dataframe
    df_valid = df[df["strata_key"].isin(valid_keys)]

    n10k_name = "N10k"
    n100k_name = "N100k"
    if thread:
        n10k_name += "_thread"
        n100k_name += "_thread"
        n10k_same_mid_name = n10k_name + "_same_mid"

        df_10k = load_db("N10k")

        df_10k_thread_same_mid = df_valid[df_valid["mid"].isin(df_10k["mid"])]

        n10k_same_mid_sampled_df, _ = train_test_split(
            df_10k_thread_same_mid,
            train_size=10_000,
            stratify=df_valid["strata_key"],
            random_state=42,
        )

        n10k_same_mid_sampled_df = n10k_same_mid_sampled_df.drop(columns=["strata_key"])

        write_to_db(n10k_same_mid_sampled_df, n10k_same_mid_name)

    n10k_sampled_df, _ = train_test_split(
        df_valid, train_size=10_000, stratify=df_valid["strata_key"], random_state=42
    )

    n10k_sampled_df = n10k_sampled_df.drop(columns=["strata_key"])

    write_to_db(n10k_sampled_df, n10k_name)

    n100k_sampled_df, _ = train_test_split(
        df_valid, train_size=100_000, stratify=df_valid["strata_key"], random_state=42
    )

    n100k_sampled_df = n100k_sampled_df.drop(columns=["strata_key"])

    write_to_db(n100k_sampled_df, n100k_name)


def main():
    parser = argparse.ArgumentParser(description="A simple argument parser")

    # Positional argument (Required)
    parser.add_argument("table_name", help="table name")
    parser.add_argument("--thread", help="thread", action="store_true")
    args = parser.parse_args()
    thread = args.thread

    create_10k_100k_dataset(args.table_name, thread)


if __name__ == "__main__":
    main()
    # create_10k_100k_dataset("Message")
