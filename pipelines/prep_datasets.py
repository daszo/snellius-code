from sklearn.model_selection import train_test_split
from CE.utils.database import load_db, write_to_db
import argparse


def create_10k_100k_dataset(table_name: str = "full_text_rank_d2q_q1"):

    df = load_db(table_name)

    df["strata_key"] = df["sender"].astype(str) + "_" + df["folder"].astype(str)

    # 2. Identify keys with at least 2 instances
    key_counts = df["strata_key"].value_counts()
    valid_keys = key_counts[key_counts >= 2].index

    # 3. Filter the dataframe
    df_valid = df[df["strata_key"].isin(valid_keys)]

    n10k_sampled_df, _ = train_test_split(
        df_valid, train_size=10000, stratify=df_valid["strata_key"], random_state=42
    )

    n10k_sampled_df = n10k_sampled_df.drop(columns=["strata_key"])

    write_to_db(n10k_sampled_df, "N10k")

    n100k_sampled_df, _ = train_test_split(
        df_valid, train_size=100000, stratify=df_valid["strata_key"], random_state=42
    )

    n100k_sampled_df = n100k_sampled_df.drop(columns=["strata_key"])

    write_to_db(n100k_sampled_df, "N100k")


def main():
    parser = argparse.ArgumentParser(description="A simple argument parser")

    # Positional argument (Required)
    parser.add_argument("table_name", help="table name")
    args = parser.parse_args()

    create_10k_100k_dataset(args.table_name)


if __name__ == "__main__":
    main()
    # create_10k_100k_dataset("Message")
