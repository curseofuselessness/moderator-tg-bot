"""Data loading from all sources."""

import pandas as pd

from shared.config.paths import RAW_DIR


# ==============================
# Specific sources
# ==============================
def load_raw_01() -> pd.DataFrame:
    """Load 01_ru_toxic_comments_14k.csv from raw — Small dataset with labeled comments from 2ch.hk and pikabu.ru, 14412 rows"""
    path = RAW_DIR / "01_ru_toxic_comments_14k.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path)

    df = df.rename(columns={"comment": "text", "toxic": "label"})  # Rename columns

    return df


# ==============================
# Main function
# ==============================
def load_all_data() -> pd.DataFrame:
    """
    Load data from one source.
    """

    dfs = [load_raw_01()]

    df = pd.concat(dfs, ignore_index=True)

    return df


# ==============================
# CLI
# ==============================
if __name__ == "__main__":
    df = load_all_data()
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("\nFirst rows:")
    print(df.head())
