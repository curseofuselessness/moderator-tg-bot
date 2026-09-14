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
def load_all_data(
    source: str = "raw",  # "raw" or "processed"
    include_external: bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Load data from one source.

    Args:
        source: "raw" — downloaded data, "processed" — processed data
        include_external: include your own labeled data
        drop_duplicates: delete duplicates
    """
    if source not in {"raw", "processed"}:
        raise ValueError(f"Unknown source: {source}")

    dfs = []

    if source == "raw":
        dfs.append(load_raw_01())
        # dfs.append(load_raw_02())   # добавить позже

    # elif source == "processed":
    # dfs.append(load_processed_01())   # добавить позже

    # if include_external:
    # dfs.append(load_external()) # добавить позже

    if not dfs:
        raise ValueError("No data sources enabled")

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
