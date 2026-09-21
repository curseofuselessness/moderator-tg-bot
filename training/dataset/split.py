"""Train/val/test splitting."""

import pandas as pd
from sklearn.model_selection import train_test_split

from shared.config.paths import PROCESSED_DIR

SEED = 42


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/val/test with stratification.

    Args:
        df: full DataFrame with 'text' and 'label'
        test_size: fraction for test (0.1 = 10%)
        val_size: fraction for val (0.1 = 10%)
        seed: random seed for reproducibility

    Returns:
        (train, val, test) DataFrames
    """
    # first split off test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )

    # then split val from remaining
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val["label"],
        random_state=seed,
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir=PROCESSED_DIR,
) -> None:
    """Save splits to processed/."""
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(f"Saved splits to {out_dir}")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load prepared splits from processed/."""
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train, val, test


if __name__ == "__main__":
    from training.dataset.load import load_all_data

    df = load_all_data()
    train, val, test = make_splits(df)

    print(f"Total: {len(df)} \n")
    print(f"Train: {len(train)} ({100*len(train)/len(df):.1f}%)")
    print(f"Val:   {len(val)} ({100*len(val)/len(df):.1f}%)")
    print(f"Test:  {len(test)} ({100*len(test)/len(df):.1f}%)")
    print(f"\nTrain labels:\n{train['label'].value_counts(normalize=True)}")
    print(f"\nVal labels:\n{val['label'].value_counts(normalize=True)}")
    print(f"\nTest labels:\n{test['label'].value_counts(normalize=True)}")

    save_splits(train, val, test)
