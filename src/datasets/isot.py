"""
Read raw files from ISOT dataset
Create text field 
Label encoding + splitting (deterministic, with seed)
Save to data/processed
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.text_cleaning import clean_text
from src.common.seed import set_seed

@dataclass(frozen=True)
class IsotSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

def load_isot(fake_csv: str | Path, true_csv: str | Path) -> pd.DataFrame:
    fake_df = pd.read_csv(fake_csv)
    true_df = pd.read_csv(true_csv)

    # Label encoding: fake=1, true=0
    fake_df['label'] = 1
    true_df['label'] = 0

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Build a single model input field 
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["input_text"] = (df["title"].astype(str) + "\n\n" + df["text"].astype(str)).map(clean_text)

    # Keep only necessary columns
    df = df[["input_text", "label", "subject", "date"]].copy()

    # Drop empty text rows
    df = df[df["input_text"].str.strip().str.len() > 0].reset_index(drop=True)
    return df

# Produces train, val, test splits
# Stratified by label
def split_isot(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> IsotSplits:
    set_seed(seed)

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    # val is a fraction of the remaining train_val set
    val_fraction_of_train_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        random_state=seed,
        stratify=train_val["label"],
    )

    return IsotSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True)
    )