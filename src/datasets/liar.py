"""
Read raw files from LIAR dataset
Create text field 
Encode labels 
Data is already split into train/val/test
Save to data/processed
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.common.text_cleaning import clean_text

LIAR_COLUMNS = [
    "id", "label", "statement", "subjects", "speaker", "job",
    "state", "party", "barely_true", "false", "half_true", "mostly_true",
    "pants_on_fire", "context"
]

LIAR_LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
LABEL2ID = {label: i for i, label in enumerate(LIAR_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LIAR_LABELS)}

# Map LIAR's 6 labels to integer class ids 0..5
def encode_liar_label(label: str) -> int:
    label = str(label).strip()
    if label not in LIAR_LABELS:
        raise ValueError(f"Unknown LIAR label: {label}")
    return LABEL2ID[label]

@dataclass(frozen=True)
class LiarSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

def _load_one(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)

    # Clean and keep non-empty statements
    df["statement"] = df["statement"].fillna("").map(clean_text)
    df = df[df["statement"].str.strip().str.len() > 0].copy()

    # Encode labels
    df["label_id"] = df["label"].map(encode_liar_label)

    # Keep the fields we need
    return df[["statement", "label", "label_id", "speaker", "party", "context"]].reset_index(drop=True)

def load_liar_splits(
    train_tsv: str | Path,
    val_tsv: str | Path,
    test_tsv: str | Path,
) -> LiarSplits:
    return LiarSplits(
        train=_load_one(train_tsv),
        val=_load_one(val_tsv),
        test=_load_one(test_tsv),
    )