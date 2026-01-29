"""
Tokenization.
Load splits fron datasets/isot.py
Apply tokenizer.
"""

from __future__ import annotations 

import argparse
import os 
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from src.datasets.isot import load_isot, split_isot

"""
Turns DataFrame rows into training items
Each item returns a dict 
{
    'input_ids': tensor(...),
    'attention_mask': tensor(...),
    'labels': tensor(label)
"""
@dataclass
class EncodedDataset(torch.utils.data.Dataset):
    encodings: Dict[str, torch.Tensor]
    labels: torch.Tensor

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# Takes predicted logits -> argmax to choose class -> compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "macro_precision": precision, "macro_recall": recall, "macro_f1": f1}

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--fake_csv", default="data/raw/ISOT/Fake.csv")
    p.add_argument("--true_csv", default="data/raw/ISOT/True.csv")
    p.add_argument("--model_name", default="roberta-base")
    p.add_argument("--output_dir", default="models/article_roberta")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="Limit total rows (0 = no limit)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)

    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--save_steps", type=int, default=250)
    p.add_argument("--logging_steps", type=int, default=50)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    print("=== Training ISOT Article Classifier ===")
    print(f"Fake CSV: {args.fake_csv}")
    print(f"True CSV: {args.true_csv}")
    print(f"Model: {args.model_name}")
    print(f"Output dir: {args.output_dir}")

    # Load, clean, split data
    df = load_isot(args.fake_csv, args.true_csv)
    if args.limit and args.limit > 0:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
        print(f"LIMIT enabled -> using {len(df):,} rows")
    print(f"Loaded ISOT dataset: {len(df):,} rows")
    print("Label distribution:")
    print(df["label"].value_counts())
    splits = split_isot(df, seed=args.seed, test_size=0.1, val_size=0.1)
    print("Dataset splits:")
    print(f"  Train: {len(splits.train):,}")
    print(f"  Val:   {len(splits.val):,}")
    print(f"  Test:  {len(splits.test):,}")

    # Tokenizer + datasets
    # Converts raw text to input_ids, attention_mask tensors
    # Cut off text at max_length, pad if shorter
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    print("Tokenizer loaded successfully")
    print(f"Max sequence length: {args.max_length}")

    def encode_texts(tokenizer, texts, max_length):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    # Turn DataFrame rows into training examples
    train_enc = encode_texts(tokenizer, splits.train["input_text"].tolist(), args.max_length)
    val_enc   = encode_texts(tokenizer, splits.val["input_text"].tolist(), args.max_length)
    test_enc  = encode_texts(tokenizer, splits.test["input_text"].tolist(), args.max_length)

    train_ds = EncodedDataset(train_enc, torch.tensor(splits.train["label"].astype(int).tolist()))
    val_ds   = EncodedDataset(val_enc,   torch.tensor(splits.val["label"].astype(int).tolist()))
    test_ds  = EncodedDataset(test_enc,  torch.tensor(splits.test["label"].astype(int).tolist()))

    # Model - True and Fake labels
    id2label = {0: "true", 1: "fake"}
    label2id = {"true": 0, "fake": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # Train
    # Evaluate every eval_steps, save every save_steps
    # Keep only last 2 checkpoints
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "_runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        report_to="none",
        fp16=torch.cuda.is_available(),  # safe default
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    print("Starting training...")
    trainer.train()
    print("Training finished")

    # Final test eval
    test_metrics = trainer.evaluate(test_ds)
    print("Test metrics:", test_metrics)

    # Save best model + Tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()