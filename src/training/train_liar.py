from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict

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

from src.datasets.liar import load_liar_splits, LIAR_LABELS, LABEL2ID, ID2LABEL

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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
    }


def main():
    p = argparse.ArgumentParser()

    # Paths (match your folder casing)
    p.add_argument("--train_tsv", default="data/raw/LIAR/train.tsv")
    p.add_argument("--valid_tsv", default="data/raw/LIAR/valid.tsv")
    p.add_argument("--test_tsv", default="data/raw/LIAR/test.tsv")

    p.add_argument("--model_name", default="roberta-base")
    p.add_argument("--output_dir", default="models/claim_roberta")

    # LIAR statements are short
    p.add_argument("--max_length", type=int, default=128)

    # Optional: speed up iteration by sampling each split
    p.add_argument("--limit", type=int, default=0, help="Limit rows per split (0 = no limit)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)

    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=50)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print("=== Training LIAR Claim Classifier (6-way) ===")
    print(f"Train TSV: {args.train_tsv}")
    print(f"Valid TSV: {args.valid_tsv}")
    print(f"Test TSV:  {args.test_tsv}")
    print(f"Model: {args.model_name}")
    print(f"Output dir: {args.output_dir}")

    # 1) Load splits (already train/val/test)
    splits = load_liar_splits(args.train_tsv, args.valid_tsv, args.test_tsv)

    # Optional limiting (per split)
    if args.limit and args.limit > 0:
        splits = type(splits)(
            train=splits.train.sample(n=min(args.limit, len(splits.train)), random_state=args.seed).reset_index(drop=True),
            val=splits.val.sample(n=min(args.limit, len(splits.val)), random_state=args.seed).reset_index(drop=True),
            test=splits.test.sample(n=min(args.limit, len(splits.test)), random_state=args.seed).reset_index(drop=True),
        )
        print(f"LIMIT enabled -> using up to {args.limit:,} rows per split")

    print("Split sizes:")
    print(f"  Train: {len(splits.train):,}")
    print(f"  Val:   {len(splits.val):,}")
    print(f"  Test:  {len(splits.test):,}")

    print("Train label distribution:")
    print(splits.train["label"].value_counts())

    # 2) Tokenize once (fast)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    print("Tokenizer loaded successfully")
    print(f"Max sequence length: {args.max_length}")

    def encode_texts(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )

    train_enc = encode_texts(splits.train["statement"].tolist())
    val_enc = encode_texts(splits.val["statement"].tolist())
    test_enc = encode_texts(splits.test["statement"].tolist())

    train_labels = torch.tensor(splits.train["label_id"].astype(int).tolist(), dtype=torch.long)
    val_labels = torch.tensor(splits.val["label_id"].astype(int).tolist(), dtype=torch.long)
    test_labels = torch.tensor(splits.test["label_id"].astype(int).tolist(), dtype=torch.long)

    train_ds = EncodedDataset(train_enc, train_labels)
    val_ds = EncodedDataset(val_enc, val_labels)
    test_ds = EncodedDataset(test_enc, test_labels)

    # 3) Model (6-way)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LIAR_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 4) TrainingArguments (match your transformers version + correct metric key)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "_runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        # Your install expects eval_strategy (not evaluation_strategy)
        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",  # must match eval metrics key
        greater_is_better=True,

        logging_steps=args.logging_steps,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # IMPORTANT: no tokenizer=... here (your version rejects it)
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

    test_metrics = trainer.evaluate(test_ds)
    print("Test metrics:", test_metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()