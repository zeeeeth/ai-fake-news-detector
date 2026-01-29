from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Container to store everything needed to run inference for one model
@dataclass
class ModelBundle:
    name: str                 # Model name
    path: str                 # Model path
    tokenizer: Any            # Tokenizer
    model: Any                # Model 
    id2label: Dict[int, str]  # Mapping from label IDs to label names e.g. {0: "REAL", 1: "FAKE"}
    device: torch.device      # Device the model is on

# Loads model and tokenizer from model_dir
# Moves model to GPU if available, puts model in inference mode so dropout is disabled
def _load_bundle(name: str, model_dir: str, device: torch.device) -> ModelBundle:
    model_dir = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Label mapping extraction, transformers store them in model.config.id2label
    cfg = getattr(model, "config", None)
    id2label = {}
    if cfg is not None and getattr(cfg, "id2label", None):
        # Sometimes keys are strings, normalise to int
        for k, v in cfg.id2label.items():
            try:
                id2label[int(k)] = str(v)
            except Exception:
                pass

    return ModelBundle(
        name=name,
        path=model_dir,
        tokenizer=tokenizer,
        model=model,
        id2label=id2label,
        device=device,
    )

# Model outputs logits, convert to probabilities summing to 1 via softmax
def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)

# Infer on a single text input using the given model bundle
def _predict_text(
    bundle: ModelBundle,
    text: str,
    max_length: int,
) -> Dict[str, Any]:
    # Throw error on empty input
    if text is None or not str(text).strip():
        raise ValueError("Empty input text")

    # Converts text -> token IDs + attention masks
    inputs = bundle.tokenizer(
        str(text),
        truncation=True,        # Truncate down to max_length tokens if too long
        padding=True,           # Pad to the required length
        max_length=max_length,  # Max length for this model
        return_tensors="pt",    # Return PyTorch tensors
    )
    inputs = {k: v.to(bundle.device) for k, v in inputs.items()}

    # Model forward pass
    with torch.no_grad():  # Disable gradient calculation for inference since not training
        outputs = bundle.model(**inputs)
        logits = outputs.logits.squeeze(0)  # Change shape from [1, num_labels] to [num_labels]
        probs = _softmax_probs(logits).detach().cpu()

    probs_list = probs.tolist()
    pred_id = int(torch.argmax(probs).item())               # Class with highest probability
    pred_label = bundle.id2label.get(pred_id, str(pred_id)) # Map id to label name
    confidence = float(probs_list[pred_id])                 # Probability of the predicted class

    probabilities = {
        bundle.id2label.get(i, str(i)): float(p)
        for i, p in enumerate(probs_list)
    }

    return {
        "label": pred_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }

# Loads models once and serves predictions
class Predictor:

    def __init__(
        self,
        article_model_dir: str = "models/article_roberta",
        claim_model_dir: str = "models/claim_roberta",
        device: Optional[str] = None,
    ):
        # Determine device to run models on
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # Load both models on startup, do not reload for each prediction
        self.article = _load_bundle("article", article_model_dir, self.device)
        self.claim = _load_bundle("claim", claim_model_dir, self.device)

    def predict_article(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        return _predict_text(self.article, text=text, max_length=max_length)

    def predict_claim(self, statement: str, max_length: int = 128) -> Dict[str, Any]:
        return _predict_text(self.claim, text=statement, max_length=max_length)
    # Returned by /api/models endpoint
    def info(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "article_model": {
                "path": self.article.path,
                "labels": [self.article.id2label[i] for i in sorted(self.article.id2label)],
            },
            "claim_model": {
                "path": self.claim.path,
                "labels": [self.claim.id2label[i] for i in sorted(self.claim.id2label)],
            },
        }
