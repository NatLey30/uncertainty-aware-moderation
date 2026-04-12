from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data import load_and_prepare_datasets
from src.utils import load_model


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a saved DistilBERT classification head model."
    )
    parser.add_argument("--model_path", type=str, default="models/finetuning_baseline/best")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def evaluate_finetuning(
    model_path: str = "models/finetuning_baseline/best",
    batch_size: int = 32,
    max_length: int = 128,
    threshold: float = 0.5,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate a saved classification head model on the test split.

    Parameters
    ----------
    model_path : str
        Path to the saved model directory.
    batch_size : int
        Test batch size.
    max_length : int
        Tokenization max length.
    threshold : float
        Probability threshold for predictions.
    seed : int
        Seed to recreate dataset splits.

    Returns
    -------
    Dict[str, float]
        Micro and macro F1 scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(model_path, device=device)

    datasets, _ = load_and_prepare_datasets(
        tokenizer=tokenizer,
        max_length=max_length,
        seed=seed,
    )

    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    model.eval()

    all_probs = []
    all_labels = []

    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(batch["labels"].detach().cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    return metrics


if __name__ == "__main__":
    args = parse_args()

    metrics = evaluate_finetuning(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        threshold=args.threshold,
        seed=args.seed,
    )

    print(metrics)
