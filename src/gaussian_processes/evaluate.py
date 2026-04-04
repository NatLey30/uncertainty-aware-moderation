from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data import load_and_prepare_datasets
from src.gaussian_processes.train_functions import predict_from_gp
from src.utils import load_model


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate a saved DistilBERT + GP model.")
    parser.add_argument("--model_path", type=str, default="models/gp_best")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def evaluate_gp(
    model_path: str = "models/gp_best",
    batch_size: int = 32,
    max_length: int = 128,
    threshold: float = 0.5,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate a saved GP model on the test split.

    Parameters
    ----------
    model_path : str, default="models/gp_best"
        Path to the saved model directory.
    batch_size : int, default=32
        Test batch size.
    max_length : int, default=128
        Tokenization max length.
    threshold : float, default=0.5
        Probability threshold used to compute hard predictions.
    seed : int, default=42
        Seed used when recreating the splits.

    Returns
    -------
    Dict[str, float]
        Micro and macro F1.
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

    all_predictions = []
    all_labels = []

    for batch in test_loader:
        batch = {key: value.to(device) for key, value in batch.items()}

        gp_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        _, predictions = predict_from_gp(
            gp_outputs=gp_outputs,
            likelihoods=model.likelihoods,
            threshold=threshold,
        )

        all_predictions.append(predictions)
        all_labels.append(batch["labels"].detach().cpu().numpy())

    y_pred = np.concatenate(all_predictions, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    metrics = {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics


if __name__ == "__main__":
    arguments = parse_args()
    metrics = evaluate_gp(
        model_path=arguments.model_path,
        batch_size=arguments.batch_size,
        max_length=arguments.max_length,
        threshold=arguments.threshold,
        seed=arguments.seed,
    )
    print(metrics)
