"""
Inference script for multilabel classification with Gaussian Processes.

This script:
- Loads a trained DistilBERT + GP model
- Computes predictive probabilities per label
- Extracts uncertainty (variance / std)
- Returns structured predictions

Uncertainty is derived from the predictive distribution of each GP head.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from src.utils import load_model


class PredictionOutput(TypedDict):
    """
    Structured prediction output returned by the inference helper.
    """

    active_labels: List[tuple[str, float, float]]
    top_k: List[tuple[str, float, float]]
    all_scores: List[tuple[str, float, float]]


@torch.no_grad()
def predict_with_uncertainty(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    id2label: List[str],
    device: torch.device,
    max_length: int = 128,
    threshold: float = 0.5,
    top_k: int = 3,
) -> Dict:
    """
    Predict multilabel outputs with uncertainty using GP heads.

    Parameters
    ----------
    model : torch.nn.Module
        Trained DistilBERT + GP model.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer used during training.
    text : str
        Input text.
    id2label : List[str]
        List mapping label indices to label names.
    device : torch.device
        CPU or CUDA device.
    max_length : int
        Maximum token length.
    threshold : float
        Threshold for active labels.
    top_k : int
        Number of top labels to return.

    Returns
    -------
    Dict
        {
            "active_labels": [(label, prob, std)],
            "top_k": [(label, prob, std)],
            "all_scores": [(label, prob, std)],
        }
    """

    model.eval()

    encoded_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}

    # Forward pass
    gp_outputs = model(
        input_ids=encoded_inputs["input_ids"],
        attention_mask=encoded_inputs["attention_mask"],
    )

    probs: List[float] = []
    variances: List[float] = []

    # Process each GP head
    for gp_output, likelihood in zip(gp_outputs, model.likelihoods):
        pred_dist = likelihood(gp_output)

        # Mean probability
        mean = pred_dist.mean.squeeze().cpu().item()

        # Variance (uncertainty)
        var = pred_dist.variance.squeeze().cpu().item()

        probs.append(mean)
        variances.append(var)

    probs = np.array(probs)
    variances = np.array(variances)
    stds = np.sqrt(variances)

    # Build label tuples
    label_info = [
        (label, float(p), float(s))
        for label, p, s in zip(id2label, probs, stds)
    ]

    # Active labels (above threshold)
    active = [
        (label, p, s)
        for (label, p, s) in label_info
        if p >= threshold
    ]
    active_sorted = sorted(active, key=lambda x: x[1], reverse=True)

    # Top-k labels
    top_k_sorted = sorted(label_info, key=lambda x: x[1], reverse=True)[:top_k]

    return {
        "active_labels": active_sorted,
        "top_k": top_k_sorted,
        "all_scores": label_info,
    }


def load_and_predict(
    model_path: str,
    text: str,
    id2label: List[str],
) -> Dict:
    """
    Convenience wrapper to load model and run prediction.

    Args:
        model_path: Path to saved model.
        text: Input text.
        id2label: Label names.

    Returns:
        Prediction with uncertainty.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(model_path, device=device)

    return predict_with_uncertainty(
        model=model,
        tokenizer=tokenizer,
        text=text,
        id2label=id2label,
        device=device,
    )


if __name__ == "__main__":
    # Example usage

    id2label = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    text = "You are the worst person ever."

    result = load_and_predict(
        model_path="models/gp_best",
        text=text,
        id2label=id2label,
    )
    
    print(result)
    print("\n=== Prediction with Uncertainty ===\n")

    for label, prob, std in result["all_scores"]:
        print(f"{label:15s} | prob={prob:.4f} | std={std:.4f}")
