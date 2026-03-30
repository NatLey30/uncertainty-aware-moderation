import torch
import numpy as np


def predict_with_scores(model, tokenizer, text: str, id2label, device, threshold=0.5, top_k=3):
    """
    Predict multilabel toxicity scores for a single text.
    Returns:
        - active_labels: list of (label, prob) where prob > threshold, sorted by prob desc
        - top_k_labels: top-k labels sorted by prob desc (even if below threshold)
        - raw_probs: list of probabilities per label (index corresponds to id2label)
    """

    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]   # shape: (6,)

    # Convert to list of floats
    probs = probs.tolist()

    # List of (label, probability)
    label_scores = [(label, p) for label, p in zip(id2label, probs)]

    # 1️⃣ Active labels > threshold
    active = [(label, p) for label, p in label_scores if p >= threshold]
    active_sorted = sorted(active, key=lambda x: x[1], reverse=True)

    # 2️⃣ Top-k regardless of threshold
    top_k_sorted = sorted(label_scores, key=lambda x: x[1], reverse=True)[:top_k]

    return {
        "active_labels": active_sorted,
        "top_k": top_k_sorted,
        "raw_probs": label_scores,
    }
