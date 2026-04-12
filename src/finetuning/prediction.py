from __future__ import annotations

import torch
from typing import List, Dict


def predict_with_scores(
    model,
    tokenizer,
    text: str,
    id2label: List[str],
    device,
    threshold: float = 0.5,
    top_k: int = 3,
) -> Dict:
    """
    Predict multilabel probabilities and return structured outputs.
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
        logits = model(**inputs)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    label_scores = list(zip(id2label, probs.tolist()))

    active = [(l, p) for l, p in label_scores if p >= threshold]
    active = sorted(active, key=lambda x: x[1], reverse=True)

    top_k_labels = sorted(label_scores, key=lambda x: x[1], reverse=True)[:top_k]

    return {
        "active_labels": active,
        "top_k": top_k_labels,
        "raw_probs": label_scores,
    }
