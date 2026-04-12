from __future__ import annotations

from typing import Dict

import torch
from sklearn.metrics import f1_score
import numpy as np


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute F1 micro and macro for multilabel classification.
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= threshold).astype(int)
    labels = labels.cpu().numpy()

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer,
    device,
    threshold: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    """
    Train for one epoch.
    """
    model.train()

    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)

        logits, loss = model(input_ids, attention_mask, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    metrics = compute_metrics(logits, labels, threshold)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


def val_step(
    model: torch.nn.Module,
    dataloader,
    device,
    threshold: float,
) -> Dict[str, float]:
    """
    Validation step.
    """
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            logits = model(input_ids, attention_mask)
            all_logits.append(logits)
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    return compute_metrics(logits, labels, threshold)
