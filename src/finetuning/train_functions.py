from typing import Dict

import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
) -> float:
    """
    Runs one epoch of training.

    Args:
        model: The model to train.
        dataloader: DataLoader for the training set.
        optimizer: Optimizer used for weight updates.
        device: Device where tensors and the model should be loaded.
        scheduler: Optional learning rate scheduler.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # Predictions
        preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    avg_loss = total_loss / len(dataloader)
    f1_micro = f1_score(all_labels, all_preds, average="micro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates a model on a validation or test DataLoader.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for evaluation.
        device: Device where tensors and the model should be loaded.

    Returns:
        A dictionary with evaluation metrics:
            - loss
            - accuracy
            - f1
    """
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()

        all_preds.extend(list(preds))
        all_labels.extend(list(labels))

    avg_loss = total_loss / len(dataloader)
    f1_micro = f1_score(all_labels, all_preds, average="micro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }


def test_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Test step function.
    Semantically identical to `evaluate`, but separated for clarity.
    Useful for adding extra test-only metrics in the future.
    """
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits

        logits_cpu = logits.detach().cpu().numpy()
        probs = 1 / (1 + np.exp(-logits_cpu))
        preds = (probs > 0.5).astype(int)

        labels = batch["labels"].detach().cpu().numpy()

        all_probs.extend(list(logits_cpu))
        all_preds.extend(list(preds))
        all_labels.extend(list(labels))

    f1_micro = f1_score(all_labels, all_preds, average="micro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return all_preds, all_probs, {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }
