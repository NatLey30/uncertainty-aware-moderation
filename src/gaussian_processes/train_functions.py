from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import gpytorch
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import Tensor

# from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


def compute_gp_loss(
    gp_outputs: Sequence[gpytorch.distributions.MultivariateNormal],
    gp_heads: Sequence[gpytorch.models.ApproximateGP],
    likelihoods: Sequence[gpytorch.likelihoods.BernoulliLikelihood],
    labels: Tensor,
    num_data: int,
) -> Tensor:
    """
    Compute the summed variational ELBO across all label-specific GPs.

    Args:
        gp_outputs: Latent GP outputs for the current batch.
        gp_heads: GP modules, one per label.
        likelihoods: Bernoulli likelihoods, one per label.
        labels: Multi-label targets of shape ``(batch_size, num_labels)``.
        num_data: Total number of training examples, required by the variational ELBO.

    Returns:
        Scalar loss summed across labels.
    """

    total_loss = 0.0

    for i, (gp_output, gp_head, likelihood) in enumerate(
        zip(gp_outputs, gp_heads, likelihoods)
    ):
        target = labels[:, i].float()

        mll = gpytorch.mlls.VariationalELBO(
            likelihood,
            gp_head,
            num_data=num_data,
        )

        loss = -mll(gp_output, target)
        total_loss += loss

    return total_loss


@torch.no_grad()
def predict_from_gp(
    gp_outputs: Sequence[gpytorch.distributions.MultivariateNormal],
    likelihoods: Sequence[gpytorch.likelihoods.BernoulliLikelihood],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert latent GP outputs into probabilities and binary predictions.

    Args:
        gp_outputs: Latent GP outputs.
        likelihoods: Label-specific Bernoulli likelihoods.
        threshold: Decision threshold applied to probabilities.

    Returns:
        Probabilities and hard predictions (batch_size, num_labels)
    """

    probs: List[np.ndarray] = []

    for gp_output, likelihood in zip(gp_outputs, likelihoods):
        pred_dist = likelihood(gp_output)
        pred_mean = pred_dist.mean
        probs.append(pred_mean.detach().cpu().numpy())

    probs = np.stack(probs, axis=1)
    preds = (probs > threshold).astype(int)

    return probs, preds


def _move_batch_to_device(
    batch: Dict[str, Tensor], device: torch.device
) -> Dict[str, Tensor]:
    """
    Move every tensor in a batch dictionary to the requested device.
    """
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _compute_epoch_f1(
    all_predictions: List[np.ndarray],
    all_labels: List[np.ndarray],
) -> Tuple[float, float]:
    """
    Compute micro and macro F1 for one epoch.
    """
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    return float(f1_micro), float(f1_macro)


def train_one_epoch_gp(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float = 0.5,
    max_grad_norm: float | None = None,
) -> Dict[str, float]:
    """
    Train the GP model for one epoch.

    Parameters:
        model: DistilBERT + GP model.
        dataloader: Training dataloader.
        optimizer: Optimizer over trainable parameters.
        device: Target device.
        threshold: Threshold used to convert probabilities into hard predictions.
        max_grad_norm: If provided, gradients are clipped to this norm.

    Returns:
        Training loss and F1 scores.
    """

    model.train()

    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad()

        gp_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = compute_gp_loss(
            gp_outputs,
            model.gp_heads,
            model.likelihoods,
            batch["labels"],
            num_data=len(dataloader.dataset),
        )

        loss.backward()

        # if max_grad_norm is not None:
        #     clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()

        _, preds = predict_from_gp(
            gp_outputs=gp_outputs,
            likelihoods=model.likelihoods,
            threshold=threshold,
        )

        all_preds.append(preds)
        all_labels.append(batch["labels"].cpu().numpy())

    f1_micro, f1_macro = _compute_epoch_f1(all_preds, all_labels)

    return {
        "loss": total_loss / len(dataloader),
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }


@torch.no_grad()
def val_step_gp(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate the GP model for one epoch.

    Parameters:
        model: DistilBERT + GP model.
        dataloader: Validation or test dataloader.
        device: Target device.
        threshold: Threshold used to convert probabilities into hard predictions.

    Returns:
        Validation F1 scores.
    """
    model.eval()

    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)

        gp_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        _, preds = predict_from_gp(
            gp_outputs=gp_outputs,
            likelihoods=model.likelihoods,
            threshold=threshold,
        )

        all_preds.append(preds)
        all_labels.append(batch["labels"].cpu().numpy())

    f1_micro, f1_macro = _compute_epoch_f1(all_preds, all_labels)
    return {"f1_micro": f1_micro, "f1_macro": f1_macro}
