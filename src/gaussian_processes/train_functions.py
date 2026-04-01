# src/train_functions.py

from typing import Dict, List

import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import gpytorch


def compute_gp_loss(
    gp_outputs,
    gp_heads,
    likelihoods,
    labels,
    num_data: int,
):
    """
    Compute ELBO loss for multilabel GP.
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


def predict_from_gp(
    gp_outputs,
    likelihoods,
):
    """
    Convert GP outputs into probabilities.
    """

    probs = []

    for gp_output, likelihood in zip(gp_outputs, likelihoods):
        pred_dist = likelihood(gp_output)
        pred_dist = likelihood(gp_output)
        pred_mean = pred_dist.mean
        # sigmoid = torch.sigmoid(pred_mean)
        probs.append(pred_mean.detach().cpu().numpy())

    probs = np.stack(probs, axis=1)
    preds = (probs > 0.5).astype(int)

    return probs, preds


def train_one_epoch_gp(
    model,
    dataloader: DataLoader,
    optimizer,
    device,
):
    """
    Train one epoch for GP model.
    """

    model.train()

    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}

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
        optimizer.step()
        

        total_loss += loss.item()

        probs, preds = predict_from_gp(gp_outputs, model.likelihoods)

        all_preds.append(preds)
        all_labels.append(batch["labels"].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        "loss": total_loss / len(dataloader),
        "f1_micro": f1_score(all_labels, all_preds, average="micro"),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }


@torch.no_grad()
def val_step_gp(model, dataloader, device):
    model.eval()

    all_preds, all_labels = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        gp_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        probs, preds = predict_from_gp(gp_outputs, model.likelihoods)

        all_preds.append(preds)
        all_labels.append(batch["labels"].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        "f1_micro": f1_score(all_labels, all_preds, average="micro"),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }
