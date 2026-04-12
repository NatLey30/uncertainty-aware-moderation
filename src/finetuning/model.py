from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DistilBERTClassifier(nn.Module):
    """
    DistilBERT-based multilabel classifier with a linear classification head.
    """

    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        num_labels: int,
        freeze_encoder: bool = False,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim

        encoder_dim = self.encoder.config.hidden_size

        self.projection = nn.Linear(encoder_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Returns logits and optionally loss.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls = outputs.last_hidden_state[:, 0]
        x = self.projection(cls)
        logits = self.classifier(x)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return logits, loss

        return logits


def build_model(
    model_name: str,
    hidden_dim: int,
    freeze_encoder: bool,
    pos_weight: Optional[torch.Tensor],
    num_labels: int = 6,
) -> DistilBERTClassifier:
    """
    Build classification model.
    """
    return DistilBERTClassifier(
        model_name=model_name,
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        freeze_encoder=freeze_encoder,
        pos_weight=pos_weight,
    )


def load_tokenizer(model_name: str):
    """
    Load tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)