from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DistilBERTClassifier(nn.Module):
    """
    DistilBERT-based multilabel classifier with one head per class.
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

        self.model_name = model_name
        self.hidden_dim = hidden_dim

        self.encoder = AutoModel.from_pretrained(model_name)

        encoder_dim = self.encoder.config.hidden_size

        self.projection = nn.Linear(encoder_dim, hidden_dim)

        # 🔹 One classifier per label
        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_labels)]
        )

        self.num_labels = num_labels

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.freeze_encoder = freeze_encoder
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

        Returns:
            logits: (batch_size, num_labels)
            loss (optional)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls = outputs.last_hidden_state[:, 0]  # (B, H)
        x = self.projection(cls)  # (B, hidden_dim)

        # Apply each classifier independently
        logits_list: List[torch.Tensor] = [clf(x) for clf in self.classifiers]

        # Each is (B, 1) → concatenate → (B, num_labels)
        logits = torch.cat(logits_list, dim=1)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return logits, loss

        return logits

    def get_model_config(self) -> dict[str, int | str | bool]:
        """
        Return the minimal configuration needed to rebuild the model.
        """
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "hidden_dim": self.hidden_dim,
            "freeze_encoder": self.freeze_encoder,
        }


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
