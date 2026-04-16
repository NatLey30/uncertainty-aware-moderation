from __future__ import annotations

from typing import List, Sequence, Tuple

import gpytorch
import torch
import torch.nn as nn
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# =========================
# GP HEAD (single label)
# =========================


class WeightedBernoulliLikelihood(BernoulliLikelihood):
    def __init__(self, pos_weight: torch.Tensor):
        """
        pos_weight: tensor de tamaño [num_classes] o escalar
        """
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def expected_log_prob(self, target, function_dist, *args, **kwargs):
        """
        target: shape [batch_size]
        function_dist: distribución variacional q(f)
        """
        # log p(y|f) esperado
        device = "cuda" if torch.cuda.is_available() else "cpu"
        target.to(device)
        self.to(device)
        log_prob = super().expected_log_prob(target, function_dist, *args, **kwargs)

        # weights: w(y) = pos_weight si y=1, 1 si y=0
        weights = target * self.pos_weight + (1 - target)

        return weights * log_prob


class GPBinaryClassifier(ApproximateGP):
    """
    Variational Gaussian Process for binary classification.

    This GP models a latent function f(x) and uses a Bernoulli likelihood
    for classification. It is designed to be used on top of fixed embeddings.
    """

    def __init__(self, input_dim: int, num_inducing: int = 64) -> None:
        inducing_points = torch.randn(num_inducing, input_dim) * 0.1

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        # Mean and kernel
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass of the GP.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            MultivariateNormal distribution over latent function values.
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# =========================
# FULL MODEL
# =========================


class DistilBERTWithGP(nn.Module):
    """
    DistilBERT encoder followed by a linear projection and one GP per label.

    The encoder can be frozen for the classical "fixed embeddings + GP" setup,
    or unfrozen for joint end-to-end fine-tuning.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        hidden_dim: int = 128,
        num_inducing: int = 64,
        freeze_encoder: bool = True,
        pos_weight: list = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.num_inducing = num_inducing
        self.freeze_encoder = freeze_encoder

        # Load transformer encoder (no classification head)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder_dim = int(self.encoder.config.hidden_size)

        # Optional projection (important for GP stability)
        self.projection = nn.Linear(self.encoder_dim, hidden_dim)

        # One GP + likelihood per label
        self.gp_heads = nn.ModuleList(
            [GPBinaryClassifier(hidden_dim, num_inducing) for _ in range(num_labels)],
        )

        if pos_weight is not None:
            self.likelihoods = [
                WeightedBernoulliLikelihood(pos_weight=torch.tensor(w_c))
                for w_c in pos_weight
            ]
        else:
            self.likelihoods = nn.ModuleList(
                [BernoulliLikelihood() for _ in range(num_labels)],
            )

        self.set_encoder_trainable(not freeze_encoder)

    def set_encoder_trainable(self, trainable: bool) -> None:
        """
        Freeze or unfreeze the transformer encoder.

        Args:
            trainable: Whether the encoder parameters should receive gradients.
        """
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable
        self.freeze_encoder = not trainable

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Encode text inputs into projected dense features.

        Args:
            input_ids: Token ids of shape (batch_size, seq_len).
            attention_mask:  Attention mask of shape (batch_size, seq_len).

        Returns:
            Projected features of shape (batch_size, hidden_dim).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # DistilBERT does not expose a pooled output, so we use the first token.
        # Use CLS token representation
        cls_embedding: Tensor = outputs.last_hidden_state[:, 0]

        # Project to lower dimension
        features: Tensor = self.projection(cls_embedding)
        return features

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> List[Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            List of latent GP outputs (one per label)
        """

        features = self.encode(input_ids=input_ids, attention_mask=attention_mask)

        # Pass through each GP head
        gp_outputs = [gp_head(features) for gp_head in self.gp_heads]

        return gp_outputs

    def get_model_config(self) -> dict[str, int | str | bool]:
        """
        Return the minimal configuration needed to rebuild the model.
        """
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "hidden_dim": self.hidden_dim,
            "num_inducing": self.num_inducing,
            "freeze_encoder": self.freeze_encoder,
        }


# =========================
# BUILDER FUNCTION
# =========================


def load_tokenizer(
    model_name: str = "distilbert-base-uncased",
) -> PreTrainedTokenizerBase:
    """
    Build tokenizer and GP-based multilabel model.

    Args:
        model_name: HuggingFace model name

    Returns:
        tokenizer: HuggingFace tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


def build_model(
    model_name: str = "distilbert-base-uncased",
    labels: List[str] = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ],
    hidden_dim: int = 128,
    num_inducing: int = 64,
    freeze_encoder: bool = True,
    pos_weight: list = None,
) -> DistilBERTWithGP:
    """
    Build tokenizer and GP-based multilabel model.

    Args:
        model_name: HuggingFace model name
        labels: List of label names
        hidden_dim: Projection dimension used before the GP heads.
        num_inducing: Number of inducing points per GP.
        freeze_encoder: If True, only the projection and GP components are trained.

    Returns:
        model: DistilBERT + GP heads
    """

    model = DistilBERTWithGP(
        model_name=model_name,
        num_labels=len(labels),
        hidden_dim=hidden_dim,
        num_inducing=num_inducing,
        freeze_encoder=freeze_encoder,
        pos_weight=pos_weight,
    )

    return model
