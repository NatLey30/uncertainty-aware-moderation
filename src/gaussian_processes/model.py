from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.likelihoods import BernoulliLikelihood


# =========================
# GP HEAD (single label)
# =========================

class GPBinaryClassifier(ApproximateGP):
    """
    Variational Gaussian Process for binary classification.

    This GP models a latent function f(x) and uses a Bernoulli likelihood
    for classification. It is designed to be used on top of fixed embeddings.
    """

    def __init__(self, input_dim: int, num_inducing: int = 64) -> None:
        inducing_points = torch.randn(num_inducing, input_dim)*0.1

        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing)

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        # Mean and kernel
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

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
    Multilabel classification model using:

    DistilBERT encoder + independent GP head per label.

    Each label is modeled as a binary GP classifier.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        hidden_dim: int = 128,
        num_inducing: int = 64,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()

        # Load transformer encoder (no classification head)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        encoder_dim = self.encoder.config.hidden_size

        # Optional projection (important for GP stability)
        self.projection = nn.Linear(encoder_dim, hidden_dim)

        # One GP + likelihood per label
        self.gp_heads = nn.ModuleList(
            [GPBinaryClassifier(hidden_dim, num_inducing) for _ in range(num_labels)]
        )

        self.likelihoods = nn.ModuleList(
            [BernoulliLikelihood() for _ in range(num_labels)]
        )

        self.num_labels = num_labels

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> List[Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            List of latent GP outputs (one per label)
        """

        # Transformer forward
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token representation
        cls_embedding: Tensor = outputs.last_hidden_state[:, 0]

        # Project to lower dimension
        features: Tensor = self.projection(cls_embedding)

        # Pass through each GP head
        gp_outputs = [
            gp_head(features) for gp_head in self.gp_heads
        ]

        return gp_outputs


# =========================
# BUILDER FUNCTION
# =========================

def build_model_and_tokenizer(
    model_name: str = "distilbert-base-uncased",
    labels: List[str] = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ],
) -> Tuple[DistilBERTWithGP, PreTrainedTokenizerBase]:
    """
    Build tokenizer and GP-based multilabel model.

    Args:
        model_name: HuggingFace model name
        labels: List of label names

    Returns:
        model: DistilBERT + GP heads
        tokenizer: HuggingFace tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = DistilBERTWithGP(
        model_name=model_name,
        num_labels=len(labels),
        hidden_dim=128,
        num_inducing=64,
        freeze_encoder=True,
    )

    return model, tokenizer
