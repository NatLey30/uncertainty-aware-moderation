from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import load_and_prepare_datasets, set_global_seed
from src.gaussian_processes.model import build_model_and_tokenizer
from src.gaussian_processes.train_functions import train_one_epoch_gp, val_step_gp
from src.utils import save_model


VERSION = "gp"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for GP training.
    """
    parser = argparse.ArgumentParser(
        description="Train DistilBERT + Gaussian Processes for multilabel classification.",
    )

    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_inducing", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_encoder", type=float, default=2e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="If set, only projection + GP heads + likelihoods are trained.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"outputs/{VERSION}",
        help="Directory used to store metrics and plots.",
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default=f"models/{VERSION}/best",
        help="Directory used to save the best checkpoint.",
    )
    parser.add_argument(
        "--last_model_dir",
        type=str,
        default=f"models/{VERSION}/last",
        help="Directory used to save the last checkpoint.",
    )

    return parser.parse_args()


def save_history(history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "metrics_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def plot_history(history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]

    # Plot 1: loss
    if len(history["train_loss"]) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_loss"], marker="o", label="train_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss.png")
        plt.close()

    # Plot 2: F1
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_f1_micro"], marker="o", label="train_f1_micro")
    plt.plot(epochs, history["train_f1_macro"], marker="o", label="train_f1_macro")
    plt.plot(epochs, history["val_f1_micro"], marker="o", label="val_f1_micro")
    plt.plot(epochs, history["val_f1_macro"], marker="o", label="val_f1_macro")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Train/Validation F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "f1_scores.png")
    plt.close()


def build_optimizer(
    model: torch.nn.Module,
    lr_head: float,
    lr_encoder: float,
    weight_decay: float,
    freeze_encoder: bool,
) -> torch.optim.Optimizer:
    """
    Build an optimizer with separate learning rates for encoder and GP head.

    Parameters:
        model: DistilBERT + GP model.
        lr_head: Learning rate for projection, GP heads, and likelihoods.
        lr_encoder: Learning rate for the transformer encoder.
        weight_decay: Weight decay used by AdamW.
        freeze_encoder: Whether the encoder is frozen.

    Returns:
        Configured optimizer.
    """
    parameter_groups: List[Dict[str, Any]] = []

    if not freeze_encoder:
        encoder_parameters = [
            parameter
            for parameter in model.encoder.parameters()
            if parameter.requires_grad
        ]

        if len(encoder_parameters) > 0:
            parameter_groups.append(
                {
                    "params": encoder_parameters,
                    "lr": lr_encoder,
                    "weight_decay": weight_decay,
                }
            )

    head_parameters: List[torch.nn.Parameter] = []
    head_parameters.extend(
        parameter
        for parameter in model.projection.parameters()
        if parameter.requires_grad
    )

    for gp_head in model.gp_heads:
        head_parameters.extend(
            parameter
            for parameter in gp_head.parameters()
            if parameter.requires_grad
        )

    for likelihood in model.likelihoods:
        head_parameters.extend(
            parameter
            for parameter in likelihood.parameters()
            if parameter.requires_grad
        )

    if len(head_parameters) == 0:
        raise ValueError("No trainable head parameters were found.")

    parameter_groups.append(
        {
            "params": head_parameters,
            "lr": lr_head,
            "weight_decay": weight_decay,
        }
    )

    return torch.optim.AdamW(parameter_groups)


def main():
    """
    Train the GP-based model.

    Two setups are supported:
    - frozen encoder + GP head
    - joint encoder fine-tuning + GP head
    """
    args = parse_args()

    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model, tokenizer = build_model_and_tokenizer(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        num_inducing=args.num_inducing,
        freeze_encoder=args.freeze_encoder,
    )
    model.to(device)
    print("[INFO] Model loaded.")

    datasets, _ = load_and_prepare_datasets(
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print("[INFO] Dataset loaded.")

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = build_optimizer(
        model=model,
        lr_head=args.lr_head,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        freeze_encoder=args.freeze_encoder,
    )

    model_config = model.get_model_config()

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_f1_micro": [],
        "train_f1_macro": [],
        "val_f1_micro": [],
        "val_f1_macro": [],
    }

    best_f1_macro = float("-inf")

    epoch_pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in epoch_pbar:
        train_metrics = train_one_epoch_gp(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            threshold=args.threshold,
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics = val_step_gp(
            model=model,
            dataloader=val_loader,
            device=device,
            threshold=args.threshold,
        )

        # Update progress bar description with current metrics
        epoch_pbar.set_description(
            # f"Epoch {epoch}/{args.num_epochs} | "
            f"Train loss: {train_metrics['loss']:.4f}, "
            f"Train F1-micro: {train_metrics['f1_micro']:.4f}, "
            f"Train F1-macro: {train_metrics['f1_macro']:.4f} | "
            # f"Val loss: {val_metrics['loss']:.4f}, "
            f"Val F1-micro: {val_metrics['f1_micro']:.4f}, "
            f"Val F1-macro: {val_metrics['f1_macro']:.4f}"
        )

        # print(f"\nEpoch {epoch + 1}/{args.epochs}")
        # print("Train:", train_metrics)
        # print("Val:", val_metrics)

        history["epoch"].append(float(epoch + 1))
        history["train_loss"].append(train_metrics["loss"])
        history["train_f1_micro"].append(train_metrics["f1_micro"])
        history["train_f1_macro"].append(train_metrics["f1_macro"])
        history["val_f1_micro"].append(val_metrics["f1_micro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        save_history(history, output_dir)

        if val_metrics["f1_macro"] > best_f1_macro:
            best_f1_macro = val_metrics["f1_macro"]
            save_model(
                model=model,
                tokenizer=tokenizer,
                save_dir=args.best_model_dir,
                model_config=model_config,
            )

    plot_history(history, output_dir)
    save_model(
        model=model,
        tokenizer=tokenizer,
        save_dir=args.last_model_dir,
        model_config=model_config,
    )

    print(f"[INFO] History saved to: {output_dir / 'metrics_history.json'}")
    print(f"[INFO] Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()