import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.data import load_and_prepare_datasets, set_global_seed
from src.gaussian_processes.model import build_model_and_tokenizer
from src.gaussian_processes.train_functions import train_one_epoch_gp, val_step_gp
from src.utils import save_model


def parse_args():
    parser = argparse.ArgumentParser("Train GP model")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/gp_training")

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


def main():
    args = parse_args()

    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = build_model_and_tokenizer()
    print("Loaded model")

    datasets, _ = load_and_prepare_datasets(tokenizer)
    print("Loaded dataset")

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
    )

    model.to(device)
    print("Loaded model")

    model_config = {
        "model_name": "distilbert-base-uncased",
        "num_labels": model.num_labels,
        "hidden_dim": 128,
        "num_inducing": 64,
        "freeze_encoder": True,
    }

    # IMPORTANT: only train projection + GP
    params = list(model.projection.parameters())
    for gp in model.gp_heads:
        params += list(gp.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_f1 = 0.0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_f1_micro": [],
        "train_f1_macro": [],
        "val_f1_micro": [],
        "val_f1_macro": [],
    }

    for epoch in tqdm(range(args.epochs), desc="Training"):
        train_metrics = train_one_epoch_gp(model, train_loader, optimizer, device)
        val_metrics = val_step_gp(model, val_loader, device)

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("Train:", train_metrics)
        print("Val:", val_metrics)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_f1_micro"].append(train_metrics["f1_micro"])
        history["train_f1_macro"].append(train_metrics["f1_macro"])
        history["val_f1_micro"].append(val_metrics["f1_micro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        save_history(history, output_dir)

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            save_model(model, tokenizer, "models/gp_best", model_config=model_config)

    plot_history(history, output_dir)
    print(f"\nSaved metrics to: {output_dir / 'metrics_history.json'}")
    print(f"Saved plots to: {output_dir / 'loss.png'} and {output_dir / 'f1_scores.png'}")
    save_model(model, tokenizer, "models/gp_last", model_config=model_config)


if __name__ == "__main__":
    main()