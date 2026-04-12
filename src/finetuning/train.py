from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.data import load_and_prepare_datasets, set_global_seed
from src.finetuning.model import build_model, load_tokenizer
from src.finetuning.train_functions import train_one_epoch, val_step
from src.utils import save_model


VERSION = "finetuning_baseline"


def parse_args() -> argparse.Namespace:
    """
    Parse arguments for baseline training.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_encoder", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--use_weights", action="store_true")

    parser.add_argument("--output_dir", type=str, default=f"outputs/{VERSION}")
    parser.add_argument("--best_model_dir", type=str, default=f"models/{VERSION}/best")
    parser.add_argument("--last_model_dir", type=str, default=f"models/{VERSION}/last")

    return parser.parse_args()


def compute_pos_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights.
    """
    weights = []
    for c in range(labels.shape[1]):
        pos = labels[:, c].sum()
        neg = len(labels) - pos
        weights.append(neg / (pos + 1e-6))
    return torch.tensor(weights, dtype=torch.float32)


def main():
    args = parse_args()
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = load_tokenizer(args.model_name)

    datasets, _ = load_and_prepare_datasets(
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    if args.use_weights:
        labels = np.array(datasets["train"]["labels"])
        pos_weights = compute_pos_weights(labels)
    else:
        pos_weights = None

    model = build_model(
        args.model_name,
        args.hidden_dim,
        args.freeze_encoder,
        pos_weights,
    ).to(device)

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr_encoder},
        {"params": model.classifier.parameters(), "lr": args.lr_head},
    ])

    history = {"epoch": [], "train_loss": [], "val_f1_macro": []}

    best = -1.0

    for epoch in tqdm(range(args.epochs)):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            args.threshold, args.max_grad_norm
        )

        val_metrics = val_step(model, val_loader, device, args.threshold)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        if val_metrics["f1_macro"] > best:
            best = val_metrics["f1_macro"]
            save_model(model, tokenizer, args.best_model_dir)

    save_model(model, tokenizer, args.last_model_dir)

    with open(Path(args.output_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
