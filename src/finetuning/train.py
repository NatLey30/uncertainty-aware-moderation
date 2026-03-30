# src/train.py

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data import load_and_prepare_datasets, set_global_seed
from src.model import build_model_and_tokenizer
from src.train_functions import train_one_epoch, val_step, test_step
from src.utils import save_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train toxicity classifier (DistilBERT)")

    # Model and data
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--cache_dir", type=str, default=None)

    # Training settings
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="models/distilbert_toxic")
    parser.add_argument("--num_workers", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()

    #  Set all random seeds for reproducibility
    set_global_seed(args.seed)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    id2label = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    # Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(
        model_name=args.model_name,
        num_labels=len(id2label),
        id2label=id2label,
    )

    # Load dataset
    datasets_tokenized, _ = load_and_prepare_datasets(
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    # DataLoaders
    train_loader = DataLoader(
        datasets_tokenized["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        datasets_tokenized["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        datasets_tokenized["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Optimizer & learning scheduler
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0

    # === Training Loop ===
    epoch_pbar = tqdm(range(1, args.num_epochs + 1), desc="Training epochs")

    for epoch in epoch_pbar:
        # Train one epoch → returns loss, acc, f1
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
        )

        # Validation
        val_metrics = val_step(model, val_loader, device)

        # Update progress bar description with current metrics
        epoch_pbar.set_description(
            f"Epoch {epoch}/{args.num_epochs} | "
            f"Train loss: {train_metrics['loss']:.4f}, "
            f"Train F1-micro: {train_metrics['f1_micro']:.4f}, "
            f"Train F1-macro: {train_metrics['f1_macro']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f}, "
            f"Val F1-micro: {val_metrics['f1_micro']:.4f}, "
            f"Val F1-macro: {val_metrics['f1_macro']:.4f}"
        )

        # Track best based on F1_macro
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            print(f"\nNew best model (F1-macro={best_val_f1:.4f}). Saving to best_model.")
            save_model(model, tokenizer, "models/best_model")

    print(f"\nNew model. Saving to {args.output_dir}".)
    save_model(model, tokenizer, args.output_dir)

    # === Final Test Evaluation ===
    print("\n=== Evaluating test set using best saved model ===")
    model.to(device)
    _, _, test_metrics = test_step(model, test_loader, device)

    print(test_metrics)


if __name__ == "__main__":
    main()
