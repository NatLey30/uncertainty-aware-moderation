# src/train.py

import argparse
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer = build_model_and_tokenizer()
    print("Loaded model")
    datasets, _ = load_and_prepare_datasets(tokenizer)
    
    print("Loaded dataset")
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size)

    model.to(device)

    # IMPORTANT: only train projection + GP
    params = list(model.projection.parameters())
    for gp in model.gp_heads:
        params += list(gp.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_f1 = 0

    for epoch in tqdm(range(args.epochs)):
        train_metrics = train_one_epoch_gp(model, train_loader, optimizer, device)
        val_metrics = val_step_gp(model, val_loader, device)

        print(f"\nEpoch {epoch}")
        print(train_metrics)
        print(val_metrics)

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            save_model(model, tokenizer, "models/gp_best")


if __name__ == "__main__":
    main()
