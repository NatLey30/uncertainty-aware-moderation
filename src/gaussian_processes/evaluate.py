# src/evaluate.py

import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.data import load_and_prepare_datasets
from src.utils import load_model
from src.gaussian_processes.train_functions import predict_from_gp


def evaluate_gp(model_path="models/gp_best"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(model_path, device=device)

    datasets, id2label = load_and_prepare_datasets(tokenizer)

    test_loader = DataLoader(datasets["test"], batch_size=32)

    model.eval()

    all_preds, all_labels = [], []

    for batch in test_loader:
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

    print({
        "f1_micro": f1_score(all_labels, all_preds, average="micro"),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    })
