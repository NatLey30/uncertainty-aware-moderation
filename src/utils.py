import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.finetuning.model import DistilBERTClassifier
from src.gaussian_processes.model import DistilBERTWithGP


def save_model(model, tokenizer, save_dir: str, model_config: dict | None = None):
    """
    Guarda un modelo PyTorch custom + tokenizer.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Saving model to: {save_dir}")

    # pesos
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    # tokenizer
    tokenizer.save_pretrained(save_dir)

    # config mínima para reconstruir el modelo
    if model_config is not None:
        with open(
            os.path.join(save_dir, "model_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(model_config, f, indent=2)

    print("[INFO] Model saved successfully.")


def load_model(save_dir: str, device: torch.device):
    """
    Carga el modelo custom DistilBERTWithGP + tokenizer.
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"[ERROR] Directory not found: {save_dir}")

    print(f"[INFO] Loading model from: {save_dir}")

    config_path = os.path.join(save_dir, "model_config.json")
    weights_path = os.path.join(save_dir, "pytorch_model.bin")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Missing config file: {config_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[ERROR] Missing weights file: {weights_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    model = DistilBERTWithGP(
        model_name=cfg["model_name"],
        num_labels=cfg["num_labels"],
        hidden_dim=cfg["hidden_dim"],
        num_inducing=cfg["num_inducing"],
        freeze_encoder=cfg["freeze_encoder"],
    )

    print(weights_path)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)

    print("[INFO] Model loaded successfully.")
    return model, tokenizer


def load_finetuning_model(save_dir: str, device: torch.device):
    """
    Load a saved DistilBERT classification head model + tokenizer.

    Expects:
        - pytorch_model.bin
        - model_config.json
        - tokenizer files

    Returns:
        model, tokenizer
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"[ERROR] Directory not found: {save_dir}")

    print(f"[INFO] Loading finetuning model from: {save_dir}")

    config_path = os.path.join(save_dir, "model_config.json")
    weights_path = os.path.join(save_dir, "pytorch_model.bin")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Missing config file: {config_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[ERROR] Missing weights file: {weights_path}")

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    # Model reconstruction
    model = DistilBERTClassifier(
        model_name=cfg["model_name"],
        hidden_dim=cfg["hidden_dim"],
        num_labels=cfg["num_labels"],
        freeze_encoder=cfg["freeze_encoder"],
        pos_weight=None,
    )

    # Load weights
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("[INFO] Finetuning model loaded successfully.")

    return model, tokenizer


def load_model_weights(model, weights_path: str, device):
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"[INFO] Weights loaded from {weights_path}")
    return model


def load_model_weights(model, weights_path: str, device):
    """
    Loads only the model weights (state_dict).

    Useful when loading .pt or .bin trained weights.
    """
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"[INFO] Weights loaded from {weights_path}")
    return model
