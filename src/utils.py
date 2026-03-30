import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def save_model(model, tokenizer, save_dir: str):
    """
    Saves a HuggingFace model and its tokenizer to a directory.

    Args:
        model: The trained HuggingFace model (nn.Module).
        tokenizer: The tokenizer associated with the model.
        save_dir: Directory where the model will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Saving model to: {save_dir}")

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("[INFO] Model saved successfully.")


def load_model(save_dir: str, device: torch.device):
    """
    Loads a HuggingFace model and tokenizer from a directory.

    Args:
        save_dir: Directory where the model is stored.
        device: The device ('cpu' or 'cuda').

    Returns:
        (model, tokenizer)
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"[ERROR] Directory not found: {save_dir}")

    print(f"[INFO] Loading model from: {save_dir}")

    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)

    model.to(device)

    print("[INFO] Model loaded successfully.")

    return model, tokenizer


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
