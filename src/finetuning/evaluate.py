import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_and_prepare_datasets, set_global_seed
from src.train_functions import test_step
from src.utils import load_model



def evaluate_model(
    model_path="models/distilbert_toxic",
    batch_size=32,
    max_length=128
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # Load tokenizer and model
    # -----------------------
    model, tokenizer = load_model(model_path, device=device)

    # -----------------------
    # Load dataset
    # -----------------------
    print("Loading dataset...")
    # Must use same splits as training
    tokenized, id2label = load_and_prepare_datasets(
        tokenizer=tokenizer,
        max_length=max_length,
        val_size=0.15,
        seed=42,
        cache_dir=None,
    )

    test_ds = tokenized["test"]
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    print("\n=== TEST RESULTS ===")
    torch.cuda.empty_cache()
    all_preds, all_probs, test_metrics = test_step(model, test_loader, device)
    print(test_metrics)

    # -----------------------
    # Example predictions
    # -----------------------
    print("\n=== Example predictions ===")
    for i in range(10):
        sorted_idx = np.argsort(all_probs[i])[::-1]
        print(f"\nExample #{i}")
        for idx in sorted_idx:
            print(
                f"  {id2label[idx]:15s} — prob={all_probs[i][idx]:.4f} — pred={all_preds[i][idx]}"
            )


if __name__ == "__main__":
    set_global_seed(42)
    evaluate_model()
