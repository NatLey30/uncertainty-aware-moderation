import argparse
import random
from typing import Tuple
import os
import zipfile
import subprocess
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import numpy as np
import torch


def unzip_all_recursively(directory: str):
    """
    Unzips all .zip files found inside `directory`, including nested zips.
    After unzipping a zip file, it deletes the zip to avoid reprocessing.
    """
    extracted_something = True

    while extracted_something:
        extracted_something = False
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith(".zip"):
                    zip_path = os.path.join(root, f)
                    print(f"Extracting: {zip_path}")

                    try:
                        with zipfile.ZipFile(zip_path, "r") as z:
                            z.extractall(root)
                        os.remove(zip_path)  # delete after extraction
                        extracted_something = True
                    except zipfile.BadZipFile:
                        print(f"Bad zip file: {zip_path}")


def download_jigsaw_from_kaggle(output_dir="data/jigsaw"):
    """
    Downloads the Jigsaw Toxicity dataset from Kaggle using the Kaggle API.
    Requires that kaggle.json is correctly installed in ~/.kaggle/.
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. Check if already downloaded
    if (
        os.path.exists(os.path.join(output_dir, "train.csv"))
        and os.path.exists(os.path.join(output_dir, "test.csv"))
    ):
        print("Jigsaw dataset already exists. Skipping download.")
        return

    print("Downloading Jigsaw Toxicity dataset from Kaggle...")

    # 2. Download using Kaggle CLI
    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "jigsaw-toxic-comment-classification-challenge",
                "-p",
                output_dir,
            ],
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ERROR: Kaggle CLI not found. Install it with 'pip install kaggle' "
            "and place kaggle.json in ~/.kaggle/"
        )

    # 3. Find downloaded zip file
    zip_files = [f for f in os.listdir(output_dir) if f.endswith(".zip")]
    if not zip_files:
        raise RuntimeError("Download finished but no zip file was found.")

    # zip_path = "jigsaw-toxic-comment-classification-challenge.zip"

    print("Extracting dataset...")

    # 4. Extract dataset
    unzip_all_recursively("./data")

    print(f"Dataset downloaded and extracted to: {output_dir}")


def preprocess_batch(examples, tokenizer, max_length):
    encodings = tokenizer(
        examples["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    labels = np.stack(
        [
            examples["toxic"],
            examples["severe_toxic"],
            examples["obscene"],
            examples["threat"],
            examples["insult"],
            examples["identity_hate"],
        ],
        axis=1,
    )

    encodings["labels"] = labels.astype("float32")
    return encodings


def load_and_prepare_datasets(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    cache_dir: str | None = None,
) -> Tuple[DatasetDict, list]:
    """
    Downloads the Jigsaw Toxicity dataset, creates train/val/test splits,
    and tokenizes everything for use with PyTorch.

    Returns:
        - tokenized_datasets: DatasetDict with 'train', 'val', and 'test' splits
        - id2label: list with class names (for the model)
    """
    # Ensure dataset exists, otherwise download it
    download_jigsaw_from_kaggle("data/jigsaw")

    # Load train.csv with pandas
    csv_path = "data/jigsaw/train.csv"
    df = pd.read_csv(csv_path)

    required_cols = [
        "comment_text",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {csv_path}")

    # Drop rows with missing text
    df = df.dropna(subset=["comment_text"]).reset_index(drop=True)

    # Convert to HuggingFace Dataset
    full_train = Dataset.from_pandas(df)

    # Train / validation / test split
    train_test = full_train.train_test_split(
        test_size=test_size,
        seed=seed
    )
    train_val = train_test["train"].train_test_split(
        test_size=val_size,
        seed=seed
    )

    train_ds = train_val["train"]
    val_ds = train_val["test"]
    test_ds = train_test["test"]

    # Define binary class names
    id2label = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    # Apply preprocessing in batches
    train_enc = train_ds.map(
        lambda e: preprocess_batch(e, tokenizer, max_length),
        batched=True
    )
    val_enc = val_ds.map(
        lambda e: preprocess_batch(e, tokenizer, max_length),
        batched=True
    )
    test_enc = test_ds.map(
        lambda e: preprocess_batch(e, tokenizer, max_length),
        batched=True
    )

    # Format for PyTorch
    cols = ["input_ids", "attention_mask", "labels"]
    for split in [train_enc, val_enc, test_enc]:
        split.set_format(type="torch", columns=cols)

    return DatasetDict(train=train_enc, val=val_enc, test=test_enc), id2label


def set_global_seed(seed: int = 42):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_data_args():
    """
    CLI argument parser for dataset preprocessing script.
    """
    parser = argparse.ArgumentParser(description="Download and prepare Jigsaw Toxicity dataset")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory where HuggingFace stores datasets")
    return parser.parse_args()


if __name__ == "__main__":
    """
    Optional usage for testing dataset download/preprocessing:

    python -m src.data --max_length 128 --val_size 0.1 --seed 42
    """

    args = parse_data_args()
    set_global_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized, id2label = load_and_prepare_datasets(
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    print(tokenized)
    print("Clases:", id2label)
