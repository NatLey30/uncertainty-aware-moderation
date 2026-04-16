from fastapi import FastAPI
from pydantic import BaseModel
import torch

from src.utils import load_model
from src.gaussian_processes.prediction import predict_with_uncertainty

app = FastAPI(title="Toxicity Detection API")

# —— Load model once at startup ——
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "models/gp/best"
model, tokenizer = load_model(model_path, device)

# These are your labels
id2label = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

app = FastAPI(title="Toxicity API")


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict(
    input_data: TextInput,
    threshold: float,
    top_k: int,
):
    result = predict_with_uncertainty(
        model=model,
        tokenizer=tokenizer,
        text=input_data.text,
        id2label=id2label,
        device=device,
        threshold=threshold,
        top_k=top_k,
    )
    return result
