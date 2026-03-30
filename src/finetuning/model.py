from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)


def build_model_and_tokenizer(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    id2label: list = [
                    "toxic",
                    "severe_toxic",
                    "obscene",
                    "threat",
                    "insult",
                    "identity_hate",
                ],
):
    """
    Load tokenizer and HuggingFace model ready for finetuning.
    """
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(id2label)},
        label2id={label: i for i, label in enumerate(id2label)},
        problem_type="multi_label_classification"
    )

    return model, tokenizer
