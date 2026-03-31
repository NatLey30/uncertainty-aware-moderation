import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import tqdm
from src.data import load_and_prepare_datasets
from src.utils import load_model
from src.gaussian_processes.prediction import predict_with_uncertainty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model, tokenizer = load_model("models/gp_best", device=device)
model.eval()

datasets, id2label = load_and_prepare_datasets(
    tokenizer=tokenizer,
    max_length=128,
    val_size=0.1,
    test_size=0.1,
    seed=42
)
 
test_set = datasets["test"]
print("longitud test set:", len(test_set))

i = 4
sample = test_set[i]
 
text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
label = sample["labels"]

print("======== EJEMPLO =========")
 
print("TEXT:", text)
print("TRUE LABELS:", label.numpy())
print("CLASS NAMES:", id2label)

print("\nPREDICCIONES:")

result = predict_with_uncertainty(model, tokenizer, text, id2label, device="cuda")
for label, prob, std in result["all_scores"]:
        print(f"{label:15s} | prob={prob:.4f} | std={std:.4f}")

print("=========================")


# --------------------------------------------------
# Agrupar samples por combinación y guardar incertidumbre
# --------------------------------------------------
combos = defaultdict(list)

for i, sample in tqdm.tqdm(enumerate(test_set)):
    labels = tuple(int(x) for x in sample["labels"].numpy())
    text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)

    result = predict_with_uncertainty(
        model=model,
        tokenizer=tokenizer,
        text=text,
        id2label=id2label,
        device=device,
        threshold=0.5,
        top_k=3,
    )

    # probs y stds por label
    probs = np.array([p for _, p, _ in result["all_scores"]])
    stds  = np.array([s for _, _, s in result["all_scores"]])

    pred_labels = (probs >= 0.5).astype(int)
    correct = np.array_equal(np.array(labels), pred_labels)

    # métrica global de incertidumbre
    uncertainty = float(stds.mean())   # opción recomendada

    combos[labels].append({
        "index": i,
        "text": text,
        "combo_true": labels,
        "combo_pred": tuple(pred_labels.tolist()),
        "correct": correct,
        "uncertainty": uncertainty,
        "probs": probs.tolist(),
        "stds": stds.tolist(),
    })

# --------------------------------------------------
# Quedarte con los 2 más inciertos por combinación
# --------------------------------------------------
top2_uncertain_per_combo = {}

for combo, samples in combos.items():
    top2 = sorted(samples, key=lambda x: x["uncertainty"], reverse=True)[:2]
    top2_uncertain_per_combo[combo] = top2

serializable = {
    str(combo): samples
    for combo, samples in top2_uncertain_per_combo.items()
}

# serializable = [
#     {
#         "combo": list(combo),
#         "samples": samples
#     }
#     for combo, samples in top2_uncertain_per_combo.items()
# ]

with open("top2_uncertain_per_combo.json", "w", encoding="utf-8") as f:
    json.dump(serializable, f, ensure_ascii=False, indent=2)

# --------------------------------------------------
# Mostrar resultados
# --------------------------------------------------
for combo in sorted(top2_uncertain_per_combo.keys()):
    print("\n==========================================")
    print("COMBO:", combo)

    samples = top2_uncertain_per_combo[combo]

    if not samples:
        print("No hay samples")
        continue

    for rank, s in enumerate(samples, 1):
        print(f"\n#{rank} más incierto — idx {s['index']}")
        print("TRUE:", s["combo_true"])
        print("PRED:", s["combo_pred"])
        print("CORRECT:", s["correct"])
        print(f"UNCERTAINTY(mean std): {s['uncertainty']:.4f}")
        print("STDS:", np.round(s["stds"], 4))
        print("PROBS:", np.round(s["probs"], 4))
        print("TEXT:", s["text"])


#########################################################

# Asegúrate de que este orden coincide con tus labels
label2id = {label: i for i, label in enumerate(id2label)}


def decode_text(sample, tokenizer):
    return tokenizer.decode(sample["input_ids"], skip_special_tokens=True)


def get_active_labels(label_vector, id2label):
    """
    Devuelve las etiquetas activas de una muestra como tupla.
    Ejemplo: (1,0,1,0,0,0) -> ('toxic', 'obscene')
    """
    return tuple(id2label[i] for i, v in enumerate(label_vector) if int(v) == 1)


def sample_two_per_single_category(dataset, tokenizer, id2label, max_samples=2):
    """
    Devuelve hasta 2 frases por categoría individual.
    Una frase entra en una categoría si esa etiqueta está activa,
    aunque también tenga otras etiquetas.
    """
    results = {label: [] for label in id2label}

    for sample in dataset:
        labels = sample["labels"].numpy()
        text = decode_text(sample, tokenizer)

        for i, label_name in enumerate(id2label):
            if int(labels[i]) == 1 and len(results[label_name]) < max_samples:
                results[label_name].append({
                    "text": text,
                    "labels": labels.copy(),
                    "active_labels": get_active_labels(labels, id2label)
                })

        # salir pronto si ya tenemos todo lleno
        if all(len(v) >= max_samples for v in results.values()):
            break

    return results


def sample_two_per_exact_combination(dataset, tokenizer, id2label, max_samples=2, include_clean=True):
    """
    Devuelve hasta 2 frases por combinación exacta de categorías.
    Ejemplo:
      ('toxic', 'insult') -> frases que tienen exactamente esas etiquetas y no más.
    """
    results = defaultdict(list)

    for sample in dataset:
        labels = sample["labels"].numpy()
        text = decode_text(sample, tokenizer)
        combo = get_active_labels(labels, id2label)

        if not combo and not include_clean:
            continue

        if len(results[combo]) < max_samples:
            results[combo].append({
                "text": text,
                "labels": labels.copy(),
                "active_labels": combo
            })

    return dict(results)


def print_results(title, results):
    print("=" * 100)
    print(title)
    print("=" * 100)

    for group, samples in results.items():
        group_name = group if group else ("clean",)
        print(f"\n{group_name} -> {len(samples)} muestra(s)")
        print("-" * 100)

        for j, s in enumerate(samples, 1):
            print(f"[{j}] {s['text']}")
            print(f"labels vector: {s['labels']}")
            print(f"active_labels: {s['active_labels']}")
            print()


# ----------------------------
# 1) Dos frases por categoría
# ----------------------------
single_category_examples = sample_two_per_single_category(
    dataset=test_set,
    tokenizer=tokenizer,
    id2label=id2label,
    max_samples=2
)

print_results("DOS FRASES POR CADA CATEGORÍA", single_category_examples)


# ----------------------------------------
# 2) Dos frases por combinación exacta
# ----------------------------------------
combination_examples = sample_two_per_exact_combination(
    dataset=test_set,
    tokenizer=tokenizer,
    id2label=id2label,
    max_samples=2,
    include_clean=True  # incluye también frases sin etiquetas
)

print_results("DOS FRASES POR CADA COMBINACIÓN EXACTA", combination_examples)