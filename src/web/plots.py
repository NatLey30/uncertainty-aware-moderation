import numpy as np
import matplotlib.pyplot as plt
import requests
import re
import random


# ---- STYLE ----
plt.rcParams.update({
    "figure.figsize": (5, 3),
    "font.size": 10,
})

COLOR_PROB = "#1f77b4"
COLOR_UNC = "#6c757d"
COLOR_ACTIVE = "#2a9d8f"
COLOR_INACTIVE = "#adb5bd"

FIGSIZE = (4, 2.5)
DPI = 120


# ---- PROBABILITIES ----
def plot_probabilities(labels, probs, threshold):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    colors = [COLOR_ACTIVE if p >= threshold else COLOR_INACTIVE for p in probs]

    ax.bar(labels, probs, color=colors)
    ax.axhline(threshold, linestyle="--", linewidth=1)

    ax.set_ylim(0, 1)
    ax.set_title("Probabilidades")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


# ---- UNCERTAINTY ----
def plot_uncertainty(labels, uncertainty):
    sorted_idx = np.argsort(uncertainty)[::-1]

    labels_sorted = [labels[i] for i in sorted_idx]
    unc_sorted = [uncertainty[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.bar(labels_sorted, unc_sorted, color=COLOR_UNC)
    ax.set_title("Incertidumbre")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


# ---- SCATTER ----
def plot_prob_vs_unc(labels, probs, uncertainty):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.scatter(probs, uncertainty)

    for i, label in enumerate(labels):
        ax.annotate(label, (probs[i], uncertainty[i]), fontsize=8)

    ax.set_xlabel("Prob")
    ax.set_ylabel("Unc")
    ax.set_title("Confianza")

    plt.tight_layout()
    return fig


# ---- TOP-K ----
def plot_top_k(top_k):
    labels_top = [x[0] for x in top_k]
    probs_top = [x[1] for x in top_k]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.barh(labels_top[::-1], probs_top[::-1], color=COLOR_PROB)
    ax.set_xlim(0, 1)
    ax.set_title("Top-K")

    plt.tight_layout()
    return fig


# ---- NORMALS ----
def plot_normals(labels, probs, uncertainty):
    fig, axes = plt.subplots(2, 3, figsize=(8, 4), dpi=DPI)
    axes = axes.flatten()

    x = np.linspace(0, 1, 200)

    for i, ax in enumerate(axes):
        mu = probs[i]
        sigma = max(uncertainty[i], 1e-3)

        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

        ax.plot(x, y)
        ax.axvline(mu, linestyle="--", linewidth=1)

        ax.set_title(labels[i], fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_yticks([])

    plt.tight_layout()
    return fig


# ---- RADAR ----
def plot_radar(labels, probs, uncertainty):
    N = len(labels)

    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    probs_plot = probs + probs[:1]

    upper = [min(p + u, 1.0) for p, u in zip(probs, uncertainty)]
    lower = [max(p - u, 0.0) for p, u in zip(probs, uncertainty)]

    upper += upper[:1]
    lower += lower[:1]

    # ---- FIGURA PEQUEÑA ----
    fig, ax = plt.subplots(
        subplot_kw=dict(polar=True),
        figsize=(3, 3),
        dpi=120
    )

    # orientación bonita
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # ---- QUITAR GRID CIRCULAR ----
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    # ---- DIBUJAR HEXÁGONO (grid manual) ----
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [r]*len(angles), color="#dddddd", linewidth=0.8)

    # radios
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1], color="#eeeeee", linewidth=0.8)

    # ---- INCERTIDUMBRE (banda) ----
    ax.fill_between(
        angles,
        lower,
        upper,
        color="#6c757d",   # COLOR_UNC
        alpha=0.08
    )

    # ---- PROBABILIDAD ----
    ax.plot(
        angles,
        probs_plot,
        color="#1f77b4",   # COLOR_PROB
        linewidth=1
    )

    # ---- LABELS ----
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    ax.set_yticks([])

    ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig


##### ATAQUES
def leet_speak(text):
    return text.translate(str.maketrans('aeiost', '431057'))


def add_spaces(text):
    return ' '.join(' '.join(w) if len(w) > 4 else w for w in text.split())


def unicode_homoglyphs(text):
    return text.translate(str.maketrans('aeoi', 'аеоі'))


def add_typos(text, rate=0.12, seed=42):
    rng = random.Random(seed)
    result = []
    for ch in text:
        r = rng.random()
        if r < rate / 2:
            pass
        elif r < rate:
            result.extend([ch, ch])
        else:
            result.append(ch)
    return ''.join(result)


def punct_noise(text):
    return re.sub(r'(\w)(\w)', r'\1.\2', text)


def neutral_prefix(text):
    return 'In my humble opinion, ' + text


TRANSFORMS = {
    'leet_speak': leet_speak,
    'add_spaces': add_spaces,
    'homoglyphs': unicode_homoglyphs,
    'typos': add_typos,
    'punct_noise': punct_noise,
    'prefix': neutral_prefix
}


def apply_attacks(text):
    """
    Apply all text transformations.

    Returns:
        dict: {attack_name: transformed_text}
    """
    attacked = {}

    for name, fn in TRANSFORMS.items():
        try:
            attacked[name] = fn(text)
        except Exception:
            attacked[name] = text  # fallback

    return attacked


def evaluate_attacks(
    text,
    api_url,
    threshold=0.5,
    top_k=3
):
    """
    Returns:
        dict: {
            attack_name: {
                "text": attacked_text,
                "labels": [...],
                "probs": [...]
            }
        }
    """
    results = {}

    attacked_texts = apply_attacks(text)
    attacked_texts["original"] = text

    for name, attacked_text in attacked_texts.items():
        try:
            response = requests.post(
                api_url,
                json={"text": attacked_text},
                params={
                    "threshold": threshold,
                    "top_k": top_k
                }
            )

            if response.status_code != 200:
                continue

            data = response.json()
            all_scores = data["all_scores"]

            labels = [x[0] for x in all_scores]
            probs = [x[1] for x in all_scores]

            results[name] = {
                "text": attacked_text,
                "labels": labels,
                "probs": probs
            }

        except Exception:
            continue

    return results


def plot_attack_comparison(results):
    """
    Compare probabilities across attacks using grouped bars.
    """

    attacks = list(results.keys())
    labels = results[attacks[0]]["labels"]

    x = np.arange(len(labels))
    width = 0.8 / len(attacks)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)

    for i, name in enumerate(attacks):
        probs = results[name]["probs"]

        ax.bar(
            x + i * width,
            probs,
            width=width,
            label=name
        )

    ax.set_xticks(x + width * (len(attacks) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45)

    ax.set_ylim(0, 1)
    ax.set_title("Probability under perturbations")

    ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_attack_deltas(results):
    """
    Bar plot of Δ probabilities vs original.
    """

    labels = results["original"]["labels"]
    base_probs = results["original"]["probs"]

    attacks = [k for k in results.keys() if k != "original"]

    x = np.arange(len(labels))
    width = 0.8 / len(attacks)  # repartir espacio

    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)

    for i, name in enumerate(attacks):
        probs = results[name]["probs"]
        delta = np.array(probs) - np.array(base_probs)

        ax.bar(
            x + i * width,
            delta,
            width=width,
            label=name
        )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x + width * (len(attacks) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45)

    ax.set_title("Δ Probability vs original")

    ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def compute_attack_sensitivity(results):
    """
    Compute average absolute change per attack.
    """

    if "original" not in results:
        return {}

    base_probs = results["original"]["probs"]

    scores = {}

    for name, data in results.items():
        if name == "original":
            continue

        probs = data["probs"]

        delta = np.abs(np.array(probs) - np.array(base_probs))
        scores[name] = float(delta.mean())

    return scores


def plot_attack_sensitivity(scores):
    """
    Plot average sensitivity per attack.
    """

    if not scores:
        return None

    names = list(scores.keys())
    values = list(scores.values())

    # ---- ordenar de mayor a menor impacto ----
    sorted_idx = np.argsort(values)[::-1]

    names_sorted = [names[i] for i in sorted_idx]
    values_sorted = [values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=120)

    ax.barh(names_sorted, values_sorted)

    # invertir eje Y para que el mayor quede arriba
    ax.invert_yaxis()

    ax.set_title("Attack sensitivity")
    ax.set_xlabel("Mean |Δ probability|")

    plt.tight_layout()
    return fig
