import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from pathlib import Path


plt.rcParams.update({
    "figure.figsize": (5, 3),
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

COLOR_PROB = "#1f77b4"       # azul
COLOR_UNC = "#6c757d"        # gris
COLOR_ACTIVE = "#2a9d8f"     # verde suave
COLOR_INACTIVE = "#adb5bd"   # gris claro

FIGSIZE = (4, 2.5)
DPI = 120

API_URL = "http://localhost:8000/predict"

# ---- Load dataset robustamente ----
BASE_DIR = Path(__file__).resolve().parents[2]
TEST_PATH = BASE_DIR / "data/jigsaw/test.csv"
df_test = pd.read_csv(TEST_PATH)
texts = df_test["comment_text"].tolist()


# ---- UI ----
st.title("Toxic Comment Classifier")

# --- Input mode ---
mode = st.radio("Modo de entrada:", ["Seleccionar del dataset", "Escribir texto"])

if mode == "Seleccionar del dataset":
    selected_text = st.selectbox("Comentario:", texts)
else:
    selected_text = st.text_area("Introduce tu texto:")

st.write("### Texto:")
st.write(selected_text)


# --- Sliders ---
threshold = st.slider("Threshold de decisión", 0.0, 1.0, 0.5, 0.01)
top_k_val = st.slider("Top-K predicciones", 1, 6, 3, 1)

# ---- Prediction ----
if st.button("Clasificar"):

    try:
        response = requests.post(
            API_URL,
            json={"text": selected_text},
            params={
                "threshold": threshold,
                "top_k": top_k_val
            }
        )

        if response.status_code != 200:
            st.error(f"Error API: {response.status_code}")
            st.text(response.text)
            st.stop()

        result = response.json()

    except Exception as e:
        st.error(f"Error conexión API: {e}")
        st.stop()

    # ---- Parse ----
    all_scores = result["all_scores"]

    labels = [x[0] for x in all_scores]
    probs = [x[1] for x in all_scores]
    uncertainty = [x[2] for x in all_scores]

    active = result["active_labels"]
    top_k = result["top_k"]

    # ---- Active labels con threshold dinámico ----
    # active = [(l, p, u) for l, p, u in all_scores if p >= threshold]

    st.write("## Labels activas")

    if len(active) == 0:
        st.info("No hay labels activas con este threshold")
    else:
        for l, p, u in active:
            st.write(f"- **{l}** | prob={p:.3f} | uncertainty={u:.3f}")


    # ---- Gráfico de probabilidades ----
    st.write("## Distribución de probabilidades")

    # fig1, ax = plt.subplots()
    fig1, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    colors = [COLOR_ACTIVE if p >= threshold else COLOR_INACTIVE for p in probs]

    ax.bar(labels, probs, color=colors)
    ax.axhline(threshold, linestyle="--", linewidth=1)

    ax.set_ylim(0, 1)
    ax.set_title("Probabilidades por label")
    ax.set_ylabel("p(y=1)")
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig1)

    # ---- Gráfico de incertidumbre ----
    st.write("## Incertidumbre")

    sorted_idx = np.argsort(uncertainty)[::-1]

    labels_sorted = [labels[i] for i in sorted_idx]
    unc_sorted = [uncertainty[i] for i in sorted_idx]

    # fig2, ax = plt.subplots()
    fig2, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.bar(labels_sorted, unc_sorted, color=COLOR_UNC)
    ax.set_title("Incertidumbre (ordenada)")
    ax.set_ylabel("Std / Var")
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig2)


    # ---- Plot combinado (más interesante) ----
    st.write("## Probabilidad vs Incertidumbre")

    # fig3, ax = plt.subplots()
    fig3, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.scatter(probs, uncertainty)

    for i, label in enumerate(labels):
        ax.annotate(label, (probs[i], uncertainty[i]), fontsize=8)

    ax.set_xlabel("Probabilidad")
    ax.set_ylabel("Incertidumbre")
    ax.set_title("Confianza del modelo")

    st.pyplot(fig3)

    # ---- Plot top k ----
    st.write("## Probabilidad top k")

    labels_top = [x[0] for x in top_k]
    probs_top = [x[1] for x in top_k]

    # fig4, ax = plt.subplots()
    fig4, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.barh(labels_top[::-1], probs_top[::-1], color=COLOR_PROB)
    ax.set_xlim(0, 1)
    ax.set_title("Top-K predicciones")

    st.pyplot(fig4)

    # ---- Plot normals ----
    st.write("## Distrubuciones de las labels")

    fig5, axes = plt.subplots(2, 3, figsize=FIGSIZE, dpi=DPI)
    axes = axes.flatten()

    x = np.linspace(0, 1, 200)

    for i, ax in enumerate(axes):
        mu = probs[i]
        sigma = uncertainty[i]

        # evitar sigma=0
        sigma = max(sigma, 1e-3)

        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

        ax.plot(x, y)
        ax.axvline(mu, linestyle="--", linewidth=1)

        ax.set_title(labels[i], fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_yticks([])

    plt.tight_layout()
    st.pyplot(fig5)
