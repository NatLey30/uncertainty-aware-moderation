import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from pathlib import Path


from src.web.plots import (
    plot_probabilities,
    plot_uncertainty,
    plot_prob_vs_unc,
    plot_top_k,
    plot_normals,
    plot_radar,
    evaluate_attacks,
    plot_attack_comparison,
    plot_attack_deltas,
    compute_attack_sensitivity,
    plot_attack_sensitivity,
)


API_URL = "http://localhost:8000/predict"

# ---- Load dataset robustamente ----
BASE_DIR = Path(__file__).resolve().parents[2]
TEST_PATH = BASE_DIR / "data/jigsaw/test.csv"
df_test = pd.read_csv(TEST_PATH)
texts = df_test["comment_text"].tolist()


ATTACK_DESCRIPTIONS = {
    "original": "Original input text",
    "leet_speak": "Characters replaced with visually similar numbers (l33t)",
    "add_spaces": "Words broken into characters with spaces",
    "homoglyphs": "Characters replaced with visually similar Unicode symbols",
    "typos": "Random character deletions and duplications",
    "punct_noise": "Extra punctuation inserted between characters",
    "prefix": "Neutral prefix added to dilute toxicity signal",
}


def get_prediction(text, threshold, top_k):
    response = requests.post(
        API_URL,
        json={"text": text},
        params={
            "threshold": threshold,
            "top_k": top_k
        }
    )

    return response


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
        # response = requests.post(
        #     API_URL,
        #     json={"text": selected_text},
        #     params={
        #         "threshold": threshold,
        #         "top_k": top_k_val
        #     }
        # )
        response = get_prediction(
            text=selected_text,
            threshold=threshold,
            top_k=top_k_val,
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

    # st.pyplot(plot_radar(labels, probs, uncertainty))
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.pyplot(plot_radar(labels, probs, uncertainty), width="stretch")

    # ---- Gráficos ----
    col1, col2 = st.columns(2)

    with col1:
        st.write("## Probabilidades")
        st.pyplot(plot_probabilities(labels, probs, threshold), width="stretch")

    with col2:
        st.write("## Incertidumbre")
        st.pyplot(plot_uncertainty(labels, uncertainty), width="stretch")

    col3, col4 = st.columns(2)

    with col3:
        st.write("## Prob vs Incertidumbre")
        st.pyplot(plot_prob_vs_unc(labels, probs, uncertainty), width="stretch")

    with col4:
        st.write("## Top-K labels")
        st.pyplot(plot_top_k(top_k), width="stretch")

    # st.write("## Probabilidades")
    # st.pyplot(plot_probabilities(labels, probs, threshold))

    # st.write("## Incertidumbre")
    # st.pyplot(plot_uncertainty(labels, uncertainty))

    # st.write("## Prob vs Incertidumbre")
    # st.pyplot(plot_prob_vs_unc(labels, probs, uncertainty))

    # st.write("## Top-K")
    # st.pyplot(plot_top_k(top_k))

    st.write("## Distribuciones")
    st.pyplot(plot_normals(labels, probs, uncertainty))

    # if st.button("Run attacks"):

    results = evaluate_attacks(
        selected_text,
        API_URL,
        threshold,
        top_k_val
    )

    st.write("## Text transformations")

    for name, data in results.items():
        with st.expander(name):
            st.markdown(f"**Attack:** {ATTACK_DESCRIPTIONS.get(name, 'Unknown attack')}")
            st.write(data["text"])

    st.write("## Robustness analysis")

    # st.pyplot(plot_attack_comparison(results))
    # st.markdown("""
    # This plot shows how the predicted class probabilities change under different text perturbations.
    # Each group of bars corresponds to a class, and each color represents a different attack.
    # It allows us to observe how the model's output distribution shifts when the input is modified.
    # """)

    st.pyplot(plot_attack_deltas(results), width="stretch")
    st.markdown("""
    This plot shows the change in predicted probabilities with respect to the original input.
    Positive values indicate an increase, while negative values indicate a decrease.
    It highlights which classes are most affected by each perturbation.
    """)

    scores = compute_attack_sensitivity(results)
    st.pyplot(plot_attack_sensitivity(scores), width="stretch")
    st.markdown("""
    This plot summarizes the overall impact of each attack.
    It measures the average absolute change in predicted probabilities across all classes.
    Higher values indicate that the model is more sensitive to that perturbation.
    """)

