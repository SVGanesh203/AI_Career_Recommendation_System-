"""
Streamlit page: compare ML / NLP techniques used in this project vs common alternatives.
Run the app from project root: python -m streamlit run app.py
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ml_benchmark import run_benchmark


def _fig_classification_charts(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    df = df.copy()
    df = df.sort_values("Accuracy (%)", ascending=True)

    acc_fig = go.Figure(
        go.Bar(
            x=df["Accuracy (%)"],
            y=df["Model"],
            orientation="h",
            marker_color="#2E86AB",
            text=[f"{v:.1f}%" for v in df["Accuracy (%)"]],
            textposition="outside",
            hovertemplate="%{y}<br>Accuracy: %{x:.2f}%<extra></extra>",
        )
    )
    acc_fig.update_layout(
        title="Classification accuracy (higher is better)",
        xaxis_title="Accuracy (%)",
        yaxis_title="",
        height=max(320, 48 * len(df)),
        margin=dict(l=10, r=80, t=40, b=10),
        showlegend=False,
    )

    df_speed = df.sort_values("Model")
    speed_fig = go.Figure()
    speed_fig.add_trace(
        go.Bar(
            name="Train time",
            x=df_speed["Model"],
            y=df_speed["Train time (s)"],
            marker_color="#A23B72",
        )
    )
    speed_fig.add_trace(
        go.Bar(
            name="Inference time",
            x=df_speed["Model"],
            y=df_speed["Inference time (s)"],
            marker_color="#F18F01",
        )
    )
    speed_fig.update_layout(
        title="Training vs inference time",
        barmode="group",
        xaxis_title="",
        yaxis_title="Seconds",
        height=max(360, 40 * len(df)),
        margin=dict(l=10, r=10, t=40, b=120),
        xaxis_tickangle=-35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return acc_fig, speed_fig


def _fig_regression_charts(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    df = df.copy()
    df_bar = df.sort_values("RMSE (LPA)", ascending=True)

    rmse_fig = go.Figure(
        go.Bar(
            x=df_bar["RMSE (LPA)"],
            y=df_bar["Model"],
            orientation="h",
            marker_color="#1B998B",
            text=[f"{v:.3f}" for v in df_bar["RMSE (LPA)"]],
            textposition="outside",
            hovertemplate="%{y}<br>RMSE: %{x:.3f} LPA<extra></extra>",
        )
    )
    rmse_fig.update_layout(
        title="CTC regression RMSE (lower is better)",
        xaxis_title="RMSE (LPA)",
        yaxis_title="",
        height=max(320, 48 * len(df_bar)),
        margin=dict(l=10, r=80, t=40, b=10),
        showlegend=False,
    )

    trade_fig = go.Figure(
        go.Scatter(
            x=df["Train time (s)"],
            y=df["RMSE (LPA)"],
            mode="markers+text",
            text=df["Model"],
            textposition="top center",
            marker=dict(size=12, color="#C73E1D"),
            hovertemplate="%{text}<br>Train: %{x:.4f}s<br>RMSE: %{y:.3f} LPA<extra></extra>",
        )
    )
    trade_fig.update_layout(
        title="Speed vs error trade-off",
        xaxis_title="Train time (s)",
        yaxis_title="RMSE (LPA)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return rmse_fig, trade_fig

st.set_page_config(page_title="ML Technique Comparison", page_icon="📊", layout="wide")

st.title("📊 ML & NLP Technique Comparison")
st.markdown(
    "This page **benchmarks models on your `data/dataset.csv`** (same train/test split for fairness) "
    "and summarizes **how other techniques** could fit your pipeline."
)

results_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "benchmark_results.json",
)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Run benchmark now", type="primary"):
        with st.spinner("Training and evaluating models…"):
            try:
                run_benchmark()
                st.success("Benchmark finished. Results saved to `data/benchmark_results.json`.")
            except Exception as e:
                st.error(f"Benchmark failed: {e}")
with col_b:
    st.caption("First run may take a minute (neural nets + stacking).")

if os.path.isfile(results_path):
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    st.info(
        f"Loaded results: **{data.get('n_samples', '?')}** samples, "
        f"train **{data.get('n_train', '?')}** / test **{data.get('n_test', '?')}**. "
        f"XGBoost installed: **{data.get('xgboost_available', False)}**."
    )

    st.subheader("Career role prediction (classification)")
    df_clf = pd.DataFrame(data["classification"])
    acc_fig, speed_clf_fig = _fig_classification_charts(df_clf)
    st.plotly_chart(acc_fig, use_container_width=True)
    st.plotly_chart(speed_clf_fig, use_container_width=True)
    with st.expander("View classification results as table"):
        st.dataframe(df_clf, use_container_width=True, hide_index=True)

    st.subheader("CTC prediction (regression)")
    df_reg = pd.DataFrame(data["regression"])
    rmse_fig, trade_fig = _fig_regression_charts(df_reg)
    st.plotly_chart(rmse_fig, use_container_width=True)
    st.plotly_chart(trade_fig, use_container_width=True)
    with st.expander("View regression results as table"):
        st.dataframe(df_reg, use_container_width=True, hide_index=True)

    st.caption(
        "Lower RMSE is better. With a small or synthetic dataset, rankings can change a lot; "
        "use a larger, real dataset for stable comparisons."
    )
else:
    st.warning("No benchmark file yet. Click **Run benchmark now** above.")

st.markdown("---")
st.subheader("Reference: classification alternatives (typical trade-offs)")
st.markdown(
    """
| Aspect | Random Forest (current) | XGBoost | Neural network (MLP) | Logistic regression |
| :--- | :--- | :--- | :--- | :--- |
| **Typical strength** | Strong default, feature importances | Often top accuracy on tabular data | Flexible nonlinear boundaries | Very fast, strong baseline |
| **Training cost** | Moderate | Moderate–high | Higher (tuning + data) | Low |
| **Interpretability** | High (importances) | Medium | Low (black box) | High (coefficients) |
| **Overfitting risk** | Medium | Medium–high if untuned | High if small data | Lower |
"""
)

st.markdown("---")
st.subheader("Reference: regression alternatives (typical trade-offs)")
st.markdown(
    """
| Aspect | Random Forest (current) | XGBoost | Quantile regression | Ensemble stacking |
| :--- | :--- | :--- | :--- | :--- |
| **Output** | Single predicted CTC | Single predicted CTC | Can target **median** or other quantiles | Often best RMSE |
| **Range prediction** | No (you add ± band in app) | No | **Yes** (train low/high quantiles) | Usually still single value |
| **Training cost** | Moderate | Moderate | Low–moderate | High |
| **Best for** | Quick, robust tabular | Maximum accuracy on tabular | **Salary ranges** users trust | Competitions / heavy tuning |
"""
)

st.markdown("---")
st.subheader("Resume processing (NLP) — options vs current spaCy")
st.markdown(
    """
| Approach | What it is | When to use |
| :--- | :--- | :--- |
| **spaCy (current)** | Fast linguistic NLP (`en_core_web_sm`) | Baseline entity/token-based skill matching |
| **Sentence Transformers** | Embedding similarity for synonyms | “ML” vs “machine learning” alignment |
| **Fuzzy matching** (`rapidfuzz`) | Typos / near duplicates | Production speed + robust strings |
| **BERT / RoBERTa** | Transformer encoders | Highest accuracy if you fine-tune on labeled resumes |
| **Word2Vec** | Static word vectors | Older; usually worse than sentence embeddings today |

**Practical hybrid (common in production):** keep spaCy for speed, add **sentence embeddings** or **fuzzy** matching for skill normalization.
"""
)

st.caption(
    "NLP rows above are design guidance. Benchmarking NLP fairly needs labeled resume–skill pairs, "
    "which is separate from tabular ML on `dataset.csv`."
)
