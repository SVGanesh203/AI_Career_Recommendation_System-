"""
Train and compare multiple ML approaches on the same split as production models.
Run: python -m src.ml_benchmark
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, QuantileRegressor, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_preprocessor(categorical_features: list[str], numerical_features: list[str]) -> ColumnTransformer:
    # Dense output helps linear / quantile models that do not accept sparse X.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )


def run_benchmark(dataset_path: str | None = None) -> dict[str, Any]:
    root = _project_root()
    if dataset_path is None:
        dataset_path = os.path.join(root, "data", "dataset.csv")
    elif not os.path.isabs(dataset_path):
        dataset_path = os.path.join(root, dataset_path)

    df = pd.read_csv(dataset_path)
    X = df.drop(["career_role", "ctc"], axis=1)
    y_role = df["career_role"]
    y_ctc = df["ctc"]

    categorical_features = ["degree"]
    numerical_features = ["total_skills", "ai_skills", "web_skills", "data_skills", "cloud_skills"]
    # Single split for fair comparison (same rows for role and CTC)
    X_train, X_test, y_role_train, y_role_test, y_ctc_train, y_ctc_test = train_test_split(
        X, y_role, y_ctc, test_size=0.2, random_state=42
    )

    classification_specs: list[tuple[str, Any]] = [
        (
            "Random Forest (current)",
            RandomForestClassifier(n_estimators=200, random_state=42),
        ),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        (
            "Neural Network (MLP)",
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1200, random_state=42),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(n_estimators=100, random_state=42),
        ),
    ]
    if HAS_XGBOOST:
        classification_specs.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0,
                ),
            )
        )

    regression_specs: list[tuple[str, Any]] = [
        (
            "Random Forest (current)",
            RandomForestRegressor(n_estimators=200, random_state=42),
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(n_estimators=100, random_state=42),
        ),
        (
            "Neural Network (MLP)",
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1200, random_state=42),
        ),
        (
            "Quantile Regression (median)",
            QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"),
        ),
    ]
    if HAS_XGBOOST:
        regression_specs.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0,
                ),
            )
        )

    clf_rows: list[dict[str, Any]] = []
    for name, estimator in classification_specs:
        preprocessor = _make_preprocessor(categorical_features, numerical_features)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        y_fit = y_role_train
        if name == "XGBoost" and HAS_XGBOOST:
            role_le = LabelEncoder()
            y_fit = role_le.fit_transform(y_role_train)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_fit)
        train_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        preds = pipe.predict(X_test)
        infer_sec = time.perf_counter() - t1
        if name == "XGBoost" and HAS_XGBOOST:
            preds = role_le.inverse_transform(preds.astype(int))

        acc = accuracy_score(y_role_test, preds)
        clf_rows.append(
            {
                "Model": name,
                "Accuracy (%)": round(acc * 100, 2),
                "Train time (s)": round(train_sec, 4),
                "Inference time (s)": round(infer_sec, 6),
            }
        )

    reg_rows: list[dict[str, Any]] = []
    for name, estimator in regression_specs:
        preprocessor = _make_preprocessor(categorical_features, numerical_features)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        t0 = time.perf_counter()
        pipe.fit(X_train, y_ctc_train)
        train_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        preds = pipe.predict(X_test)
        infer_sec = time.perf_counter() - t1
        rmse = float(np.sqrt(mean_squared_error(y_ctc_test, preds)))
        reg_rows.append(
            {
                "Model": name,
                "RMSE (LPA)": round(rmse, 3),
                "Train time (s)": round(train_sec, 4),
                "Inference time (s)": round(infer_sec, 6),
            }
        )

    # Stacking ensemble (two strong sklearn bases + Ridge meta)
    pre_rf = _make_preprocessor(categorical_features, numerical_features)
    pre_gb = _make_preprocessor(categorical_features, numerical_features)
    est_rf = Pipeline(
        [
            ("preprocessor", pre_rf),
            ("model", RandomForestRegressor(n_estimators=150, random_state=42)),
        ]
    )
    est_gb = Pipeline(
        [
            ("preprocessor", pre_gb),
            ("model", GradientBoostingRegressor(n_estimators=80, random_state=42)),
        ]
    )
    stack = StackingRegressor(
        estimators=[("rf", est_rf), ("gb", est_gb)],
        final_estimator=Ridge(alpha=1.0),
        passthrough=False,
    )
    t0 = time.perf_counter()
    stack.fit(X_train, y_ctc_train)
    st_train = time.perf_counter() - t0
    t1 = time.perf_counter()
    sp = stack.predict(X_test)
    st_infer = time.perf_counter() - t1
    st_rmse = float(np.sqrt(mean_squared_error(y_ctc_test, sp)))
    reg_rows.append(
        {
            "Model": "Ensemble stacking (RF + GB + Ridge)",
            "RMSE (LPA)": round(st_rmse, 3),
            "Train time (s)": round(st_train, 4),
            "Inference time (s)": round(st_infer, 6),
        }
    )

    # Sort for readability: classification by accuracy desc, regression by RMSE asc
    clf_rows.sort(key=lambda r: r["Accuracy (%)"], reverse=True)
    reg_rows.sort(key=lambda r: r["RMSE (LPA)"])

    out: dict[str, Any] = {
        "dataset_path": dataset_path,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "xgboost_available": HAS_XGBOOST,
        "classification": clf_rows,
        "regression": reg_rows,
    }

    out_path = os.path.join(root, "data", "benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


if __name__ == "__main__":
    r = run_benchmark()
    print(json.dumps(r, indent=2))
