"""Predict career role with XGBoost; CTC band uses the same quantile models as `predict.py`."""
from __future__ import annotations

import joblib
import os

import numpy as np
import pandas as pd


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_models_xgb():
    root = _project_root()
    models_dir = os.path.join(root, "models")

    career_xgb_path = os.path.join(models_dir, "career_model_xgb.pkl")
    ctc_low_path = os.path.join(models_dir, "ctc_quantile_low.pkl")
    ctc_high_path = os.path.join(models_dir, "ctc_quantile_high.pkl")

    if not os.path.exists(career_xgb_path):
        raise FileNotFoundError(
            "XGBoost career model not found. Install xgboost and retrain:\n"
            "  pip install xgboost\n"
            "  python src/model_train.py\n"
            f"Expected: {career_xgb_path}"
        )
    if not (os.path.exists(ctc_low_path) and os.path.exists(ctc_high_path)):
        raise FileNotFoundError(
            "CTC quantile models missing. Run:\n"
            "  python src/model_train.py\n"
            f"Expected:\n  - {ctc_low_path}\n  - {ctc_high_path}"
        )

    bundle = joblib.load(career_xgb_path)
    ctc_low = joblib.load(ctc_low_path)
    ctc_high = joblib.load(ctc_high_path)
    return bundle, ctc_low, ctc_high


def predict_role_and_ctc_xgb(features_df: pd.DataFrame) -> tuple[str, float, float]:
    bundle, ctc_model_low, ctc_model_high = load_models_xgb()

    pipe = bundle["pipeline"]
    enc = bundle["label_encoder"]

    pred_enc = pipe.predict(features_df)
    pred_enc = np.asarray(pred_enc).astype(int)
    predicted_role = str(enc.inverse_transform(pred_enc.ravel())[0])

    pred_low = float(ctc_model_low.predict(features_df)[0])
    pred_high = float(ctc_model_high.predict(features_df)[0])
    low = min(pred_low, pred_high)
    high = max(pred_low, pred_high)
    low = max(low, 0.0)

    return predicted_role, round(low, 2), round(high, 2)
