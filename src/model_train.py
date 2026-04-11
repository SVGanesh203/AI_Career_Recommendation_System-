import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Predicted CTC band: lower / upper quantiles of salary (train on labels in LPA).
CTC_QUANTILE_LOW = 0.1
CTC_QUANTILE_HIGH = 0.9


def _one_hot_encoder():
    try:
        return __import__("sklearn.preprocessing", fromlist=["OneHotEncoder"]).OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
    except TypeError:
        return __import__("sklearn.preprocessing", fromlist=["OneHotEncoder"]).OneHotEncoder(
            handle_unknown="ignore", sparse=False
        )


def _build_preprocessor(categorical_features, numerical_features):
    return ColumnTransformer(
        transformers=[
            ("cat", _one_hot_encoder(), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )


def train_models(dataset_path="data/dataset.csv"):
    # Get the project root directory (parent of src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Make paths absolute
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(project_root, dataset_path)
    df = pd.read_csv(dataset_path)

    # Features and targets
    X = df.drop(["career_role", "ctc"], axis=1)
    y_role = df["career_role"]
    y_ctc = df["ctc"]

    # Columns
    categorical_features = ["degree"]
    numerical_features = ["total_skills", "ai_skills", "web_skills", "data_skills", "cloud_skills"]

    # Separate preprocessors per pipeline (each fits its own copy).
    # Dense one-hot so QuantileRegressor receives dense features.
    # Career Role Model (Classification)
    role_model = Pipeline(steps=[
        ("preprocessor", _build_preprocessor(categorical_features, numerical_features)),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    # CTC: two quantile regressors for a data-driven low–high band.
    ctc_model_low = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(categorical_features, numerical_features)),
            (
                "regressor",
                QuantileRegressor(
                    quantile=CTC_QUANTILE_LOW,
                    alpha=0.0,
                    solver="highs",
                ),
            ),
        ]
    )
    ctc_model_high = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(categorical_features, numerical_features)),
            (
                "regressor",
                QuantileRegressor(
                    quantile=CTC_QUANTILE_HIGH,
                    alpha=0.0,
                    solver="highs",
                ),
            ),
        ]
    )

    # Single split so role and CTC targets stay aligned.
    X_train, X_test, y_role_train, y_role_test, y_ctc_train, y_ctc_test = train_test_split(
        X, y_role, y_ctc, test_size=0.2, random_state=42
    )

    # Train
    role_model.fit(X_train, y_role_train)
    ctc_model_low.fit(X_train, y_ctc_train)
    ctc_model_high.fit(X_train, y_ctc_train)

    # Evaluate
    role_preds = role_model.predict(X_test)
    role_acc = accuracy_score(y_role_test, role_preds)

    pred_low = ctc_model_low.predict(X_test)
    pred_high = ctc_model_high.predict(X_test)
    # Enforce ordering for metrics if quantiles cross on noisy data
    band_low = np.minimum(pred_low, pred_high)
    band_high = np.maximum(pred_low, pred_high)
    median_mid = 0.5 * (band_low + band_high)
    ctc_rmse = np.sqrt(mean_squared_error(y_ctc_test, median_mid))

    y_test_arr = np.asarray(y_ctc_test)
    in_band = float(np.mean((y_test_arr >= band_low) & (y_test_arr <= band_high)) * 100)

    print("Career Role Model (Random Forest) Accuracy:", round(role_acc * 100, 2), "%")

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    career_xgb_path = os.path.join(models_dir, "career_model_xgb.pkl")
    if HAS_XGBOOST:
        role_le = LabelEncoder()
        role_le.fit(y_role)
        y_train_enc = role_le.transform(y_role_train)
        role_xgb = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(categorical_features, numerical_features)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        verbosity=0,
                    ),
                ),
            ]
        )
        role_xgb.fit(X_train, y_train_enc)
        pred_enc = role_xgb.predict(X_test)
        pred_labels = role_le.inverse_transform(np.asarray(pred_enc).astype(int))
        xgb_acc = accuracy_score(y_role_test, pred_labels)
        print("Career Role Model (XGBoost) Accuracy:", round(xgb_acc * 100, 2), "%")
        joblib.dump({"pipeline": role_xgb, "label_encoder": role_le}, career_xgb_path)
    else:
        print("XGBoost not installed; skipped saving career_model_xgb.pkl")
    label = "CTC band (quantiles {:.0f}%-{:.0f}%): RMSE vs mid-band ~".format(
        100 * CTC_QUANTILE_LOW,
        100 * CTC_QUANTILE_HIGH,
    )
    print(label, round(ctc_rmse, 2), "LPA")
    print("Share of test CTC within predicted band:", round(in_band, 1), "%")

    # Save models
    career_model_path = os.path.join(models_dir, "career_model.pkl")
    ctc_low_path = os.path.join(models_dir, "ctc_quantile_low.pkl")
    ctc_high_path = os.path.join(models_dir, "ctc_quantile_high.pkl")

    joblib.dump(role_model, career_model_path)
    joblib.dump(ctc_model_low, ctc_low_path)
    joblib.dump(ctc_model_high, ctc_high_path)

    print("\nModels saved inside /models folder successfully!")
    if HAS_XGBOOST:
        print("XGBoost career classifier:", career_xgb_path)


if __name__ == "__main__":
    train_models()
