import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np


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

    # Preprocessor (OneHot for degree + pass numerical as-is)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )

    # Career Role Model (Classification)
    role_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    # CTC Model (Regression)
    ctc_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # Split data
    X_train, X_test, y_role_train, y_role_test = train_test_split(
        X, y_role, test_size=0.2, random_state=42
    )

    _, _, y_ctc_train, y_ctc_test = train_test_split(
        X, y_ctc, test_size=0.2, random_state=42
    )

    # Train
    role_model.fit(X_train, y_role_train)
    ctc_model.fit(X_train, y_ctc_train)

    # Evaluate
    role_preds = role_model.predict(X_test)
    role_acc = accuracy_score(y_role_test, role_preds)

    ctc_preds = ctc_model.predict(X_test)
    ctc_rmse = np.sqrt(mean_squared_error(y_ctc_test, ctc_preds))

    print("Career Role Model Accuracy:", round(role_acc * 100, 2), "%")
    print("CTC Model RMSE:", round(ctc_rmse, 2), "LPA")

    # Save models
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    career_model_path = os.path.join(models_dir, "career_model.pkl")
    ctc_model_path = os.path.join(models_dir, "ctc_model.pkl")
    
    joblib.dump(role_model, career_model_path)
    joblib.dump(ctc_model, ctc_model_path)

    print("\nModels saved inside /models folder successfully!")


if __name__ == "__main__":
    train_models()
