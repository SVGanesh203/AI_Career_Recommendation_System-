import joblib
import os

def load_models():
    # Get the project root directory (parent of src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")

    career_model_path = os.path.join(models_dir, "career_model.pkl")
    ctc_low_path = os.path.join(models_dir, "ctc_quantile_low.pkl")
    ctc_high_path = os.path.join(models_dir, "ctc_quantile_high.pkl")

    if not os.path.exists(career_model_path):
        raise FileNotFoundError(
            f"Model file not found. Train models first:\n"
            f"  python src/model_train.py\n"
            f"Expected: {career_model_path}"
        )
    if not (os.path.exists(ctc_low_path) and os.path.exists(ctc_high_path)):
        raise FileNotFoundError(
            f"CTC quantile models not found. Retrain to generate low/high band:\n"
            f"  python src/model_train.py\n"
            f"Expected:\n"
            f"  - {ctc_low_path}\n"
            f"  - {ctc_high_path}"
        )

    career_model = joblib.load(career_model_path)
    ctc_model_low = joblib.load(ctc_low_path)
    ctc_model_high = joblib.load(ctc_high_path)
    return career_model, ctc_model_low, ctc_model_high


def predict_role_and_ctc(features_df):
    career_model, ctc_model_low, ctc_model_high = load_models()

    predicted_role = career_model.predict(features_df)[0]
    pred_low = float(ctc_model_low.predict(features_df)[0])
    pred_high = float(ctc_model_high.predict(features_df)[0])

    low = min(pred_low, pred_high)
    high = max(pred_low, pred_high)
    low = max(low, 0.0)

    return predicted_role, round(low, 2), round(high, 2)
