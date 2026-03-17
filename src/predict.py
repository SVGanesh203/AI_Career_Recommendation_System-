import joblib
import numpy as np
import os

def load_models():
    # Get the project root directory (parent of src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    
    career_model_path = os.path.join(models_dir, "career_model.pkl")
    ctc_model_path = os.path.join(models_dir, "ctc_model.pkl")
    
    if not os.path.exists(career_model_path) or not os.path.exists(ctc_model_path):
        raise FileNotFoundError(
            f"Model files not found. Please train the models first by running:\n"
            f"python src/model_train.py\n"
            f"Expected paths:\n"
            f"  - {career_model_path}\n"
            f"  - {ctc_model_path}"
        )
    
    career_model = joblib.load(career_model_path)
    ctc_model = joblib.load(ctc_model_path)
    return career_model, ctc_model

def predict_role_and_ctc(features_df):
    career_model, ctc_model = load_models()

    predicted_role = career_model.predict(features_df)[0]
    predicted_ctc = ctc_model.predict(features_df)[0]

    # Return CTC range (more realistic than exact)
    low = max(predicted_ctc - 1.0, 0)
    high = predicted_ctc + 1.0

    return predicted_role, round(low, 2), round(high, 2)
