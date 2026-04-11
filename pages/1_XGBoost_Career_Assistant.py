import streamlit as st

from src.career_app_page import render_career_assistant
from src.predict_xgb import predict_role_and_ctc_xgb

st.set_page_config(page_title="XGBoost Career Assistant", page_icon="🚀", layout="wide")

st.title("🚀 AI Career Assistant (XGBoost)")
st.write(
    "Same workflow as the home app. **Career role** is predicted with **XGBoost**; "
    "**CTC low–high** still uses the trained **quantile regression** models."
)

render_career_assistant(
    predict_role_and_ctc_xgb,
    model_badge="**XGBoost** (career role) + **quantile regression** (CTC band)",
)
