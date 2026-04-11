import streamlit as st

from src.career_app_page import render_career_assistant
from src.predict import predict_role_and_ctc

st.set_page_config(page_title="AI Career Recommendation System", page_icon="🤖", layout="wide")

st.title("🤖 AI Career Recommendation & CTC Prediction System")
st.write("Upload your resume or enter your skills manually to get career predictions and improvement guidance.")

render_career_assistant(
    predict_role_and_ctc,
    model_badge="**Random Forest** (career role) + **quantile regression** (CTC band)",
)
