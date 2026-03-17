import streamlit as st
import pandas as pd

from src.resume_parser import extract_resume_text
from src.skill_extractor import extract_resume_info
from src.preprocess import build_features
from src.predict import predict_role_and_ctc
from src.skill_gap_advisor import skill_gap_advice


st.set_page_config(page_title="AI Career Recommendation System", page_icon="🤖", layout="wide")

st.title("🤖 AI Career Recommendation & CTC Prediction System")
st.write("Upload your resume or enter your skills manually to get career predictions and improvement guidance.")

# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Input Options")
input_mode = st.sidebar.radio("Choose Input Mode:", ["Upload Resume", "Manual Entry"])

# ------------------ MAIN UI ------------------
resume_info = None

if input_mode == "Upload Resume":
    st.subheader("📄 Upload Resume (PDF/DOCX)")
    uploaded_file = st.file_uploader("Upload your Resume", type=["pdf", "docx"])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract resume text
        resume_text = extract_resume_text(file_path)

        if resume_text:
            st.success("✅ Resume text extracted successfully!")

            # Extract info using NLP
            resume_info = extract_resume_info(resume_text)

            st.subheader("🔍 Extracted Resume Information")
            st.json(resume_info)

elif input_mode == "Manual Entry":
    st.subheader("✍️ Enter Details Manually")

    degree = st.selectbox("Select Qualification:", ["BTECH", "BSC", "MTECH", "MSC", "OTHER"])

    skills_input = st.text_area(
        "Enter Skills (comma separated):",
        placeholder="Example: Python, Machine Learning, SQL, Pandas, Git"
    )

    if skills_input:
        skills_list = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

        resume_info = {
            "name": None,
            "email": None,
            "phone": None,
            "degrees": [degree],
            "skills": skills_list
        }

        st.subheader("📌 Your Input Summary")
        st.json(resume_info)

# ------------------ Prediction Section ------------------
if resume_info is not None:
    st.markdown("---")
    st.subheader("🎯 Career Prediction Result")

    # Convert resume info to features
    features_df = build_features(resume_info)

    st.write("📊 Extracted Feature Vector:")
    st.dataframe(features_df)

    # Predict career role + ctc
    predicted_role, ctc_low, ctc_high = predict_role_and_ctc(features_df)

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"✅ Predicted Career Role: **{predicted_role}**")

    with col2:
        st.info(f"💰 Predicted CTC Range: **{ctc_low} LPA - {ctc_high} LPA**")

    # Skill gap advice
    st.markdown("---")
    st.subheader("📌 Skill Gap & Improvement Advice")

    advice = skill_gap_advice(predicted_role, resume_info["skills"])

    col3, col4 = st.columns(2)

    with col3:
        st.write("### ❌ Missing Required Skills")
        if advice["missing_required_skills"]:
            st.warning(", ".join(advice["missing_required_skills"]))
        else:
            st.success("You already have all required skills 🎉")

    with col4:
        st.write("### ⭐ Missing Recommended Skills")
        if advice["missing_recommended_skills"]:
            st.info(", ".join(advice["missing_recommended_skills"]))
        else:
            st.success("Great! You have recommended skills too 🚀")

    st.write("### 🛠 Recommended Projects")
    if advice["recommended_projects"]:
        for proj in advice["recommended_projects"]:
            st.write("✅", proj)
    else:
        st.write("No project recommendations available.")

    st.write("### 🎓 Suggested Certifications")
    if advice["certifications"]:
        for cert in advice["certifications"]:
            st.write("📌", cert)
    else:
        st.write("No certifications suggested.")
