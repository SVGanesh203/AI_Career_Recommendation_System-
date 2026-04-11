"""Shared Streamlit UI for Random Forest and XGBoost career assistant pages."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.resume_parser import extract_resume_text
from src.skill_extractor import extract_resume_info
from src.preprocess import build_features
from src.skill_gap_advisor import skill_gap_advice


def render_career_assistant(
    predict_fn: Callable[[pd.DataFrame], tuple[Any, float, float]],
    *,
    model_badge: str,
) -> None:
    st.sidebar.info(model_badge)
    st.sidebar.header("Input Options")
    input_mode = st.sidebar.radio("Choose Input Mode:", ["Upload Resume", "Manual Entry"])

    resume_info = None

    if input_mode == "Upload Resume":
        st.subheader("Upload Resume (PDF/DOCX)")
        uploaded_file = st.file_uploader("Upload your Resume", type=["pdf", "docx"])

        if uploaded_file is not None:
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            resume_text = extract_resume_text(file_path)

            if resume_text:
                st.success("Resume text extracted successfully.")

                resume_info = extract_resume_info(resume_text)

                st.subheader("Extracted Resume Information")
                st.json(resume_info)

    elif input_mode == "Manual Entry":
        st.subheader("Enter Details Manually")

        degree = st.selectbox("Select Qualification:", ["BTECH", "BSC", "MTECH", "MSC", "OTHER"])

        skills_input = st.text_area(
            "Enter Skills (comma separated):",
            placeholder="Example: Python, Machine Learning, SQL, Pandas, Git",
        )

        if skills_input:
            skills_list = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

            resume_info = {
                "name": None,
                "email": None,
                "phone": None,
                "degrees": [degree],
                "skills": skills_list,
            }

            st.subheader("Your Input Summary")
            st.json(resume_info)

    if resume_info is None:
        return

    st.markdown("---")
    st.subheader("Career Prediction Result")

    features_df = build_features(resume_info)

    st.write("Extracted Feature Vector:")
    st.dataframe(features_df)

    try:
        predicted_role, ctc_low, ctc_high = predict_fn(features_df)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Predicted Career Role: **{predicted_role}**")

    with col2:
        st.info(f"Predicted CTC Range: **{ctc_low} LPA - {ctc_high} LPA**")
    st.caption(
        "CTC band uses the same **quantile regression** models (10th–90th percentile) as the main app; "
        "only the **career role** classifier differs on this page."
    )

    st.markdown("---")
    st.subheader("Skill Gap & Improvement Advice")

    advice = skill_gap_advice(predicted_role, resume_info["skills"])

    col3, col4 = st.columns(2)

    with col3:
        st.write("### Missing Required Skills")
        if advice["missing_required_skills"]:
            st.warning(", ".join(advice["missing_required_skills"]))
        else:
            st.success("You already have all required skills.")

    with col4:
        st.write("### Missing Recommended Skills")
        if advice["missing_recommended_skills"]:
            st.info(", ".join(advice["missing_recommended_skills"]))
        else:
            st.success("Great! You have recommended skills too.")

    st.markdown("---")
    st.subheader("CTC Growth Visualization")

    current_ctc = (ctc_low + ctc_high) / 2
    missing_required_count = len(advice["missing_required_skills"])
    missing_recommended_count = len(advice["missing_recommended_skills"])

    required_skills_boost = min(missing_required_count * 0.35, 4.0)
    recommended_skills_boost = min(missing_recommended_count * 0.20, 3.0)

    ctc_after_required = round(current_ctc + required_skills_boost, 2)
    ctc_after_all = round(ctc_after_required + recommended_skills_boost, 2)
    current_ctc = round(current_ctc, 2)

    st.caption(
        "These values are estimated projections based on current prediction and"
        " skill-gap completion. Actual salary growth depends on market, experience,"
        " projects, and interview performance."
    )

    st.write("### Progressive Growth Line Chart (Learning Journey)")
    learning_stages = ["Starting"]
    ctc_progress = [current_ctc]
    running_ctc = current_ctc

    for skill in advice["missing_required_skills"]:
        running_ctc = round(running_ctc + 0.35, 2)
        learning_stages.append(skill.title())
        ctc_progress.append(running_ctc)

    for skill in advice["missing_recommended_skills"]:
        running_ctc = round(running_ctc + 0.20, 2)
        learning_stages.append(skill.title())
        ctc_progress.append(running_ctc)

    if len(learning_stages) == 1:
        learning_stages.append("Current Skillset")
        ctc_progress.append(current_ctc)

    progress_fig = go.Figure()
    progress_fig.add_trace(
        go.Scatter(
            x=learning_stages,
            y=ctc_progress,
            mode="lines+markers",
            line={"color": "#1f77b4", "width": 3},
            marker={"size": 8},
            name="Estimated CTC",
        )
    )
    progress_fig.update_layout(
        xaxis_title="Learning Milestones",
        yaxis_title="Estimated CTC (LPA)",
        xaxis={
            "showgrid": True,
            "gridcolor": "#e6e6e6",
            "gridwidth": 1,
        },
        yaxis={
            "rangemode": "tozero",
            "showgrid": True,
            "gridcolor": "#e6e6e6",
            "gridwidth": 1,
        },
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    st.plotly_chart(progress_fig, use_container_width=True)

    st.write("### Gauge Chart (Before vs After Learning Journey)")

    gauge_max = max(current_ctc, ctc_after_all, 1.0) + 2.0

    def build_gauge(value, title, color):
        return go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                number={"suffix": " LPA"},
                title={"text": title},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, gauge_max * 0.5], "color": "#ececec"},
                        {"range": [gauge_max * 0.5, gauge_max], "color": "#d8d8d8"},
                    ],
                },
            )
        )

    gauge_col1, gauge_col2 = st.columns(2)
    with gauge_col1:
        st.plotly_chart(
            build_gauge(current_ctc, "Before Learning Gap Skills", "#b3472c"),
            use_container_width=True,
        )
    with gauge_col2:
        st.plotly_chart(
            build_gauge(ctc_after_all, "After Learning All Gap Skills", "#1e5f14"),
            use_container_width=True,
        )

    st.write("### Recommended Projects")
    if advice["recommended_projects"]:
        for proj in advice["recommended_projects"]:
            st.write("-", proj)
    else:
        st.write("No project recommendations available.")

    st.write("### Suggested Certifications")
    if advice["certifications"]:
        for cert in advice["certifications"]:
            st.write("-", cert)
    else:
        st.write("No certifications suggested.")
