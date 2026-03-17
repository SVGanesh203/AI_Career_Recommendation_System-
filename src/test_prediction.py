from src.resume_parser import extract_resume_text
from src.skill_extractor import extract_resume_info
from src.preprocess import build_features
from src.predict import predict_role_and_ctc
from src.skill_gap_advisor import skill_gap_advice

resume_text = extract_resume_text("data/sample_resumes/sample.pdf")
info = extract_resume_info(resume_text)

features_df = build_features(info)

role, ctc_low, ctc_high = predict_role_and_ctc(features_df)

advice = skill_gap_advice(role, info["skills"])

print("\n===== Prediction Result =====")
print("Predicted Career Role:", role)
print(f"Predicted CTC Range: {ctc_low} LPA - {ctc_high} LPA")

print("\n===== Skill Gap Advice =====")
print("Missing Required Skills:", advice["missing_required_skills"])
print("Missing Recommended Skills:", advice["missing_recommended_skills"])

print("\nRecommended Projects:")
for p in advice["recommended_projects"]:
    print("-", p)

print("\nSuggested Certifications:")
for c in advice["certifications"]:
    print("-", c)

