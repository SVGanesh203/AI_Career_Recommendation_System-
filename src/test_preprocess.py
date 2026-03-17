from src.resume_parser import extract_resume_text
from src.skill_extractor import extract_resume_info
from src.preprocess import build_features

resume_text = extract_resume_text("data/sample_resumes/sample.pdf")
info = extract_resume_info(resume_text)

features_df = build_features(info)

print("\n===== Resume Info =====")
print(info)

print("\n===== ML Features =====")
print(features_df)
