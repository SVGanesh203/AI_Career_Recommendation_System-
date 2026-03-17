from src.resume_parser import extract_resume_text
from src.skill_extractor import extract_resume_info

resume_text = extract_resume_text("data/sample_resumes/sample.pdf")
info = extract_resume_info(resume_text)

print("\n===== Extracted Resume Info =====")
print(info)
