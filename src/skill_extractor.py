import re
import spacy

nlp = spacy.load("en_core_web_sm")

# 🔥 Common skills list (you can expand anytime)
SKILLS_DB = [
    "python", "java", "c", "c++", "sql", "html", "css", "javascript",
    "machine learning", "deep learning", "data science", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "flask", "fastapi", "streamlit", "git", "github", "docker",
    "aws", "azure", "linux", "power bi", "tableau"
]

DEGREES_DB = [
    "b.tech", "btech", "b.e", "be", "b.sc", "bsc", "m.tech", "mtech",
    "m.sc", "msc", "mba", "phd"
]

def extract_email(text):
    match = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match[0] if match else None

def extract_phone(text):
    match = re.findall(r"\+?\d[\d -]{8,}\d", text)
    return match[0] if match else None

def extract_degree(text):
    text_lower = text.lower()
    found = []
    for deg in DEGREES_DB:
        if deg in text_lower:
            found.append(deg.upper())
    return list(set(found))

def extract_skills(text):
    text_lower = text.lower()
    doc = nlp(text_lower)

    found_skills = set()

    # 1) Match skills directly from skill DB (fast + accurate)
    for skill in SKILLS_DB:
        if skill in text_lower:
            found_skills.add(skill)

    # 2) Extract noun chunks (for multi-word skills like "machine learning")
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if chunk_text in SKILLS_DB:
            found_skills.add(chunk_text)

    return sorted(found_skills)

def extract_name(text):
    """
    Simple method:
    - takes first PERSON entity found
    (not always perfect but good for UG project)
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_resume_info(text):
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "degrees": extract_degree(text),
        "skills": extract_skills(text)
    }
