import pandas as pd

# Skill domain mapping (you can expand later)
AI_SKILLS = {"machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch"}
WEB_SKILLS = {"html", "css", "javascript", "flask", "fastapi", "streamlit"}
DATA_SKILLS = {"python", "sql", "pandas", "numpy", "tableau", "power bi", "data science"}
CLOUD_SKILLS = {"aws", "azure", "docker", "linux", "git", "github"}

def normalize_degree(degrees):
    """
    degrees is a list like ["B.TECH", "M.TECH"]
    return a single category
    """
    if not degrees:
        return "OTHER"

    degrees_joined = " ".join(degrees).lower()

    if "b.tech" in degrees_joined or "btech" in degrees_joined or "be" in degrees_joined:
        return "BTECH"
    elif "b.sc" in degrees_joined or "bsc" in degrees_joined:
        return "BSC"
    elif "m.tech" in degrees_joined or "mtech" in degrees_joined:
        return "MTECH"
    elif "m.sc" in degrees_joined or "msc" in degrees_joined:
        return "MSC"
    else:
        return "OTHER"

def count_domain_skills(skills_set):
    skills_set = set([s.lower() for s in skills_set])

    ai = len(skills_set.intersection(AI_SKILLS))
    web = len(skills_set.intersection(WEB_SKILLS))
    data = len(skills_set.intersection(DATA_SKILLS))
    cloud = len(skills_set.intersection(CLOUD_SKILLS))

    return ai, web, data, cloud

def build_features(resume_info):
    """
    Input: resume_info dict from extract_resume_info()
    Output: pandas DataFrame with 1 row of features
    """
    skills = resume_info.get("skills", [])
    degrees = resume_info.get("degrees", [])

    degree_cat = normalize_degree(degrees)

    total_skills = len(skills)
    ai_skills, web_skills, data_skills, cloud_skills = count_domain_skills(skills)

    features = {
        "degree": degree_cat,
        "total_skills": total_skills,
        "ai_skills": ai_skills,
        "web_skills": web_skills,
        "data_skills": data_skills,
        "cloud_skills": cloud_skills
    }

    return pd.DataFrame([features])
