# src/skill_gap_advisor.py

ROLE_SKILL_MAP = {
    "Data Analyst": {
        "required_skills": ["python", "sql", "pandas", "numpy", "power bi", "tableau"],
        "recommended_projects": [
            "Sales Data Dashboard using Power BI",
            "Customer Churn Prediction using ML",
            "Exploratory Data Analysis (EDA) on real datasets"
        ],
        "certifications": ["Google Data Analytics", "Power BI basics"]
    },

    "ML Engineer": {
        "required_skills": ["python", "machine learning", "sql", "scikit-learn", "pandas", "numpy"],
        "recommended_skills": ["deep learning", "tensorflow", "pytorch", "nlp", "computer vision"],
        "recommended_projects": [
            "Resume Screening AI (NLP)",
            "House Price Prediction + Deployment",
            "Image Classification using CNN (basic)"
        ],
        "certifications": ["ML by Andrew Ng", "Deep Learning Specialization"]
    },

    "Web Developer": {
        "required_skills": ["html", "css", "javascript"],
        "recommended_skills": ["react", "node.js", "mongodb", "flask"],
        "recommended_projects": [
            "Portfolio Website",
            "E-commerce Product Page",
            "Web App with Login + Database"
        ],
        "certifications": ["Frontend Development", "JavaScript Essentials"]
    },

    "Software Engineer": {
        "required_skills": ["python", "java", "c++", "sql", "git"],
        "recommended_skills": ["data structures", "algorithms", "system design"],
        "recommended_projects": [
            "Task Manager App (CRUD + DB)",
            "Mini Compiler / Interpreter",
            "API-based backend service"
        ],
        "certifications": ["DSA course", "System Design basics"]
    },

    "QA Engineer": {
        "required_skills": ["testing", "selenium", "python", "java"],
        "recommended_skills": ["api testing", "automation testing"],
        "recommended_projects": [
            "Automated Testing for a Website",
            "API Testing with Postman",
            "Bug Tracking System"
        ],
        "certifications": ["Selenium automation basics"]
    }
}


def skill_gap_advice(predicted_role, user_skills):
    """
    Input:
        predicted_role: string
        user_skills: list of extracted skills
    Output:
        dict with missing skills + recommendations
    """

    user_skills_set = set([s.lower() for s in user_skills])

    if predicted_role not in ROLE_SKILL_MAP:
        return {
            "message": "No skill map available for this role yet.",
            "missing_skills": [],
            "recommended_skills": [],
            "recommended_projects": [],
            "certifications": []
        }

    role_data = ROLE_SKILL_MAP[predicted_role]

    required_skills = set([s.lower() for s in role_data.get("required_skills", [])])
    recommended_skills = set([s.lower() for s in role_data.get("recommended_skills", [])])

    missing_required = sorted(list(required_skills - user_skills_set))
    missing_recommended = sorted(list(recommended_skills - user_skills_set))

    return {
        "role": predicted_role,
        "missing_required_skills": missing_required,
        "missing_recommended_skills": missing_recommended,
        "recommended_projects": role_data.get("recommended_projects", []),
        "certifications": role_data.get("certifications", [])
    }
