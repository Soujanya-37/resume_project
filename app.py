import streamlit as st
import pickle
import pandas as pd
import re
import pdfplumber
import docx2txt

# --------------------------------------------
# LOAD MODEL + VECTORIZER + SKILLS
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vector.pkl", "rb"))

skills_df = pd.read_csv("skills.csv")
skills_list = skills_df["Skill"].tolist()

# --------------------------------------------
# UI
st.set_page_config(page_title="Resume Analyzer AI", layout="wide")
st.title("Resume Analyzer AI")

# --------------------------------------------
# TEXT EXTRACTION
def extract_text(file):
    try:
        if file.name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text

        elif file.name.endswith(".docx"):
            return docx2txt.process(file)

        else:
            return file.read().decode("utf-8")

    except:
        return ""

# --------------------------------------------
# CLEAN TEXT
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# --------------------------------------------
# RULE-BASED ROLE FIX
def rule_based_fix(text, prediction):
    text = text.lower()

    categories = {
        "Medical Assistant": ["patient care", "hospital", "clinic", "nurse", "healthcare"],
        "Teacher": ["lesson plan", "classroom", "students", "teaching", "education"],
        "HR": ["recruitment", "onboarding", "payroll", "employee", "hiring"],
        "Data Analyst": ["pandas", "numpy", "sql", "tableau", "power bi", "analysis"],
        "Web Developer": ["html", "css", "javascript", "react", "frontend", "node"],
        "Software Engineer": ["java", "python", "algorithms", "data structures", "backend"]
    }

    scores = {}

    for role, keywords in categories.items():
        scores[role] = sum(1 for word in keywords if word in text)

    best_role = max(scores, key=scores.get)

    if scores[best_role] >= 2:
        return best_role

    return prediction

# --------------------------------------------
# SKILL EXTRACTION
def extract_skills(text):
    found = []
    for skill in skills_list:
        if skill.lower() in text:
            found.append(skill)
    return found

# --------------------------------------------
# FILE UPLOAD
uploaded_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# --------------------------------------------
# PROCESS FILES
if uploaded_files:
    results = []

    for file in uploaded_files:
        text = extract_text(file)

        if not text.strip():
            st.warning(f"Could not read {file.name}")
            continue

        cleaned = clean_text(text)

        # ML Prediction
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        # Apply Rule-Based Fix
        final_prediction = rule_based_fix(cleaned, prediction)

        # Extract Skills
        skills = extract_skills(cleaned)

        results.append({
            "File": file.name,
            "Role": final_prediction,
            "Skills": ", ".join(skills) if skills else "No skills detected"
        })

    df = pd.DataFrame(results)

    st.success("Analysis Completed ✅")
    st.dataframe(df, use_container_width=True)