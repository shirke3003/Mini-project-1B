from flask import Flask, render_template, request
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from semantic_matcher import sbert_similarity
import pdfplumber
import docx

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =================================================
# TEXT CLEANING
# =================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =================================================
# RESUME LOADER
# =================================================
def load_resume(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif path.endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif path.endswith(".pdf"):
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text.append(page.extract_text())
        return "\n".join(text)

    return ""

# =================================================
# LOAD O*NET SKILLS
# =================================================
def load_onet_skill_db():
    skills_df = pd.read_excel("skills_onet.xlsx")
    tech_df = pd.read_excel("Technology Skills.xlsx")

    skills_df = skills_df[
        (skills_df["Scale Name"] == "Importance") &
        (skills_df["Data Value"] >= 3.0)
    ]

    onet_skills = set(
        skills_df["Element Name"]
        .dropna()
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
    )

    tech_skills = set(
        tech_df["Example"]
        .dropna()
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
    )

    return onet_skills.union(tech_skills)

SKILL_DB = load_onet_skill_db()

# =================================================
# TF-IDF
# =================================================
def compute_tfidf_similarity(resume, jd):
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform([resume, jd])
    return cosine_similarity(mat[0:1], mat[1:2])[0][0] * 100

# =================================================
# ROUTE
# =================================================
@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        jd = request.form["job_description"]
        jd_clean = clean_text(jd)

        files = request.files.getlist("resumes")[:20]

        for file in files:
            filename = file.filename
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            resume_text = load_resume(path)
            resume_clean = clean_text(resume_text)

            tfidf = compute_tfidf_similarity(resume_clean, jd_clean)
            sbert = sbert_similarity(resume_text, jd)
            final = 0.6 * sbert + 0.4 * tfidf

            results.append({
                "name": filename,
                "tfidf": round(tfidf, 2),
                "sbert": round(sbert, 2),
                "final": round(final, 2),
                "status": "SHORTLISTED" if final >= 50 else "REJECTED"
            })

        results.sort(key=lambda x: x["final"], reverse=True)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

