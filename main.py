import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from semantic_matcher import sbert_similarity

import pdfplumber
import docx
import os

# =================================================
# TEXT CLEANING
# =================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =================================================
# RESUME LOADER (PDF / DOCX / TXT)
# =================================================
def load_resume(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    elif file_path.endswith(".pdf"):
        text = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)

    else:
        raise ValueError("Unsupported format")

# =================================================
# RESUME CREATOR (BIAS-FREE)
# =================================================
def generate_resume(role, skills, experience, projects, certifications):
    resume = f"""
PROFESSIONAL SUMMARY
Aspiring {role} with strong technical skills and practical exposure.

SKILLS
{', '.join(skills)}

EXPERIENCE
{experience}

PROJECTS
{projects}
"""
    if certifications.strip():
        resume += f"\nCERTIFICATIONS\n{certifications}\n"
    return resume.strip()

# =================================================
# LOAD O*NET SKILLS
# =================================================
def load_onet_skill_db(skills_file, tech_file, threshold=3.0):
    skills_df = pd.read_excel(skills_file)
    skills_df = skills_df[
        (skills_df["Scale Name"] == "Importance") &
        (skills_df["Data Value"] >= threshold)
    ]

    onet_skills = set(
        skills_df["Element Name"]
        .dropna()
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.strip()
    )

    tech_df = pd.read_excel(tech_file)
    tech_skills = set(
        tech_df["Example"]
        .dropna()
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.strip()
    )

    return onet_skills.union(tech_skills)

# =================================================
# SKILL EXTRACTION
# =================================================
def extract_skills(text, skill_db):
    return sorted(set(text.split()).intersection(skill_db))

# =================================================
# TF-IDF
# =================================================
def compute_tfidf_similarity(resume, jd):
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform([resume, jd])
    return cosine_similarity(mat[0:1], mat[1:2])[0][0] * 100

# =================================================
# LOAD SKILLS
# =================================================
SKILL_DB = load_onet_skill_db(
    "skills_onet.xlsx",
    "Technology Skills.xlsx"
)

# =================================================
# RESUME INPUT MODE
# =================================================
print("Choose Resume Input Method:")
print("1. Upload single resume")
print("2. Manual resume input")
print("3. Create resume on platform")
print("4. Upload multiple resumes (folder)")

choice = input("Enter choice (1/2/3/4): ").strip()

resumes = []

if choice == "1":
    path = input("Enter resume file path: ")
    resumes.append(("Single_Resume", load_resume(path)))

elif choice == "2":
    print("Paste Resume (END to stop):")
    lines = []
    while True:
        l = input()
        if l.strip() == "END":
            break
        lines.append(l)
    resumes.append(("Manual_Resume", "\n".join(lines)))

elif choice == "3":
    role = input("Role: ")
    skills = input("Skills (comma separated): ").lower().split(",")
    experience = input("Experience: ")
    projects = input("Projects: ")
    certifications = input("Certifications (optional): ")

    resume = generate_resume(
        role,
        [s.strip() for s in skills],
        experience,
        projects,
        certifications
    )
    print("\nGENERATED RESUME:\n", resume)
    resumes.append(("Generated_Resume", resume))

elif choice == "4":
    folder = input("Enter folder name (inside project): ").strip()
    files = [
        f for f in os.listdir(folder)
        if f.endswith((".pdf", ".docx", ".txt"))
    ][:20]

    if not files:
        print("No valid resumes found.")
        exit()

    for f in files:
        resumes.append((f, load_resume(os.path.join(folder, f))))

else:
    print("Invalid choice")
    exit()

# =================================================
# JOB DESCRIPTION
# =================================================
print("\nPaste Job Description (END to stop):")
jd_lines = []
while True:
    l = input()
    if l.strip() == "END":
        break
    jd_lines.append(l)

jd = "\n".join(jd_lines)
jd_clean = clean_text(jd)

# =================================================
# SCORING
# =================================================
results = []

for name, resume in resumes:
    rc = clean_text(resume)
    tfidf = compute_tfidf_similarity(rc, jd_clean)
    sbert = sbert_similarity(resume, jd)
    final = 0.6 * sbert + 0.4 * tfidf
    results.append((name, tfidf, sbert, final))

results.sort(key=lambda x: x[3], reverse=True)

# =================================================
# OUTPUT
# =================================================
print("\n===== ALL RESUME SCORES =====")
for r in results:
    print(f"{r[0]:20s} → {r[3]:.2f}%")

print("\n===== SHORTLISTED RESUMES (≥ 50%) =====")
shortlisted = [r for r in results if r[3] >= 50]

for r in shortlisted:
    print(f"{r[0]:20s} → {r[3]:.2f}%")

if shortlisted:
    print(f"\n⭐ BEST RESUME: {shortlisted[0][0]} ({shortlisted[0][3]:.2f}%)")
else:
    print("\nNo resumes met the shortlist threshold.")
