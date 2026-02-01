from flask import Flask, render_template, request, redirect, session, send_file
from flask_pymongo import PyMongo
from flask import send_file
import os, re, io
import pdfplumber, docx
import pandas as pd
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from semantic_matcher import sbert_similarity
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "final_secret_key"

# ===================== MongoDB =====================
app.config["MONGO_URI"] = "mongodb+srv://resume_user:fcritru12345@cluster0.vpqwj7a.mongodb.net/ai_resume_db?retryWrites=true&w=majority"
mongo = PyMongo(app)

users = mongo.db.users
jds = mongo.db.job_descriptions
resumes = mongo.db.resumes

# ===================== Utilities =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_file(file):
    if file.filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    if file.filename.endswith(".docx"):
        d = docx.Document(file)
        return "\n".join(p.text for p in d.paragraphs)

    if file.filename.endswith(".pdf"):
        out = []
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                if p.extract_text():
                    out.append(p.extract_text())
        return "\n".join(out)
    return ""

def load_skills():
    s = pd.read_excel("skills_onet.xlsx")
    t = pd.read_excel("Technology Skills.xlsx")
    s = s[(s["Scale Name"] == "Importance") & (s["Data Value"] >= 3)]
    return set(s["Element Name"].str.lower()).union(set(t["Example"].str.lower()))

SKILL_DB = load_skills()

def score_resume(resume, jd):
    rc, jc = clean_text(resume), clean_text(jd)

    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform([rc, jc])
    tfidf = cosine_similarity(mat[0:1], mat[1:2])[0][0] * 100

    semantic = sbert_similarity(resume, jd)
    final = 0.6 * semantic + 0.4 * tfidf

    matched = sorted(set(rc.split()).intersection(SKILL_DB))
    missing = sorted(set(jc.split()).intersection(SKILL_DB) - set(matched))

    return round(tfidf,2), round(semantic,2), round(final,2), matched, missing

def resume_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, 15)
    pdf.set_font("Arial", size=11)
    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)
    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# ===================== Routes =====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        users.insert_one({
            "username": request.form["username"],
            "password": generate_password_hash(request.form["password"]),
            "role": request.form["role"]
        })
        return redirect("/login")
    return render_template("signup.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = users.find_one({"username": request.form["username"]})
        if u and check_password_hash(u["password"], request.form["password"]):
            session["user"] = u["username"]
            session["role"] = u["role"]
            return redirect(f"/{u['role']}")
    return render_template("login.html")

# ===================== HR =====================
@app.route("/hr")
def hr_dashboard():
    jd = jds.find_one()
    return render_template("hr_dashboard.html", jd_exists=bool(jd))

@app.route("/hr/upload_jd", methods=["POST"])
def upload_jd():
    jds.delete_many({})
    jd_text = load_file(request.files["jd"])
    jds.insert_one({"jd": jd_text})
    return redirect("/hr")

@app.route("/hr/delete_jd")
def delete_jd():
    jds.delete_many({})
    return redirect("/hr")

@app.route("/hr/analyze", methods=["POST"])
def hr_analyze():
    jd_doc = jds.find_one()
    if not jd_doc:
        return render_template("no_jd.html")

    results = []
    for f in request.files.getlist("resumes")[:20]:
        resume = load_file(f)
        _, _, final, _, _ = score_resume(resume, jd_doc["jd"])
        results.append((f.filename, final, "YES" if final >= 50 else "NO"))

    results.sort(key=lambda x: x[1], reverse=True)
    return render_template("hr_results.html", results=results)

# ===================== Employee =====================
@app.route("/employee")
def employee_dashboard():
    return render_template("employee_dashboard.html")

@app.route("/employee/upload", methods=["GET","POST"])
def employee_upload():
    if request.method == "POST":
        text = load_file(request.files["resume"])
        session["resume"] = text
        return render_template("employee_preview.html", resume=text)
    return render_template("employee_upload.html")

@app.route("/employee/create", methods=["GET","POST"])
def employee_create():
    if request.method == "POST":
        resume = f"""
{request.form['name']}
{request.form['email']} | {request.form['phone']}

PROFESSIONAL SUMMARY
{request.form['summary']}

SKILLS
{request.form['skills']}

EXPERIENCE
{request.form['experience']}

PROJECTS
{request.form['projects']}

EDUCATION
{request.form['education']}

CERTIFICATIONS
{request.form['certifications']}
"""
        session["resume"] = resume
        return render_template("employee_preview.html", resume=resume)
    return render_template("employee_create.html")

def resume_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)

    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return io.BytesIO(pdf_bytes)


@app.route("/employee/download")
def download_resume():
    resume_text = session.get("resume")
    if not resume_text:
        return "No resume found", 400

    pdf_stream = resume_to_pdf(resume_text)
    return send_file(
        pdf_stream,
        as_attachment=True,
        download_name="Resume.pdf",
        mimetype="application/pdf"
    )

@app.route("/employee/check")
def employee_check():
    jd_doc = jds.find_one()
    if not jd_doc:
        return render_template("no_jd.html")

    resume = session.get("resume")
    t,s,f,matched,missing = score_resume(resume, jd_doc["jd"])
    return render_template("employee_result.html",
        tfidf=t, semantic=s, final=f, matched=matched, missing=missing)

if __name__ == "__main__":
    app.run(debug=True)
