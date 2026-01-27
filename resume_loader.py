import pdfplumber
import docx

# -----------------------------
# TXT FILE
# -----------------------------
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# -----------------------------
# DOCX FILE
# -----------------------------
def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# -----------------------------
# PDF FILE
# -----------------------------
def read_pdf(file_path):
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

# -----------------------------
# UNIVERSAL LOADER
# -----------------------------
def load_resume(file_path):
    if file_path.endswith(".txt"):
        return read_txt(file_path)
    elif file_path.endswith(".docx"):
        return read_docx(file_path)
    elif file_path.endswith(".pdf"):
        return read_pdf(file_path)
    else:
        raise ValueError("Unsupported file format")
