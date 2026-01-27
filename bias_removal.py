import re

def remove_pii(text: str) -> str:
    text = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "REDACTED_NAME", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "REDACTED_EMAIL", text)
    text = re.sub(r"\b\d{10}\b", "REDACTED_PHONE", text)
    return text
