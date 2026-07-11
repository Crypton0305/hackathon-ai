"""
pdf_extract.py

Handles Modality 2: PDF claim documents.

Steps:
1. Read text out of the uploaded PDF (using pdfplumber).
2. Use simple regex patterns to pull out the fields we care about:
   Policy Number, Claim Number, Customer Name, Incident Details.

This is a simple rule-based extractor (regex), not a heavy NLP model.
It works well for structured claim forms and is easy to adjust later
if the document format changes.
"""

import re
import pdfplumber


def extract_text_from_pdf(pdf_file):
    """Reads all text from a PDF file (accepts a file path or file-like object)."""
    full_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text


def find_field(pattern, text, default="Not found"):
    """Helper: search text with a regex pattern and return the first group."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return default


def extract_claim_fields(text):
    """
    Pulls the key claim fields out of raw PDF text using regex patterns.
    Patterns are written loosely so they match common label variations
    like "Policy No:", "Policy Number:", "Policy #:" etc.
    """
    fields = {
        "policy_number": find_field(r"policy\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Za-z0-9\-]+)", text),
        "claim_number": find_field(r"claim\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Za-z0-9\-]+)", text),
        "customer_name": find_field(r"(?:customer|policy\s*holder|insured)\s*name\s*[:\-]?[ \t]*([^\n]+)", text),
        "incident_date": find_field(r"(?:incident|accident)\s*date\s*[:\-]?\s*([0-9/\-\.]+)", text),
        "incident_details": find_field(
            r"(?:incident|accident)\s*details?\s*[:\-]?\s*(.+)", text, default="Not found"
        ),
    }
    return fields
