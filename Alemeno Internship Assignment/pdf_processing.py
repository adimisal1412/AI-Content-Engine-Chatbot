# pdf_processing.py
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from the given PDF."""
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text("text")
    return text
