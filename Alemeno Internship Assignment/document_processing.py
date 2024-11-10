from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    text_chunks = split_text_into_chunks(text)
    
    return text_chunks

def split_text_into_chunks(text, chunk_size=500):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks
