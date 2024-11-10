from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pathlib import Path
import fitz  # PyMuPDF

pdf_dir = Path("data")

embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def load_pdfs(directory):
    documents = []
    for pdf_file in directory.glob("*.pdf"):
        try:
            with fitz.open(pdf_file) as pdf:
                text = ""
                for page in pdf:
                    text += page.get_text()
            documents.append(Document(page_content=text, metadata={"source": pdf_file.name}))
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
    return documents

documents = load_pdfs(pdf_dir)
faiss_index = FAISS.from_documents(documents, embeddings_model)

faiss_index.save_local("company_index.faiss")
