from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from pathlib import Path

# Load embedding model (all-MiniLM-L6-v2 for better performance)
embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load Faiss index using LangChain's FAISS wrapper
def search_index(query):
    index_path = Path(r"C:\Users\adity\OneDrive\Desktop\Alemeno Internship Assignment\company_index.faiss")
    print(f"Attempting to load FAISS index from: {index_path}")
    
    try:
        faiss_index = FAISS.load_local(str(index_path), embeddings_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        return []

    retriever = faiss_index.as_retriever()

    try:
        relevant_documents = retriever.get_relevant_documents(query)
    except Exception as e:
        print(f"Error during search: {e}")
        return []

    return [doc.metadata['source'] for doc in relevant_documents]
