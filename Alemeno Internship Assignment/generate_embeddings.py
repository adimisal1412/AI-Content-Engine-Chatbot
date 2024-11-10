from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts, batch_size=32):
    embeddings = []
    try:
        for i in range(0, len(texts), batch_size):
            batch_embeddings = model.encode(texts[i:i+batch_size], show_progress_bar=True)
            embeddings.extend(batch_embeddings)
    except Exception as e:
        print(f"Error during embedding generation: {e}")
    return np.array(embeddings)

def store_embeddings(embeddings, index_file="company_index.faiss"):
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, 100)
    
    if not index.is_trained:
        print("Training Faiss index...")
        index.train(embeddings)
    
    index.add(embeddings)
    faiss.write_index(index, index_file)
    print(f"Faiss index saved as {index_file}")
    return index

def load_faiss_index(index_file="company_index.faiss"):
    try:
        index = faiss.read_index(index_file)
        print(f"Faiss index loaded from {index_file}")
        return index
    except Exception as e:
        print(f"Error loading Faiss index: {e}")
        return None

def search_documents(index, query_text, top_k=5):
    query_embedding = model.encode([query_text])
    distances, indices = index.search(query_embedding, top_k)
    
    return distances, indices
