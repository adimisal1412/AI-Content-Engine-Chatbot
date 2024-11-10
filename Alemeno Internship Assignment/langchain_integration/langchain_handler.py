from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import GPT2LMHeadModel, GPT2Tokenizer

embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_index = FAISS.load_local("company_index.faiss", embeddings_model)

model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")

def process_query_with_langchain(query, documents):
    retriever = RetrievalQA.from_chain_type(llm=model_gpt2, retriever=faiss_index.as_retriever())
    result = retriever.run(query)
    relevant_docs = [doc.metadata['source'] for doc in documents]
    return result, relevant_docs
