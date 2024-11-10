import os
import streamlit as st
from document_processing import extract_text_from_pdf
from generate_embeddings import generate_embeddings, store_embeddings
from search_engine import search_index
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")

# Set padding token if it's not defined
if tokenizer_gpt2.pad_token is None:
    tokenizer_gpt2.add_special_tokens({'pad_token': '[PAD]'})

model_gpt2.config.pad_token_id = tokenizer_gpt2.pad_token_id

# Initialize session state for conversation history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Custom CSS for chatbot interface with scrollable history
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column-reverse;
        max-height: 400px;
        overflow-y: scroll;
        padding: 10px;
    }
    .chat-bubble {
        max-width: 70%;
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
        color: white;
        font-size: 16px;
    }
    .user-bubble {
        background-color: #0084ff;
        margin-left: auto;
    }
    .bot-bubble {
        background-color: #444654;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("AI Content Engine ChatBot")

st.write("""
    This AI-powered bot allows you to ask questions about the Form 10-K filings of Google, Tesla, and Uber.
    Query the documents and generate additional insights using the chatbot interface.
""")

# Load document texts
google_10k_text = extract_text_from_pdf("data/goog-10-k-2023 (1).pdf")
tesla_10k_text = extract_text_from_pdf("data/tsla-20231231-gen.pdf")
uber_10k_text = extract_text_from_pdf("data/uber-10-k-2023.pdf")

documents = {
    "Google 10-K": google_10k_text,
    "Tesla 10-K": tesla_10k_text,
    "Uber 10-K": uber_10k_text
}

def generate_insights(query):
    inputs = tokenizer_gpt2(query, return_tensors='pt', padding=True, truncation=True)

    outputs = model_gpt2.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=100,
        pad_token_id=tokenizer_gpt2.pad_token_id,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer_gpt2.decode(outputs[0], skip_special_tokens=True)

def handle_query(query):
    indices = search_index(query)
    result = []
    relevant_documents = []

    if 0 in indices:
        result.append("Found relevant info in **Google 10-K**")
        relevant_documents.append(('Google 10-K', google_10k_text))
    if 1 in indices:
        result.append("Found relevant info in **Tesla 10-K**")
        relevant_documents.append(('Tesla 10-K', tesla_10k_text))
    if 2 in indices:
        result.append("Found relevant info in **Uber 10-K**")
        relevant_documents.append(('Uber 10-K', uber_10k_text))

    if not result:
        result.append("No relevant documents found, but we generated some insights.")

    generated_insight = generate_insights(query)
    result.append(f"Generated Insights: {generated_insight}")
    
    return result, relevant_documents

def find_relevant_snippet(query, document_text):
    query_words = query.lower().split()
    lines = document_text.splitlines()
    
    for line in lines:
        if all(word in line.lower() for word in query_words):
            return line
    
    return "No relevant information found."

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state['messages']:
    if message['user'] == 'bot':
        st.markdown(f'<div class="chat-bubble bot-bubble">{message["text"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble user-bubble">{message["text"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

user_message = st.text_input("Enter your question:")

if user_message:
    st.session_state['messages'].append({'user': 'user', 'text': user_message})
    
    bot_response, relevant_docs = handle_query(user_message)
    
    for response in bot_response:
        st.session_state['messages'].append({'user': 'bot', 'text': response})

if st.checkbox('Show relevant part from the document'):
    st.write("### Relevant Snippets from the Documents")
    
    for doc_name, doc_text in relevant_docs:
        relevant_snippet = find_relevant_snippet(user_message, doc_text)
        st.write(f"### {doc_name}")
        st.write(relevant_snippet)
