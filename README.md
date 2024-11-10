# AI-Content-Engine-Chatbot
AI Content Engine
Overview
This repository contains the code and resources for an AI Content Engine that processes documents, generates embeddings, stores them in a FAISS index, and allows users to query the documents through a Streamlit chatbot interface. The system is designed to be modular, scalable, and locally hosted to ensure data privacy.

Features
Document Processing: Extract text from PDF documents and convert it into embeddings using a pre-trained model.
Vector Store Ingestion: Store document embeddings in a FAISS index for efficient retrieval.
Query Engine: Search for relevant documents based on a user query and generate insights using a locally running GPT-2 model.
Chatbot Interface: A Streamlit-based interface for users to query the system and receive insights from relevant documents.
Project Structure
graphql
Copy code
├── data/                    # Directory for storing PDF documents
├── document_processing.py    # Script for extracting text from PDF files
├── generate_embeddings.py    # Script for generating embeddings using a pre-trained model
├── generate_faiss_index.py   # Script for creating and storing the FAISS index
├── search_engine.py          # Script for querying the FAISS index
├── streamlit_app.py          # Streamlit app for user interaction and chatbot interface
├── README.md                 # Project documentation (this file)
├── requirements.txt          # Required Python packages
Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/ai-content-engine.git
cd ai-content-engine
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Download the pre-trained model:

The system uses all-MiniLM-L6-v2 for embedding generation.
The GPT-2 model is used for insight generation. These models will be automatically downloaded when running the scripts for the first time.
Usage
1. Document Processing
Place your PDF documents in the data/ directory. The document_processing.py script will process the PDFs and extract text content.

2. Generate Embeddings
Run the generate_embeddings.py script to generate embeddings for the processed documents:

bash
Copy code
python generate_embeddings.py
3. Create FAISS Index
Use the generate_faiss_index.py script to create a FAISS index from the embeddings:

bash
Copy code
python generate_faiss_index.py
4. Query the System
Use the search_engine.py script or interact with the Streamlit app to query the FAISS index. You can query the system to retrieve relevant documents and generate insights using GPT-2:

bash
Copy code
python streamlit_app.py
5. Running the Streamlit Chatbot Interface
Run the Streamlit app to access the chatbot interface:

bash
Copy code
streamlit run streamlit_app.py
The interface will allow you to:

Submit a query.
Retrieve the most relevant documents.
Get insights generated by the local GPT-2 model based on your query.
Example
Start the Streamlit App:
Go to the terminal and run:
bash
Copy code
streamlit run streamlit_app.py
Open the provided URL in your browser.
Upload PDF Documents:
The documents in the data/ folder are processed, and embeddings are generated automatically.
Submit Query:
Enter a query in the chatbot interface (e.g., "Summarize the key points in the document").
Receive a list of relevant documents along with insights generated by GPT-2.
Key Files and Scripts
document_processing.py: Extracts text from PDF documents.
generate_embeddings.py: Generates embeddings using the Sentence Transformer model.
generate_faiss_index.py: Creates and stores document embeddings in a FAISS index.
search_engine.py: Retrieves relevant documents from the FAISS index based on the user's query.
streamlit_app.py: Implements the chatbot interface using Streamlit for user interaction.
Requirements
Python 3.8+
Required Python packages are listed in requirements.txt.
Models Used
Sentence Transformer: all-MiniLM-L6-v2 for embedding generation.
GPT-2: For generating insights and completing text queries.
License
This project is licensed under the MIT License. See the LICENSE file for details.

For any queries or support, please contact adimisal1412@gmai.com.
