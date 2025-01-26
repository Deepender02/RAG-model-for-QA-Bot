#Financial Question Answering Bot Using RAG
An interactive Retrieval-Augmented Generation (RAG) model designed to answer financial queries from Profit & Loss (P&L) statements extracted from PDF documents. The bot enables efficient data extraction, real-time query handling, and generative response creation for insightful decision-making.

#Features
PDF Parsing and Table Extraction:
Extracts structured data (e.g., P&L tables) from uploaded PDF files using tabula.

#Context Retrieval with Vector Embeddings:
Utilizes SentenceTransformer to encode financial data and retrieves the most relevant context for user queries.

#Generative Answer Creation:
Provides coherent, context-aware responses using a T5 model from Hugging Face.

#Interactive Web Interface:
Built with Streamlit, allowing users to upload documents, input queries, and view results in real-time.

#How It Works
Upload a Financial Document:
Upload a PDF containing financial data (e.g., P&L statements).

Data Extraction:
Automatically extracts tables, identifies the P&L table, and preprocesses it into embeddings.

Ask a Financial Query:
Enter a query like:

"What is the total revenue for 2024?"
"How do net income and operating expenses compare for Q1 2024?"
Context Retrieval and Response Generation:

Retrieves relevant rows from the P&L table using cosine similarity.
Generates a human-readable answer using a T5 model.

#Technologies Used
Python Libraries:
Streamlit: For creating an interactive web interface.
SentenceTransformer: For generating vector embeddings of financial data.
transformers: Hugging Face library for text generation (T5-small).
tabula: For extracting tables from PDF files.
Machine Learning:
RAG Framework: Combines retrieval and generation for high-quality answers.
Deployment:
Compatible with Streamlit Community Cloud for hosting and sharing.
