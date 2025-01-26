# Required Libraries
import streamlit as st
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from tabula import read_pdf

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="t5-small")

# Function to extract P&L tables from PDFs
def extract_pnl_table(pdf_path):
    try:
        tables = read_pdf(pdf_path, pages="all", multiple_tables=True)
        if not tables:
            st.warning("No tables were extracted from the document.")
            return None
        
        st.write("Extracted Tables:")
        for i, table in enumerate(tables):
            st.write(f"Table {i}:")
            st.dataframe(table)  # Display each extracted table
            
            # Check if this table could be a P&L table
            if any(keyword in table.columns for keyword in ["Revenue", "Income", "Expenses", "Profit"]):
                st.write("Identified a P&L table.")
                return table
        
        st.warning("No valid P&L table found in the document.")
    except Exception as e:
        st.error(f"Error extracting table: {e}")
    return None


# Function to generate embeddings from a P&L table
def generate_embeddings(pnl_table):
    embeddings = {}
    descriptive_column = pnl_table.columns[0]
    for _, row in pnl_table.iterrows():
        for col in pnl_table.columns[1:]:
            if pd.notna(row[col]):
                key = f"{row[descriptive_column]}: {col}"
                value = row[col]
                text = f"{key}: {value}"
                embeddings[key] = embedding_model.encode(text)
    return embeddings

# Function to retrieve relevant rows based on a query
def retrieve(query, embeddings, top_k=5):
    query_vector = embedding_model.encode(query)
    similarities = {
        key: cosine_similarity([query_vector], [embedding])[0][0]
        for key, embedding in embeddings.items()
    }
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:top_k]

# Function to generate answers
def generate_response(query, embeddings):
    retrieved = retrieve(query, embeddings)
    context = "\n".join([f"{item[0]}: {item[1]}" for item in retrieved])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = generator(prompt)
    return response[0]["generated_text"], retrieved

# Streamlit Interface
def main():
    st.title("Interactive Financial QA Bot")
    st.write("Upload a PDF document containing a Profit & Loss statement and ask financial queries.")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    query = st.text_input("Enter your financial query")

    if uploaded_file and query:
        with st.spinner("Processing the uploaded PDF..."):
            pdf_path = f"./temp_{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            
            pnl_table = extract_pnl_table(pdf_path)
            if pnl_table is not None:
                st.write("Extracted P&L Table:")
                st.dataframe(pnl_table)

                embeddings = generate_embeddings(pnl_table)

                with st.spinner("Generating response..."):
                    response, retrieved_context = generate_response(query, embeddings)

                st.subheader("Answer:")
                st.write(response)

                st.subheader("Retrieved Context:")
                for item in retrieved_context:
                    st.write(f"**{item[0]}**: Similarity Score = {item[1]:.2f}")
            else:
                st.warning("Could not find a valid P&L table in the PDF.")
    elif not uploaded_file:
        st.info("Please upload a PDF file to get started.")
    elif not query:
        st.info("Please enter a financial query to retrieve answers.")

if __name__ == "__main__":
    main()
