import os
import sys
import streamlit as st

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from src.data_loader import load_pdf_text
from src.utils import save_uploaded_file
from src.retriever import chunk_text, build_index, retrieve
from src.question_generator import generate_questions

# Streamlit UI config
st.set_page_config(page_title="AI Question Generator", layout="centered")
st.title("📘 AI-Based Question Generator for JEE/BITSAT")
st.markdown("Upload a PDF to automatically generate 5 smart and meaningful questions from your document.")

# File upload
uploaded = st.file_uploader("🔼 Upload your PDF", type=["pdf"])

if st.button("🚀 Generate Questions"):
    if not uploaded:
        st.error("⚠️ Please upload a PDF document first.")
    else:
        # Step 1: Load text
        file_path = save_uploaded_file(uploaded)
        text = load_pdf_text(file_path)

        # Step 2: Chunk and retrieve
        chunks = chunk_text(text)
        index, embeddings = build_index(chunks)
        top_chunks = retrieve(chunks, index, embeddings, query="general topic", top_k=3)

        # Optional: Show retrieved text chunks
        st.subheader("🔍 Retrieved Contexts")
        for i, chunk in enumerate(top_chunks, 1):
            st.code(f"Chunk {i}:\n{chunk}")

        # Step 3: Generate questions
        st.subheader("📄 Generated Questions")
        questions = generate_questions(top_chunks)
        for i, q in enumerate(questions, 1):
            st.markdown(f"**Q{i}.** {q}")
