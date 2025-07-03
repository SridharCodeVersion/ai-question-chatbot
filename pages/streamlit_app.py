import os
import sys
import streamlit as st

# Include src/ in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import internal modules
from src.data_loader import load_pdf_text
from src.utils import save_uploaded_file
from src.retriever import chunk_text, build_index, retrieve
from src.question_generator import generate_questions

# Streamlit config
st.set_page_config(page_title="AI Question Generator", layout="centered")
st.title("📘 AI-Based Question Generator for JEE/BITSAT")
st.markdown("Upload any chapter or topic PDF to automatically generate **meaningful questions** using AI.")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if st.button("🚀 Generate Questions"):
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF file.")
    else:
        # Step 1: Save + Load PDF
        file_path = save_uploaded_file(uploaded_file)
        full_text = load_pdf_text(file_path)

        # Step 2: Chunking + Indexing
        chunks = chunk_text(full_text)
        index, embeddings = build_index(chunks)

        # Step 3: Retrieve top relevant chunks using generic query
        top_chunks = retrieve(chunks, index, embeddings, query="important concepts", top_k=3)

        # Optional debug view
        st.subheader("📌 Retrieved Chunks")
        for i, chunk in enumerate(top_chunks, 1):
            st.code(f"[Chunk {i}]\n{chunk[:500]}...")

        # Step 4: Generate Questions
        st.subheader("🧠 AI-Generated Questions")
        questions = generate_questions(top_chunks, num_questions=5)

        for i, q in enumerate(questions, 1):
            st.markdown(f"**Q{i}.** {q}")
