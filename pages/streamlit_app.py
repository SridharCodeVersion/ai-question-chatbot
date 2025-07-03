import os
import sys
import streamlit as st

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_pdf_text
from src.utils import save_uploaded_file
from src.retriever import chunk_text, build_index, retrieve
from src.question_generator import generate_questions

# Streamlit UI setup
st.set_page_config(page_title="📘 AI Question Generator", layout="centered")
st.title("📘 AI-Based Question Generator")
st.markdown("Upload a PDF file to generate meaningful questions using AI.")

# File uploader
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if st.button("🚀 Generate Questions"):
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF file first.")
    else:
        # Step 1: Save & extract text
        filepath = save_uploaded_file(uploaded_file)
        full_text = load_pdf_text(filepath)

        # Step 2: Chunk & index
        chunks = chunk_text(full_text)
        index, embeddings = build_index(chunks)
        top_chunks = retrieve(chunks, index, embeddings, query="important concepts", top_k=3)

        # Step 3: Show retrieved content
        st.subheader("🔍 Retrieved Contexts")
        for i, chunk in enumerate(top_chunks, start=1):
            st.code(f"[Chunk {i}]:\n{chunk[:400]}...")

        # Step 4: Generate and display questions
        st.subheader("📄 Generated Questions")
        questions = generate_questions(top_chunks, num_questions=5)
        for i, q in enumerate(questions, 1):
            st.markdown(f"**Q{i}.** {q}")
