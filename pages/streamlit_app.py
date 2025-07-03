import os
import sys
import streamlit as st

# 📁 Add project root (parent of pages/) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ Now imports should work cleanly
from src.data_loader import load_pdf_text
from src.utils import save_uploaded_file
from src.retriever import chunk_text, build_index, retrieve
from src.question_generator import generate_questions

# ── Streamlit UI setup ─────────────────────────────────────────────
st.set_page_config(page_title="📘 AI Question Generator", layout="centered")
st.title("📘 AI-Based Question Generator")
st.markdown(
    "Upload a PDF 📄 to automatically generate meaningful questions using AI!"
)

# ── File uploader ───────────────────────────────────────────────────
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if st.button("🚀 Generate Questions"):
    if not uploaded_file:
        st.error("⚠️ Please upload a PDF first.")
    else:
        # ✅ Save & extract text
        filepath = save_uploaded_file(uploaded_file)
        full_text = load_pdf_text(filepath)

        # ✅ Chunking and retrieval
        chunks = chunk_text(full_text)
        index, embeddings = build_index(chunks)
        top_chunks = retrieve(chunks, index, embeddings, query="important concepts", top_k=3)

        # ✅ Optional: display retrieved chunks
        st.subheader("🔍 Retrieved Contexts")
        for i, chunk in enumerate(top_chunks, start=1):
            st.code(f"Chunk {i}:\n{chunk[:300]}...")

        # ✅ Generate questions
        st.subheader("🧠 AI-Generated Questions")
        questions = generate_questions(top_chunks, num_questions=5)
        for i, q in enumerate(questions, start=1):
            st.markdown(f"**Q{i}.** {q}")
