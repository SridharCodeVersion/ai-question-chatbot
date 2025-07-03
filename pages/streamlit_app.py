import sys
import os
import streamlit as st

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_pdf_text
from src.utils import save_uploaded_file
from src.retriever import chunk_text, build_index, retrieve
from src.question_generator import generate_questions

st.set_page_config(page_title="AI Question Generator", layout="centered")
st.title("📘 AI-Based Question Generator for JEE/BITSAT")
uploaded = st.file_uploader("📄 Upload Your PDF Document", type=["pdf"])
query = st.text_input("🔍 Enter Topic/Keywords to Focus On:", "")

if st.button("🚀 Generate Questions"):
    if not uploaded or not query:
        st.error("⚠️ Please upload a PDF and enter a topic/keyword.")
    else:
        path = save_uploaded_file(uploaded)
        text = load_pdf_text(path)
        chunks = chunk_text(text)
        index, embs = build_index(chunks)
        contexts = retrieve(chunks, index, embs, query, top_k=5)
        questions = generate_questions(contexts)
        st.success("✅ Questions Generated!")
        for i, q in enumerate(questions, 1):
            st.markdown(f"**Q{i}.** {q}")
