from PyPDF2 import PdfReader

def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    texts = [page.extract_text() for page in reader.pages]
    return "\n".join(texts)