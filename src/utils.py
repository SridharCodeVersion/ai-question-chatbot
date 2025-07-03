import os

def save_uploaded_file(uploaded, path="data/input.pdf"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path