import requests
import os

# Hugging Face Model Endpoint
API_URL = "https://api-inference.huggingface.co/models/t5-base"

# Secret token must be set in Streamlit Cloud → App settings → Secrets
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

def query_huggingface(prompt):
    """Send prompt to Hugging Face Inference API and get response."""
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    try:
        return response.json()[0]["generated_text"]
    except Exception as e:
        return "Failed to generate question."

def generate_questions(contexts, num_questions=5):
    """Generate questions from context chunks."""
    questions = []
    for ctx in contexts[:num_questions]:
        prompt = f"Generate a question based on the following passage:\n{ctx.strip()}"
        question = query_huggingface(prompt)
        questions.append(question.strip())
    return questions
