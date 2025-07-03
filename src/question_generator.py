import requests
import os

API_URL = "https://api-inference.huggingface.co/models/t5-base"
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

def query_huggingface(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    try:
        return response.json()[0]["generated_text"]
    except Exception:
        return "❌ Failed to generate question."

def generate_questions(contexts, num_questions=5):
    questions = []
    for ctx in contexts[:num_questions]:
        prompt = f"Generate a question based on the following passage:\n{ctx.strip()}"
        question = query_huggingface(prompt)
        questions.append(question.strip())
    return questions
