from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def generate_questions(contexts, num_questions=5):
    inputs = [
    f"Generate a logical and relevant question from the following passage:\n\n{ctx.strip()}"
    for ctx in contexts
]
    batch = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    outs = model.generate(**batch, max_length=64, num_return_sequences=1)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outs]
