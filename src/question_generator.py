from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("flan-t5-small")

def generate_questions(contexts, num_questions=5):
    inputs = [
        "generate question: " + ctx.strip().replace("\n", " ") 
        for ctx in contexts
    ]
    batch = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    outs = model.generate(**batch, max_length=64, num_return_sequences=1)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outs]