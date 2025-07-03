from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")

def generate_questions(contexts, num_questions=5):
    inputs = [
    f"generate question: context: {ctx.strip().replace('\\n', ' ')}"
    for ctx in contexts
]

    batch = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    outs = model.generate(**batch, max_length=64, num_return_sequences=1)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outs]
