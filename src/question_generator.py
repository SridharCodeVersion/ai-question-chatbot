from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load T5 model
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

def generate_questions(contexts, num_questions=5):
    """
    Generate questions using T5 with instruction prompts.
    """
    prompts = []
    for ctx in contexts:
        # Limit context length
        trimmed_ctx = " ".join(ctx.strip().split()[:60])
        prompt = f"Generate a question based on the following passage: {trimmed_ctx}"
        prompts.append(prompt)

    prompts = prompts[:num_questions]

    # Tokenize
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=64)

    # ✅ THIS is what you must return
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
