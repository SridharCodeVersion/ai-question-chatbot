from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

def generate_questions(contexts, num_questions=5):
    """
    Generate questions using general-purpose T5 model with prompt engineering.
    """
    prompts = []
    for ctx in contexts:
        # Use instruction format T5 understands well
        prompts.append(f"Generate a question based on the following passage:\n{ctx.strip()}")

    # Truncate to the requested number of questions
    prompts = prompts[:num_questions]

    # Tokenize prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=64)

    # Decode and return generated questions
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
