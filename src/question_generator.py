from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load fine-tuned question generation model
tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")

def generate_questions(contexts, num_questions=5):
    """
    Generate relevant questions from provided contexts using a highlight-style prompt format.
    """
    inputs = []

    for ctx in contexts:
        # Extract a key phrase (before dash or first sentence)
        split = ctx.split(" - ")
        if len(split) >= 2:
            # Highlight the key phrase (important for the model to generate questions)
            highlighted = "<hl> " + split[0].strip() + " <hl> - " + split[1].strip()
        else:
            # Fallback highlighting if no dash
            highlighted = "<hl> " + ctx.strip() + " <hl>"

        # Add prompt prefix
        prompt = f"generate question: {highlighted}"
        inputs.append(prompt)

    # Tokenize all prompts
    batch = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)

    # Generate questions using the model
    outputs = model.generate(**batch, max_length=64, num_return_sequences=1)

    # Decode and return all generated questions
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
