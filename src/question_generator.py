from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load fine-tuned question generation model
tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")

def generate_questions(contexts, num_questions=5):
    """
    Generate meaningful questions using highlight-style prompts for each chunk.
    """
    prompts = []

    for ctx in contexts:
        # Split into sentences
        sentences = ctx.strip().split(". ")
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) >= 6 and len(prompts) < num_questions:
                words = sentence.split()
                # Add <hl> highlight tags
                highlighted = f"<hl> {' '.join(words[:5])} <hl> {' '.join(words[5:])}"
                prompt = f"generate question: {highlighted}"
                prompts.append(prompt)
        if len(prompts) >= num_questions:
            break

    # Tokenize and generate
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=64)
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
