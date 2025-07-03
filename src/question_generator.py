from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load fine-tuned question generation model
tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")

def generate_questions(contexts, num_questions=5):
    prompts = []

    for ctx in contexts:
        # Break context into multiple sentences
        sentences = ctx.strip().split(". ")
        for sentence in sentences:
            sentence = sentence.strip().replace("\n", " ")
            if len(sentence.split()) < 6:
                continue
            words = sentence.split()
            # Highlight the first 4-5 words
            highlighted = f"<hl> {' '.join(words[:5])} <hl> {' '.join(words[5:])}"
            prompt = f"generate question: {highlighted}"
            prompts.append(prompt)
            if len(prompts) == num_questions:
                break
        if len(prompts) == num_questions:
            break

    # Tokenize and generate questions
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=64)

    # Decode results
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
