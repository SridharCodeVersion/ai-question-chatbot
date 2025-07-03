from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load fine-tuned question generation model
tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")

def generate_questions(contexts, num_questions=5):
    inputs = []

    for ctx in contexts:
        # Try to highlight a keyword before the dash
        split = ctx.split(" - ")
        if len(split) >= 2:
            highlighted = f"<hl> {split[0].strip()} <hl> - {split[1].strip()}"
        else:
            highlighted = f"<hl> {ctx.strip()} <hl>"

        inputs.append(f"generate question: {highlighted}")

    # Tokenize and generate questions
    batch = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**batch, max_length=64)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
