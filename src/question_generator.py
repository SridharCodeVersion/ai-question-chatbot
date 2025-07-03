from Questgen import main

qe = main.BoolQGen()
mcq = main.MCQGen()

def generate_questions(contexts, num_questions=5):
    """
    Generate mixed-style questions (boolean + MCQ) from contexts.
    """
    text = " ".join(contexts)
    # Generate boolean (Yes/No) questions
    bool_output = qe.predict_boolq({"input_text": text})
    # Generate MCQs
    mcq_output = mcq.predict_mcq({"input_text": text})

    questions = []
    for q in bool_output.get("Boolean Questions", []):
        questions.append(q)
        if len(questions) >= num_questions: break

    for qdict in mcq_output.get("questions", []):
        qtext = qdict.get("question")
        questions.append(qtext)
        if len(questions) >= num_questions: break

    # Fallback: simple sentence prompts if not enough
    while len(questions) < num_questions:
        questions.append("Explain the concept: " + contexts[0][:50] + "...")

    return questions[:num_questions]
