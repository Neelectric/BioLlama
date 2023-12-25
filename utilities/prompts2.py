# Part of the BioLlama library
# Written by Neel Rajani
# Utility methods for prompt engineering, culminating in "promptify"

def system_prompt():
    return "You are an excellently helpful AI assistant that answers biomedical questions."

def retrieval_augmentation(chunks):
    output = "The following chunks were retrieved from biomedical literature to help you.\n"
    for chunk in chunks:
        output += chunk
    return output + "\n"

def few_shot(benchmark):
    format_string = "You start all of your responses with <ANSWER> and end them with </ANSWER>, as shown in the following example: "
    if benchmark == "bioASQ_no_snippet":
        example = "<QUESTION>Orteronel was developed for treatment of which cancer?</QUESTION>\n<ANSWER> castration-resistant prostate cancer</ANSWER>\nYou must now answer the following biomedical question AS SUCCINCTLY AS YOU CAN. Do not use more than 5 words."
    elif benchmark == "MedQA":
        example = "<QUESTION>A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? \n(A) Ampicillin \n(B) Ceftriaxone \n(C) Ciprofloxacin \n(D) Doxycycline \n(E) Nitrofurantoin</QUESTION>\n<ANSWER> (E) Nitrofurantoin</ANSWER>\nSelect the correct choice for the following question."
    elif benchmark == "PubMedQA":
        example = "<QUESTION>Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?</QUESTION>\n<ANSWER> Yes</ANSWER>\nDo not justify your response, respond with only Yes, No or Maybe."
    elif benchmark == "MedMCQA":
        example = "<QUESTION> Which of the following is not true for myelinated nerve fibers: \n(1) Impulse through myelinated fibers is slower than non-myelinated fibers \n(2) Membrane currents are generated at nodes of Ranvier \n(3) Saltatory conduction of impulses is seen \n(4) Local anesthesia is effective only when the nerve is not covered by myelin sheath</QUESTION>\n<ANSWER> 3</ANSWER>\nSelect the correct choice for the following question. State nothing other than the index of the correct choice, without brackets."
    return format_string + "\n" + example + "\n<QUESTION>"

def promptify(benchmark, question, retrieval_mode = None, retrieved_chunks = None):
    promptified = system_prompt()
    if retrieval_mode != None:
        promptified += retrieval_augmentation(retrieved_chunks)
    promptified += few_shot(benchmark)
    promptified += question
    promptified += "</QUESTION>\n<ANSWER> "
    return promptified

def promptify_for_judging(question, true_answer, model_response):
    promptified = (
        "<question>What is the mode of inheritance of Wilson's disease?</question>\n<marking_scheme>autosomal recessive</marking_scheme>\n<student_response>Autosomal recessive disorder</student_response>\n<judgement>The student response is correct</judging>\n<question>What is the structural fold of bromodomain proteins?</question>\n<marking_scheme>All-alpha-helical fold</marking_scheme>\n<student_response>Beta-alpha-beta structural fold.</student_response>\n<judging>The student response is incorrect</judging>\n<question>"
        + question
        + "</question>\n<marking_scheme>"
        + true_answer
        + "</marking_scheme>\n<student_response>"
        + model_response
        + "</student_response>\n<judging>The student response is"
    )
    return promptified