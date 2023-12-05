def system_prompt():
    return "You are an excellently helpful AI assistant that answers biomedical questions."

def retrieval_augmentation(chunks):
    output = "The following chunks were retrieved from biomedical literature to help you."
    for chunk in chunks:
        output += chunk
    return output

def few_shot(benchmark):
    format_string = "You start all of your responses with <ANSWER> and end them with </ANSWER>, as shown in the following example: "
    if benchmark == "BioASQ5b":
        example = "<QUESTION>Orteronel was developed for treatment of which cancer?</QUESTION><ANSWER> castration-resistant prostate cancer</ANSWER> You must now answer the following biomedical question AS SUCCINCTLY AS YOU CAN. Do not use more than 5 words. <QUESTION>"
    elif benchmark == "MedQA":
        example = "<QUESTION>A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? (A) Ampicillin (B) Ceftriaxone (C) Ciprofloxacin (D) Doxycycline (E) Nitrofurantoin</QUESTION><ANSWER> (E) Nitrofurantoin</ANSWER> Select the correct choice for the following question. <QUESTION>"

    elif benchmark == "PubMedQA":
        example = "<QUESTION>Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?</QUESTION><ANSWER> Yes</ANSWER>  Do not justify your response, respond with only Yes, No or Maybe. <QUESTION>"

    elif benchmark == "MedMCQA":
        example = "<QUESTION> Which of the following is not true for myelinated nerve fibers: (1) Impulse through myelinated fibers is slower than non-myelinated fibers (2) Membrane currents are generated at nodes of Ranvier (3) Saltatory conduction of impulses is seen (4) Local anesthesia is effective only when the nerve is not covered by myelin sheath</QUESTION><ANSWER>3</ANSWER> Select the correct choice for the following question. State nothing other than the index of the correct choice, without brackets. <QUESTION>"

    return format_string + example

def promptify(retrieval, retrieved_chunks, benchmark, question):
    promptified = system_prompt()
    if retrieval:
        promptified += retrieval_augmentation(retrieved_chunks)
    promptified += few_shot(benchmark)
    promptified += question
    promptified += "</QUESTION><ANSWER> "
    return promptified
