# Part of the BioLlama library
# Written by Neel Rajani
# Utility methods for prompt engineering, culminating in "promptify"

def system_prompt():
    return ""

def retrieval_augmentation(chunks):
    output = "The following chunks were retrieved from biomedical literature to help you:\n"
    for chunk in chunks:
        output += "\"" + chunk + "\",\n"
    return output

def zero_shotify(benchmark):
    if benchmark == "PubMedQA":
        format_string = "You start all of your responses with <ANSWER> and end them with </ANSWER>. Respond only with yes, no or maybe enclosed by the <ANSWER> </ANSWER> tags."
    else:
        format_string = "You start all of your responses with <ANSWER> and end them with </ANSWER>. Respond only with the correct answer enclosed in brackets followed by its answer text, as stated in the question."
    return format_string + "\n" + "<QUESTION>"

def few_shot(benchmark):
    format_string = "You start all of your responses with <ANSWER> and end them with </ANSWER>, as shown in the following example: "
    if benchmark == "bioASQ_no_snippet" or benchmark == "bioASQ_with_snippet":
        example = "<QUESTION>Orteronel was developed for treatment of which cancer?</QUESTION>\n<ANSWER> castration-resistant prostate cancer</ANSWER>\nYou must now answer the following biomedical question AS SUCCINCTLY AS YOU CAN. Do not use more than 5 words."
    elif benchmark == "MedQA-4":
        example = "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n (A) Inhibition of proteasome\n (B) Hyperstabilization of microtubules\n (C) Generation of free radicals\n (D) Cross-linking of DNA</QUESTION>\n<ANSWER> (D) Cross-linking of DNA</ANSWER>"
    elif benchmark == "MedQA-5":
        example = "<QUESTION>A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?\n (A) Ampicillin\n (B) Ceftriaxone\n (C) Ciprofloxacin\n (D) Doxycycline\n (E) Nitrofurantoin</QUESTION>\n<ANSWER> (E) Nitrofurantoin</ANSWER>"
    elif benchmark == "PubMedQA":
        example = "<QUESTION>Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?</QUESTION>\n<ANSWER> yes</ANSWER>\nDo not justify your response, respond with only yes, maybe or no.\n"
    elif benchmark == "MedMCQA":
        example = "<QUESTION> Which of the following is not true for myelinated nerve fibers: \n(1) Impulse through myelinated fibers is slower than non-myelinated fibers \n(2) Membrane currents are generated at nodes of Ranvier \n(3) Saltatory conduction of impulses is seen \n(4) Local anesthesia is effective only when the nerve is not covered by myelin sheath</QUESTION>\n<ANSWER> 3</ANSWER>\nSelect the correct choice for the following question. State nothing other than the index of the correct choice, without brackets."
    return format_string + "\n" + example + "\n<QUESTION>"

def add_snippets(question):
    format_string = ""
    snippets = question[0]
    factoid_question = question[1]
    if snippets == []:
        return format_string, factoid_question
    output = format_string + "Using the following text snippets, answer the question that follows.\n<SNIPPETS>\n"
    for snippet in snippets:
        output += snippet + "\n"
    output += "</SNIPPETS>\n"
    return output, factoid_question

def promptify(benchmark, question, retrieval_mode = None, retrieved_chunks = None, model = None, zero_shot = False):
    promptified = system_prompt()
    if retrieval_mode != None:
        promptified += retrieval_augmentation(retrieved_chunks)
    if benchmark == "bioASQ_with_snippet" or benchmark == "PubMedQA":
        snippet_addition, question = add_snippets(question)
        promptified += snippet_addition
    if zero_shot:
        promptified += zero_shotify(benchmark)
    else:
        promptified += few_shot(benchmark)
    promptified += question
    promptified += "</QUESTION>\n<ANSWER>"
    print(promptified)
    return promptified

def promptify_for_judging(question, true_answer, model_response):
    if type(true_answer) == list: true_answer = true_answer[0]
    promptified = (
        "<QUESTION>What is the mode of inheritance of Wilson's disease?</QUESTION>\n<MARK_SCHEME>autosomal recessive</MARK_SCHEME>\n<STUDENT_RESPONSE>Autosomal recessive disorder</STUDENT_RESPONSE>\n<JUDGEMENT>The student response is correct</JUDGEMENT>\n<QUESTION>What is the structural fold of bromodomain proteins?</QUESTION>\n<MARK_SCHEME>All-alpha-helical fold</MARK_SCHEME>\n<STUDENT_RESPONSE>Beta-alpha-beta structural fold.</STUDENT_RESPONSE>\n<JUDGEMENT>The student response is incorrect</JUDGEMENT>\n<QUESTION>"
        + question
        + "</QUESTION>\n<MARK_SCHEME>"
        + true_answer
        + "</MARK_SCHEME>\n<STUDENT_RESPONSE>"
        + model_response
        + "</STUDENT_RESPONSE>\n<JUDGEMENT>The student response is"
    )
    # print(promptified)
    return promptified