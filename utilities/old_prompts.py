# # Part of the BioLlama library
# # Written by Neel Rajani
# # Retired file, old prompting approach

# def promptify_BioASQ_question_no_snippet(question):
#     promptified = (
#         "You are an excellently helpful AI assistant. For the following, your response MUST start with <ANSWER> and end with </ANSWER>. Given your training on biomedical data, you are an expert on questions related to biology and medicine, such as:<QUESTION>Orteronel was developed for treatment of which cancer?</QUESTION><ANSWER>castration-resistant prostate cancer</ANSWER> You must now answer the following biomedical question AS SUCCINCTLY AS YOU CAN. Do not use more than 5 words. <QUESTION>"
#         + question
#         + "</QUESTION>\n<ANSWER> "
#     )
#     return promptified


# def promptify_BioASQ_question_with_snippet(question):
#     snippets = question[0]
#     question_body = question[1]

#     promptified = "You are an excellently helpful AI assistant. For the following, your response MUST start with <ANSWER> and end with </ANSWER>. Do not use more than 5 words. Given your training on biomedical data, you are an expert on questions related to biology and medicine."
#     if len(snippets) > 0:
#         promptified += "You are given the following snippets, from which you must extract the answer to a question. Snippets:"
#         for snippet in snippets:
#             promptified += snippet + ""
#     promptified += (
#         "You must now answer the following biomedical question AS SUCCINCTLY AS YOU CAN. Do not use more than 5 words.<QUESTION>"
#         + question_body
#         + "</QUESTION><ANSWER> "
#     )
#     return promptified


# def promptify_MedQA_question(question):
#     promptified = (
#         "You are an excellently helpful AI assistant. Given your training on biomedical data, you are an expert on questions related to biology and medicine, such as:<QUESTION>A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? (A) Ampicillin (B) Ceftriaxone (C) Ciprofloxacin (D) Doxycycline (E) Nitrofurantoin</QUESTION><ANSWER> (E) Nitrofurantoin</ANSWER> Select the correct choice for the following question. <QUESTION>"
#         + question
#         + "</QUESTION><ANSWER> "
#     )
#     return promptified


# def promptify_PubMedQA_question(question):
#     snippets = question[0]
#     question_body = question[1]

#     promptified = "You are an excellently helpful AI assistant. For the following, your response MUST start with <ANSWER> and end with </ANSWER>. You must answer with NOTHING OTHER THAN Yes, No or Maybe. Given your training on biomedical data, you are an expert on questions related to biology and medicine."
#     if len(snippets) > 0:
#         promptified += " You are given the following snippets, from which you must extract the answer to a question. Snippets:"
#         for snippet in snippets:
#             promptified += snippet
#     promptified += " You must now respond to the following question with NOTHING OTHER THAN Yes, No or Maybe given the context above. Do not justify your response, respond with only Yes, No or Maybe. <QUESTION>" + question_body + "</QUESTION><ANSWER> "
#     return promptified


# def promptify_MedMCQA_question(question):
#     #this version caused the llm to create output that was incompatible with exact match
#     #promptified_old = "You are an excellently helpful AI assistant. Given your training on biomedical data, you are an expert on questions related to biology and medicine, such as:<QUESTION> Which of the following is not true for myelinated nerve fibers: (1) Impulse through myelinated fibers is slower than non-myelinated fibers (2) Membrane currents are generated at nodes of Ranvier (3) Saltatory conduction of impulses is seen (4) Local anesthesia is effective only when the nerve is not covered by myelin sheath</QUESTION><ANSWER> (1) Impulse through myelinated fibers is slower than non-myelinated fibers</ANSWER> Select the correct choice for the following question. State nothing other than the correct choice, in the exact same format as it is phrased in the question. Your response may not include anything else, just respond with the correct choice in the format you are given. <QUESTION>" + question + "</QUESTION><ANSWER> "

#     promptified = "You are an excellently helpful AI assistant. Given your training on biomedical data, you are an expert on questions related to biology and medicine, such as:<QUESTION> Which of the following is not true for myelinated nerve fibers: (1) Impulse through myelinated fibers is slower than non-myelinated fibers (2) Membrane currents are generated at nodes of Ranvier (3) Saltatory conduction of impulses is seen (4) Local anesthesia is effective only when the nerve is not covered by myelin sheath</QUESTION><ANSWER>3</ANSWER> Select the correct choice for the following question. State nothing other than the index of the correct choice, without brackets. <QUESTION>" + question + "</QUESTION><ANSWER>"
#     return promptified


# def promptify_for_judging(question, true_answer, model_response):
#     promptified = (
#         "<question>What is the mode of inheritance of Wilson's disease?</question><marking_scheme>autosomal recessive</marking_scheme><student_response>Autosomal recessive disorder</student_response><judgement>The student response is correct</judging><question>What is the structural fold of bromodomain proteins?</question><marking_scheme>All-alpha-helical fold</marking_scheme><student_response>Beta-alpha-beta structural fold.</student_response><judging>The student response is incorrect</judging><question>"
#         + question
#         + "</question><marking_scheme>"
#         + true_answer
#         + "</marking_scheme><student_response>"
#         + model_response
#         + "</student_response><judging>The student response is"
#     )
#     return promptified
