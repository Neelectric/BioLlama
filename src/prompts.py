def promptify_bioASQ_question(question):
    promptified = (
        "You are an excellently helpful AI assistant. For the following, your response MUST start with <ANSWER> and end with </ANSWER>. Given your training on biomedical data, you are an expert on questions related to biology and medicine, such as:\n<QUESTION>Orteronel was developed for treatment of which cancer?</QUESTION>\n<ANSWER>castration-resistant prostate cancer</ANSWER>\n You must now answer the following biomedical question AS SUCCINCTLY AS YOU CAN. Do not use more than 5 words. \n <QUESTION>"
        + question
        + "</QUESTION>\n<ANSWER> "
    )
    return promptified


def promptify_medQA_question(question):
    promptified = (
        "You are an excellently helpful AI assistant. Given your training on biomedical data, you are an expert on questions related to biology and medicine, such as:\n<QUESTION>A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? (A) Ampicillin (B) Ceftriaxone (C) Ciprofloxacin (D) Doxycycline (E) Nitrofurantoin</QUESTION>\n<ANSWER> (E) Nitrofurantoin</ANSWER>\n Select the correct choice for the following question. \n <QUESTION>"
        + question
        + "</QUESTION>\n<ANSWER> "
    )
    return promptified


def promptify_for_judging(question, true_answer, model_response):
    promptified = (
        "<question>What is the mode of inheritance of Wilson's disease?</question>\n<marking_scheme>autosomal recessive</marking_scheme>\n<student_response>Autosomal recessive disorder</student_response>\n <judgement>The student response is correct</judging>\n <question>What is the structural fold of bromodomain proteins?</question>\n<marking_scheme>All-alpha-helical fold</marking_scheme>\n <student_response>Beta-alpha-beta structural fold.</student_response>\n<judging>The student response is incorrect</judging>\n <question>"
        + question
        + "</question>\n<marking_scheme>"
        + true_answer
        + "</marking_scheme>\n <student_response>"
        + model_response
        + "</student_response>\n <judging>The student response is"
    )
    return promptified
