def promptify_benchmark_question(question):
    promptified = "You are an excellently helpful AI assistant. For the following, your response MUST start with <ANSWER> and end with </ANSWER>. Given your training on biomedical data, you are an expert on questions related to biology and medicine, such as:\n<QUESTION>Orteronel was developed for treatment of which cancer?</QUESTION>\n<ANSWER>castration-resistant prostate cancer</ANSWER>\n You must now answer the following biomedical question AS SUCCINCTLY AS YOU CAN. Do not use more than 5 words. \n <QUESTION>" + question + "</QUESTION>\n<ANSWER> "
    return promptified

def promptify_for_judging(question, true_answer, model_response):
    promptified = "You are an excellently helpful AI assistant. You have been trained on vast amounts of biomedical data, and are an expert on questions related to biology and medicine. Your task is to mark student responses to biomedical questions. Here are two examples of the question format:\n<question>What is the mode of inheritance of Wilson's disease?</question>\n<marking_scheme>autosomal recessive</marking_scheme>\n<student_response>Autosomal recessive disorder</student_response>\n <mark>correct</mark>\n In this example, the student has answered the question correctly, even if the response is not an exact match to the marking scheme.\n <question>What is the structural fold of bromodomain proteins?</question>\n<marking_scheme>All-alpha-helical fold</marking_scheme>\n <student_response>Beta-alpha-beta structural fold.</student_response>\n<mark>incorrect</mark> In this example, the student has answered the question incorrectly, even though the response is similar to the marking scheme. \n Given this context, you must say whether the student response is correct or incorrect. \n <question>" + question + "</question>\n<marking_scheme>" + true_answer + "</marking_scheme>\n <student_response>" + model_response + "</student_response>.\n <mark>"
    return promptified