def promptify_benchmark_question(question):
    promptified = "You are an excellently helpful AI assistant. For the following, your response MUST start with <ANSWER> and end with </ANSWER>. Given your training on biomedical data, you are an expert on questions related to biology and medicine, such as:\n<QUESTION>Orteronel was developed for treatment of which cancer?</QUESTION>\n<ANSWER>castration-resistant prostate cancer</ANSWER>\n You must now answer the following biomedical question AS SUCCINCTLY AS YOU CAN. Do not use more than 5 words. \n <QUESTION>" + question + "</QUESTION>\n<ANSWER> "
    return promptified

def promptify_for_judging(question, true_answer, model_response):
    promptified = "<question>What is the mode of inheritance of Wilson's disease?</question>\n<marking_scheme>autosomal recessive</marking_scheme>\n<student_response>Autosomal recessive disorder</student_response>\n <judgement>The student response is correct</judging>\n <question>What is the structural fold of bromodomain proteins?</question>\n<marking_scheme>All-alpha-helical fold</marking_scheme>\n <student_response>Beta-alpha-beta structural fold.</student_response>\n<judging>The student response is incorrect</judging>\n <question>" + question + "</question>\n<marking_scheme>" + true_answer + "</marking_scheme>\n <student_response>" + model_response + "</student_response>.\n <judging>The student response is"
    return promptified