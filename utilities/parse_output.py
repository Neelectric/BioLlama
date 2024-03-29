import json
import re
def parse_output_GPTQ(benchmark, 
                      benchmark_questions, 
                      benchmark_answers,
                      b_start,
                      raw_responses,
                      targetfile,
                      zero_shot=False):
    #detect answers to benchmark questions in response from the LLM
    pattern = r'<ANSWER>(.*?)</[aA][nN][sS][wW][eE][rR]>'
    responses = []
    weird_counter = 0
    for raw_response in raw_responses:
        # print("Raw response: " + raw_response  + "\n")
        response = re.findall(pattern, raw_response, re.DOTALL)
        # print("Response: " + str(response) + "\n")
        if len(response) > 2 and (benchmark == "MedQA-4" or benchmark == "MedQA-5"):
            responses.append(response[2][1:])
        elif len(response) > 2 and benchmark=="PubMedQA":
            responses.append(response[2])
        elif len(response) == 2:
            responses.append(response[1])
        else:
            last_answer_index = raw_response.rfind('<ANSWER>')
            response = raw_response[last_answer_index + len('<ANSWER>'):].strip()
            responses.append(response)
            weird_counter += 1
            # response = re.findall(pattern_emergency, raw_response, re.DOTALL)
            # if len(response) == 1:
            #     responses.append(response[0])
            # else:
            # responses.append("LLM SEEMS TO HAVE FAILED TO GENERATE A RESPONSE: " + raw_response)
    #parse the output and write it to file
    print(f"In total, {weird_counter} responses did not use <ANSWER> tags properly.")
    if benchmark == "bioASQ_no_snippet" or benchmark == "bioASQ_with_snippet":
        output = []
        for i in range(len(responses)):
            instance = []
            instance.append(benchmark_questions[i+b_start])
            if type(benchmark_answers[i+b_start][0]) != type("String lol"):
                instance.append(benchmark_answers[i+b_start][0][0])
            else:
                instance.append(benchmark_answers[i+b_start][0])
            
            instance.append(responses[i])
            output.append(instance)
        with open(targetfile, "w") as outfile: 
            json.dump(output, outfile)
        print("Written output to " + targetfile)

    elif benchmark == "MedQA-4" or benchmark == "MedQA-5" or benchmark == "MedMCQA":
        output = []
        for i in range(len(responses)):
            instance = []
            instance.append(benchmark_questions[i+b_start])
            instance.append(benchmark_answers[i+b_start])
            instance.append(responses[i])
            output.append(instance)
        with open(targetfile, "w") as outfile: 
            json.dump(output, outfile)
        print("Written output to " + targetfile)

    elif benchmark == "PubMedQA":
        output = []
        for i in range(len(responses)):
            instance = []
            instance.append(benchmark_questions[i+b_start][1])
            instance.append(benchmark_answers[i+b_start])
            instance.append(responses[i])
            output.append(instance)
        with open(targetfile, "w") as outfile: 
            json.dump(output, outfile)
        print("Written output to " + targetfile)


def parse_output_finetuned(benchmark, 
                           benchmark_questions, 
                           benchmark_answers,
                           b_start,
                           raw_responses,
                           targetfile):
    #detect answers to benchmark questions in response from the LLM
    pattern = r'<ANSWER>(.*?)</[aA][nN][sS][wW][eE][rR]>'
    responses = []

    for raw_response in raw_responses:
        raw_response += "</ANSWER>"
        # print("Raw response: " + raw_response  + "\n")
        response = re.findall(pattern, raw_response, re.DOTALL)
        # print("Response: " + str(response) + "\n")
        if len(response) == 1 and (benchmark == "MedQA-4" or benchmark == "MedQA-5"):
            responses.append(response[0][3:])
        elif len(response) > 2:
            responses.append(response[2])
        else:
            responses.append("LLM SEEMS TO HAVE FAILED TO GENERATE A RESPONSE: " + raw_response)
    if benchmark == "MedQA-4" or benchmark == "MedQA-5" or benchmark == "MedMCQA":
        output = []
        for i in range(len(responses)):
            instance = []
            instance.append(benchmark_questions[i+b_start])
            instance.append(benchmark_answers[i+b_start])
            instance.append(responses[i])
            output.append(instance)
        with open(targetfile, "w") as outfile: 
            json.dump(output, outfile)
        print("Written output to " + targetfile)
    elif benchmark == "bioASQ_no_snippet" or benchmark == "bioASQ_with_snippet":
        output = []
        for i in range(len(responses)):
            instance = []
            instance.append(benchmark_questions[i+b_start])
            if type(benchmark_answers[i+b_start][0]) != type("String lol"):
                instance.append(benchmark_answers[i+b_start][0][0])
            else:
                instance.append(benchmark_answers[i+b_start][0])
            instance.append(responses[i])
            output.append(instance)
        with open(targetfile, "w") as outfile: 
            json.dump(output, outfile)
        print("Written output to " + targetfile)
    elif benchmark == "PubMedQA":
        output = []
        for i in range(len(responses)):
            instance = []
            instance.append(benchmark_questions[i+b_start][1])
            instance.append(benchmark_answers[i+b_start])
            instance.append(responses[i])
            output.append(instance)
        with open(targetfile, "w") as outfile: 
            json.dump(output, outfile)
        print("Written output to " + targetfile)
        with open(targetfile, "w") as outfile: 
            json.dump(output, outfile)
        print("Written output to " + targetfile)
    return