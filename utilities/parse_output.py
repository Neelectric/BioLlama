import json
import re
def parse_output_GPTQ(benchmark, 
                      benchmark_questions, 
                      benchmark_answers,
                      b_start,
                      raw_responses,
                      targetfile):
    #detect answers to benchmark questions in response from the LLM
    pattern = r'<ANSWER>(.*?)</[aA][nN][sS][wW][eE][rR]>'
    responses = []

    for raw_response in raw_responses:
        # print("Raw response: " + raw_response  + "\n")
        response = re.findall(pattern, raw_response, re.DOTALL)
        # print("Response: " + str(response) + "\n")
        if len(response) > 2 and benchmark=="MedQA":
            responses.append(response[2][2:])
        elif len(response) > 2:
            responses.append(response[2])
        else:
            responses.append("LLM SEEMS TO HAVE FAILED TO GENERATE A RESPONSE: " + raw_response)
    #parse the output and write it to file
    if benchmark == "bioASQ_no_snippet":
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

    elif benchmark == "MedQA" or benchmark == "MedMCQA":
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
        if len(response) == 1 and benchmark=="MedQA":
            responses.append(response[0][3:])
        elif len(response) > 2:
            responses.append(response[2])
        else:
            responses.append("LLM SEEMS TO HAVE FAILED TO GENERATE A RESPONSE: " + raw_response)
    if benchmark == "MedQA" or benchmark == "MedMCQA":
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
    return