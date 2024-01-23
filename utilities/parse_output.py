import json
def parse_output_GPTQ(benchmark, 
                      benchmark_questions, 
                      benchmark_answers,
                      b_start,
                      responses,
                      targetfile):
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
                           responses,
                           targetfile):
    return