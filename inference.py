import json
import re
import time
import argparse
from tqdm import tqdm
from src.llm import llm as llm
from src.count_tokens import count_tokens
from parse_benchmark import parse_bioASQ_no_snippet, parse_BioASQ_with_snippet, parse_MedQA, parse_PubMedQA, parse_MedMCQA
from src.prompts2 import promptify, promptify_for_judging

#method summary:
# 1. Load specified benchmark
# 2. Prepare specified model
# 3. Promptify questions
# 4. Perform batch inference on benchmark questions
# 5. Parse output
# 6. Write output to file
def inference(model="Llama-2-70B-chat-GPTQ", 
              benchmark="MedMCQA", 
              b_start = 0, 
              b_end = 1, 
              max_new_tokens = 30,
              inference_mode = "std",
              retrieval = False):
    #time before batch inference
    start_time = time.time()

    #index of first question in benchmark to start/end with
    offset = b_start
    limit = b_end

    #max number of tokens we allow the model to generate
    max_new_tokens = 30 

    if benchmark == "BioASQ5b":
        parse_benchmark = parse_BioASQ_with_snippet
    elif benchmark == "MedQA":
        parse_benchmark = parse_MedQA
    elif benchmark == "PubMedQA":
        parse_benchmark = parse_PubMedQA
    elif benchmark == "MedMCQA":
        parse_benchmark = parse_MedMCQA

    targetfile = "output/" + model + "-" + benchmark + ".json"

    #directory containing model, tokenizer, generator
    model_directory =  "../models/" + model + "/"

    #load benchmark, promptify questions
    benchmark_questions, benchmark_answers = parse_benchmark()
    prompts = []
    raw_responses = []
    for question in benchmark_questions[offset:min(limit, len(benchmark_questions))]:
        prompts.append(promptify(benchmark, question))

    print(f"--------------Start of inference of {model} on questions {offset} to {limit}------------------")

    def batch_llm_inference(prompts, max_new_tokens):
        llm_output = []
        llm_generator = llm(model_directory, prompts, max_new_tokens)
        for line in llm_generator:
            llm_output.append(line)
        return llm_output

    #perform batch inference
    if inference_mode == "std":
        if len(prompts) > 10:
            for i in tqdm(range(len(prompts)//10), desc="Batch Inference"):
                temp_prompts = list(prompts[i*10:(i+1)*10])
                raw_responses += batch_llm_inference(temp_prompts, max_new_tokens)
        else:
            print("prompts is of type " + str(type(prompts)) + " and length " + str(len(prompts)))
            print("prompts[0] is of type " + str(type(prompts[0])) + " and length " + str(len(prompts[0])))
            print("prompt[0] is " + prompts[0])
            raw_responses += batch_llm_inference(prompts, max_new_tokens)
            print("Performed batch inference on prompts 0 to " + str(len(prompts)) + ".")
            with open("output/TEMPORARY_INFERENCE_FILE.json", "w") as outfile: 
                json.dump(raw_responses, outfile)
        print("We have generated " + str(len(raw_responses)) + " responses.")
    elif inference_mode == "alt":
        for prompt in tqdm(prompts, desc="Alt Inference"):
            raw_responses.append(llm(model_directory, prompt, max_new_tokens, generator_mode="alt"))
        print("We have generated " + str(len(raw_responses)) + " responses.")

    #detect answers to benchmark questions in response from the LLM
    pattern = r'<ANSWER>(.*?)</[aA][nN][sS][wW][eE][rR]>'
    responses = []

    for raw_response in raw_responses:
        response = re.findall(pattern, raw_response, re.DOTALL)
        if len(response) > 1 and benchmark != "MedMCQA":
            responses.append(response[1][2:])
        elif len(response) > 1 and benchmark == "MedMCQA":
            responses.append(response[1])
        else:
            responses.append("LLM SEEMS TO HAVE FAILED TO GENERATE A RESPONSE: " + raw_response)

    #parse the output and write it to file
    if benchmark == "BioASQ5b":
        output = []
        for i in range(len(responses)):
            instance = []
            instance.append(benchmark_questions[i+offset][1])
            if type(benchmark_answers[i+offset][0]) != type("String lol"):
                instance.append(benchmark_answers[i+offset][0][0])
            else:
                instance.append(benchmark_answers[i+offset][0])
            
            instance.append(responses[i])
            output.append(instance)
        with open(targetfile, "w") as outfile: 
            json.dump(output, outfile)
        print("Written output to " + targetfile)

    elif benchmark == "MedQA" or benchmark == "MedMCQA":
        output = []
        for i in range(len(responses)):
            instance = []
            instance.append(benchmark_questions[i+offset])
            instance.append(benchmark_answers[i+offset])
            instance.append(responses[i])
            output.append(instance)
        with open(targetfile, "w") as outfile: 
            json.dump(output, outfile)
        print("Written output to " + targetfile)

    elif benchmark == "PubMedQA":
        output = []
        for i in range(len(responses)):
            instance = []
            instance.append(benchmark_questions[i+offset][1])
            instance.append(benchmark_answers[i+offset])
            instance.append(responses[i])
            output.append(instance)
        with open(targetfile, "w") as outfile: 
            json.dump(output, outfile)
        print("Written output to " + targetfile)

    print("Time for batch inference: " + str(time.time() - start_time))


# (model="Llama-2-70B-chat-GPTQ", 
#               benchmark="MedMCQA", 
#               b_start = 0, 
#               b_end = 1, 
#               max_new_tokens = 30,
#               inference_mode = "std",
#               retrieval = False):


#main method
if __name__ == "__main__":
    #parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", default="Llama-2-70B-chat-GPTQ", help="specify model to use")
    # parser.add_argument("--benchmark", default="MedMCQA", help="specify benchmark to use")
    # parser.add_argument("--b_start", default=0, help="specify index of first question in benchmark to start with")
    # parser.add_argument("--b_end", default=1, help="specify index of last question in benchmark to end with")
    # parser.add_argument("--max_new_tokens", default=30, help="specify maximum number of tokens to generate")
    # parser.add_argument("--inference_mode", default="std", help="specify inference mode")
    # parser.add_argument("--retrieval", default=False, help="specify whether to perform retrieval")
    # args = parser.parse_args()

    # inference(model=args.model, 
    #           benchmark=args.benchmark, 
    #           b_start = int(args.b_start), 
    #           b_end = int(args.b_end), 
    #           max_new_tokens = int(args.max_new_tokens),
    #           inference_mode = args.inference_mode,
    #           retrieval = bool(args.retrieval))
    inference(model="Llama-2-70B-chat-GPTQ",
                benchmark="MedMCQA",
                b_start = 0,
                b_end = 20,
                max_new_tokens = 30,
                inference_mode = "std",
                retrieval = False)