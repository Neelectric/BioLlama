import json
import re
import time
import argparse
from tqdm import tqdm
from utilities.llm import llm as llm
from src.count_tokens import count_tokens
from utilities.parse_benchmark import parse_benchmark, parse_bioASQ_no_snippet, parse_BioASQ_with_snippet, parse_MedQA, parse_PubMedQA, parse_MedMCQA
from utilities.prompts2 import promptify, promptify_for_judging

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
    
    #preparatory steps
    start_time = time.time() # time before batch inference
    targetfile = "output/" + model + "-" + benchmark + ".json" #file to write output to
    model_directory =  "../models/" + model + "/" #directory containing model, tokenizer, generator
    benchmark_questions, benchmark_answers = parse_benchmark(benchmark) #load benchmark
    prompts = []
    raw_responses = []
    for question in benchmark_questions[b_start:min(b_end, len(benchmark_questions))]:
        prompts.append(promptify(benchmark, question)) #promptify questions

    print(f"--------------Start of inference of {model} on questions {b_start} to {b_end}------------------")

    #helper function for batch inference
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
            with open("output/TEMPORARY_INFERENCE_FILE.json", "w") as outfile: 
                json.dump(raw_responses, outfile)
        else:
            raw_responses += batch_llm_inference(prompts, max_new_tokens)
            with open("output/TEMPORARY_INFERENCE_FILE.json", "w") as outfile: 
                json.dump(raw_responses, outfile)
        print("We have generated " + str(len(raw_responses)) + " responses.")
    elif inference_mode == "alt":
        for prompt in tqdm(prompts, desc="Alt Inference"):
            raw_responses.append(llm(model_directory, prompt, max_new_tokens, generator_mode="alt"))
        with open("output/TEMPORARY_INFERENCE_FILE.json", "w") as outfile: 
            json.dump(raw_responses, outfile)
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
            instance.append(benchmark_questions[i+b_start][1])
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

    print("Time for batch inference: " + str(time.time() - start_time))

#main method
if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Llama-2-70B-chat-GPTQ", help="specify model to use")
    parser.add_argument("--benchmark", default="MedMCQA", help="specify benchmark to use")
    parser.add_argument("--b_start", default=0, help="specify index of first question in benchmark to start with")
    parser.add_argument("--b_end", default=1, help="specify index of last question in benchmark to end with")
    parser.add_argument("--max_new_tokens", default=30, help="specify maximum number of tokens to generate")
    parser.add_argument("--inference_mode", default="std", help="specify inference mode")
    parser.add_argument("--retrieval", default=False, help="specify whether to perform retrieval")
    args = parser.parse_args()

    inference(model=args.model, 
              benchmark=args.benchmark, 
              b_start = int(args.b_start), 
              b_end = int(args.b_end), 
              max_new_tokens = int(args.max_new_tokens),
              inference_mode = args.inference_mode,
              retrieval = bool(args.retrieval))