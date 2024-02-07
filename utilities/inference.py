# Part of the BioLlama library
# Written by Neel Rajani
# Primary file that allows for batch or alt inference of a GPTQ Llama2 model on a given benchmark
# Also callable from CLI if given arguments as specified at the bottom of this file

import json
import re
import numpy as np
import time
import argparse
from tqdm import tqdm
from utilities.llm import llm as llm
from src.count_tokens import count_tokens
from utilities.parse_benchmark import parse_benchmark, parse_bioASQ_no_snippet, parse_BioASQ_with_snippet, parse_MedQA, parse_PubMedQA, parse_MedMCQA
from utilities.prompts2 import promptify, promptify_for_judging
from utilities.db_retrieval import gte_FAISS_retrieval, medcpt_FAISS_retrieval
from utilities.parse_output import parse_output_GPTQ, parse_output_finetuned
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
              retrieval_model = None,
              retrieval_text_mode = "full",
              chunk_length = 16,
              top_k = 1,
              db_name = "RCT20ktrain"):
    
    #preparatory steps
    start_time = time.time() # time before batch inference
    targetfile = "output/" + model + "-" + benchmark + ".json" #file to write output to
    model_directory =  "../models/" + model + "/" #directory containing model, tokenizer, generator
    benchmark_questions, benchmark_answers = parse_benchmark(benchmark) #load benchmark
    prompts = []
    raw_responses = []

    #retrieving chunks for questions all at once
    if retrieval_model == "gte-large":
        retrieved_chunks = gte_FAISS_retrieval(benchmark_questions[b_start:min(b_end, len(benchmark_questions))], db_name, retrieval_text_mode, top_k=top_k)
    elif retrieval_model == "medcpt":
        retrieved_chunks = medcpt_FAISS_retrieval(benchmark_questions[b_start:min(b_end, len(benchmark_questions))], db_name, retrieval_text_mode, chunk_length=chunk_length, top_k=top_k)  
    #this seems dubious, just doing this so promptify does not complain...
    else:
        retrieved_chunks = np.zeros((min(b_end, len(benchmark_questions)) - b_start, 1))
    
    #promptifying questions
    chunk_index = 0
    for question in benchmark_questions[b_start:min(b_end, len(benchmark_questions))]:
        prompts.append(promptify(benchmark=benchmark, question=question, retrieval_mode=retrieval_model, retrieved_chunks=retrieved_chunks[chunk_index], model=model)) #promptify questions
        chunk_index += 1
    
    #if model string ends in "finetune", print this
    if model[-8:] == "finetune":
        model_directory = "/home/service/BioLlama/utilities/finetuning/llama2_training_output/"
    
    
    print(f"--------------Start of inference of {model} on questions {b_start} to {b_end}------------------")

    #helper function for batch inference
    def batch_llm_inference(prompts, max_new_tokens, model_object):
        llm_output = []
        llm_generator, model_object = llm(model_directory, prompts, max_new_tokens, "std", model_object)
        for line in llm_generator:
            llm_output.append(line)
        return llm_output, model_object

    #perform batch inference
    #TODO: IF NUM OF PROMPTS IS NOT A MULTIPLE OF 10, AM I MISSING THE LAST FEW PROMPTS?
    if inference_mode == "std":
        if len(prompts) > 10:
            model_object = None
            for i in tqdm(range(len(prompts)//10), desc="Batch Inference"):
                temp_prompts = list(prompts[i*10:(i+1)*10])
                #print the largest prompt length by string length
                #print("Largest prompt length: " + str(max([len(prompt) for prompt in temp_prompts])))
                #print list of string length per prompt
                # for temp_prompt in temp_prompts:
                #     print(temp_prompt)                
                # print([len(prompt) for prompt in temp_prompts])
                
                temp_responses, model_object = batch_llm_inference(temp_prompts, max_new_tokens, model_object)  
                raw_responses += temp_responses
            with open("output/TEMPORARY_INFERENCE_FILE.json", "w") as outfile: 
                json.dump(raw_responses, outfile)
        else:
            raw_responses += batch_llm_inference(prompts, max_new_tokens)
            #if there is only one prompt, ensure that raw_responses is a list containing this prompt string:
            #print("Raw responses: " + str(raw_responses))
            if type(raw_responses) != type([]):
                raw_responses = [raw_responses]
            with open("output/TEMPORARY_INFERENCE_FILE.json", "w") as outfile: 
                json.dump(raw_responses, outfile)
        print("We have generated " + str(len(raw_responses)) + " responses.")
    elif inference_mode == "alt":
        for prompt in tqdm(prompts, desc="Alt Inference"):
            raw_responses.append(llm(model_directory, prompt, max_new_tokens, generator_mode="alt"))
        with open("output/TEMPORARY_INFERENCE_FILE.json", "w") as outfile: 
            json.dump(raw_responses, outfile)
        print("We have generated " + str(len(raw_responses)) + " responses.")

    # #detect answers to benchmark questions in response from the LLM
    # pattern = r'<ANSWER>(.*?)</[aA][nN][sS][wW][eE][rR]>'
    # responses = []

    # for raw_response in raw_responses:
    #     # print("Raw response: " + raw_response  + "\n")
    #     response = re.findall(pattern, raw_response, re.DOTALL)
    #     # print("Response: " + str(response) + "\n")
    #     if len(response) > 2 and benchmark=="MedQA":
    #         responses.append(response[2][2:])
    #     elif len(response) > 2:
    #         responses.append(response[2])
    #     else:
    #         responses.append("LLM SEEMS TO HAVE FAILED TO GENERATE A RESPONSE: " + raw_response)

    #parse the output and write it to file
    if model[-8:] == "finetune":
        parse_output_finetuned(benchmark,
                                benchmark_questions,
                                benchmark_answers,
                                b_start,
                                raw_responses,
                                targetfile)
    else:
        parse_output_GPTQ(benchmark, 
                          benchmark_questions, 
                          benchmark_answers,
                          b_start,
                          raw_responses,
                          targetfile)
    # if benchmark == "bioASQ_no_snippet":
    #     output = []
    #     for i in range(len(responses)):
    #         instance = []
    #         instance.append(benchmark_questions[i+b_start])
    #         if type(benchmark_answers[i+b_start][0]) != type("String lol"):
    #             instance.append(benchmark_answers[i+b_start][0][0])
    #         else:
    #             instance.append(benchmark_answers[i+b_start][0])
            
    #         instance.append(responses[i])
    #         output.append(instance)
    #     with open(targetfile, "w") as outfile: 
    #         json.dump(output, outfile)
    #     print("Written output to " + targetfile)

    # elif benchmark == "MedQA" or benchmark == "MedMCQA":
    #     output = []
    #     for i in range(len(responses)):
    #         instance = []
    #         instance.append(benchmark_questions[i+b_start])
    #         instance.append(benchmark_answers[i+b_start])
    #         instance.append(responses[i])
    #         output.append(instance)
    #     with open(targetfile, "w") as outfile: 
    #         json.dump(output, outfile)
    #     print("Written output to " + targetfile)

    # elif benchmark == "PubMedQA":
        # output = []
        # for i in range(len(responses)):
        #     instance = []
        #     instance.append(benchmark_questions[i+b_start][1])
        #     instance.append(benchmark_answers[i+b_start])
        #     instance.append(responses[i])
        #     output.append(instance)
        # with open(targetfile, "w") as outfile: 
        #     json.dump(output, outfile)
        # print("Written output to " + targetfile)

    print("Time for batch inference: " + str(time.time() - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Llama-2-70B-chat-GPTQ", help="specify model to use")
    parser.add_argument("--benchmark", default="MedMCQA", help="specify benchmark to use")
    parser.add_argument("--b_start", default=0, help="specify index of first question in benchmark to start with")
    parser.add_argument("--b_end", default=1, help="specify index of last question in benchmark to end with")
    parser.add_argument("--max_new_tokens", default=30, help="specify maximum number of tokens to generate")
    parser.add_argument("--inference_mode", default="std", help="specify inference mode")
    parser.add_argument("--retrieval_mode", default=False, help="specify whether and how to perform retrieval")
    args = parser.parse_args()

    inference(model=args.model, 
              benchmark=args.benchmark, 
              b_start = int(args.b_start), 
              b_end = int(args.b_end), 
              max_new_tokens = int(args.max_new_tokens),
              inference_mode = args.inference_mode,
              retrieval_model = args.retrieval_mode
              )