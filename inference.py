from src.model import ExLlama, ExLlamaCache, ExLlamaConfig
from src.tokenizer import ExLlamaTokenizer
from src.generator import ExLlamaGenerator
import os, glob
import json
import argparse
import re
import time
import src.model_init as model_init
from src.llm import llm as llm
from src.prompts import promptify_BioASQ_question_no_snippet, promptify_BioASQ_question_with_snippet, promptify_MedQA_question, promptify_PubMedQA_question, promptify_MedMCQA_question
from parse_benchmark import parse_bioASQ_no_snippet, parse_bioASQ_with_snippet, parse_MedQA, parse_PubMedQA, parse_MedMCQA

#time before batch inference
start_time = time.time()

#prepare data and methods depending on model and benchmark
benchmark = "bioASQ5b"
model = "Llama-2-13B-chat-GPTQ"
if benchmark == "bioASQ5b":
    parse_benchmark = parse_bioASQ_with_snippet
    promptify = promptify_BioASQ_question_with_snippet
    targetfile = "output/" + model + "-BioASQ.json"
elif benchmark == "MedQA_US":
    print("IMPLEMENT THE OUTPUTTING AND WRITING TO FILE DOOFUS")
    parse_benchmark = parse_MedQA
    promptify = promptify_MedQA_question
    targetfile = "output/" + model + "-MedQA_USMLE.json"
elif benchmark == "PubMedQA":
    parse_benchmark = parse_PubMedQA
    promptify = promptify_PubMedQA_question
    targetfile = "output/" + model + "-PubMedQA.json"
elif benchmark == "MedMCQA":
    parse_benchmark = parse_MedMCQA
    promptify = promptify_MedMCQA_question
    targetfile = "output/" + model + "-MedQA_USMLE.json"

# Directory containing model, tokenizer, generator
model_directory =  "../models/" + model + "/"

#load benchmark, promptify questions
offset = 280
limit = 340
benchmark_questions, benchmark_answers = parse_benchmark()
prompts = []
for question in benchmark_questions[offset:min(limit, len(benchmark_questions))]:
    prompts.append(promptify(question))

print(f"--------------Start of inference of {model} on questions {offset} to {limit}------------------")

def batch_llm_inference(prompts, max_new_tokens):
    llm_output = []
    llm_generator = llm(model_directory, prompts, max_new_tokens)
    for line in llm_generator:
        llm_output.append(line)
    return llm_output

#perform batch inference
raw_responses = []
max_new_tokens = 35
if len(prompts) > 10:
    for i in range(len(prompts)//10):
        temp_prompts = list(prompts[i*10:(i+1)*10])
        raw_responses += batch_llm_inference(temp_prompts, max_new_tokens)
        print("Performed batch inference on prompts " + str(i*10) + " to " + str((i+1)*10) + ".")
        # with open("output/TEMPORARY_INFERENCE_FILE.json", "w") as outfile: 
        #     json.dump(raw_responses, outfile)
else:
    raw_responses += batch_llm_inference(prompts, max_new_tokens)
    print("Performed batch inference on prompts 0 to " + str(len(prompts)) + ".")
    with open("output/TEMPORARY_INFERENCE_FILE.json", "w") as outfile: 
        json.dump(raw_responses, outfile)
print("We have generated " + str(len(raw_responses)) + " responses.")

#detect answers to benchmark questions in response from the LLM
pattern = r'<ANSWER>(.*?)</[aA][nN][sS][wW][eE][rR]>'
responses = []
for raw_response in raw_responses:
    response = re.findall(pattern, raw_response, re.DOTALL)
    if len(response) > 1:
        responses.append(response[1][2:])
    else:
        responses.append("LLM SEEMS TO HAVE FAILED TO GENERATE A RESPONSE: " + raw_response)

#parse the output and write it to file
if benchmark == "bioASQ5b":
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
elif benchmark == "MedQA_US":
    print("NOT YET IMPLEMENTED")

print("Time for batch inference: " + str(time.time() - start_time))