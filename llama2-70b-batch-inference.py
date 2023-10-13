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
from src.prompts import promptify_bioASQ_question, promptify_medQA_question
from parse_benchmark import parse_bioASQ, parse_MedQA

#time before batch inference
start_time = time.time()

# Directory containing model, tokenizer, generator
model_directory =  "../models/Llama-2-70B-chat-GPTQ"

# Locate files we need within that directory
tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]

#this until line 48 was working code for in-file loading of bioasq5bfactoid

# with open('benchmarks/BioASQ-training5b/BioASQ-trainingDataset5b.json', 'rb') as json_file:
#     json_data = json_file.read().decode('utf-8')

# data = json.loads(json_data)

# num = 0
# benchmark_questions = []
# benchmark_answers = []
# for question in data['questions']:
#     if question['type'] == 'factoid':
#         num += 1
#         benchmark_questions.append(question['body'])
#         benchmark_answers.append(question['exact_answer'])
# print("Loaded " + str(num) + " factoid questions.")
# combo = zip(benchmark_questions, benchmark_answers)
# combo = list(combo)
# print(combo[0:5])
offset = 1
limit = 1001
benchmark_questions, benchmark_answers = parse_MedQA("US")

prompts = []
for question in benchmark_questions[offset:limit]:
    prompts.append(promptify_medQA_question(question))

print("NOW WE'LL LET THE MODEL WORK ------------------------------------------")


def batch_llm_inference(prompts, max_new_tokens):
    llm_output = []
    llm_generator = llm(prompts, max_new_tokens)
    for line in llm_generator:
        llm_output.append(line)
    return llm_output

raw_responses = []

for i in range(len(prompts)//10):
    temp_prompts = list(prompts[i*10:(i+1)*10])
    raw_responses += batch_llm_inference(temp_prompts, 35)
    print("Performed batch inference on prompts " + str(i*10) + " to " + str((i+1)*10) + ".")

print("We have generated " + str(len(raw_responses)) + " responses.")

pattern = r'<ANSWER>(.*?)</ANSWER>'
responses = []
for raw_response in raw_responses:
    response = re.findall(pattern, raw_response, re.DOTALL)
    print(response)
    if len(response) == 2:
        responses.append(response[1][2:])
    else:
        responses.append("LLM SEEMS TO HAVE FAILED TO GENERATE A RESPONSE: " + raw_response)

output = []
for i in range(len(responses)):
    instance = []
    instance.append(benchmark_questions[i+offset])
    if type(benchmark_answers[i+offset][0]) != type("String lol"):
        instance.append(benchmark_answers[i+offset])
    else:
        instance.append(benchmark_answers[i+offset])
    instance.append(responses[i])
    output.append(instance)


with open("output/Llama-2-70B-MedQA_USMLE_train.json", "w") as outfile: 
    json.dump(output, outfile)

print("Time for batch inference: " + str(time.time() - start_time))