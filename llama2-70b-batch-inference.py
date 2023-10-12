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
from src.prompts import promptify_benchmark_question

#time before batch inference

start_time = time.time()

# Directory containing model, tokenizer, generator

model_directory =  "../models/Llama-2-70B-chat-GPTQ"

# Locate files we need within that directory

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]

with open('benchmarks/BioASQ-training5b/BioASQ-trainingDataset5b.json', 'rb') as json_file:
    json_data = json_file.read().decode('utf-8')

data = json.loads(json_data)

num = 0
factoid_questions = []
factoid_answers = []
for question in data['questions']:
    if question['type'] == 'factoid':
        num += 1
        # print(question['body'])
        factoid_questions.append(question['body'])
        # print(question['exact_answer'])
        factoid_answers.append(question['exact_answer'])
print("Loaded " + str(num) + " factoid questions.")
combo = zip(factoid_questions, factoid_answers)
combo = list(combo)
print(combo[0:5])

offset = 3
prompts = []
for question in factoid_questions[offset:num]:
    prompts.append(promptify_benchmark_question(question))

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
    #print(temp_prompts)
    raw_responses += batch_llm_inference(temp_prompts, 30)
    print("Performed batch inference on prompts " + str(i*10) + " to " + str((i+1)*10) + ".")

print("We have generated " + str(len(raw_responses)) + " responses.")



pattern = r'<ANSWER>(.*?)</ANSWER>'
responses = []
for raw_response in raw_responses:
    response = re.findall(pattern, raw_response, re.DOTALL)
    # print(raw_response)
    # print(response)
    if len(response) > 2:
        responses.append(response[2])
    else:
        responses.append("LLM SEEMS TO HAVE FAILED TO GENERATE A RESPONSE: " + raw_response)

output = []
for i in range(len(responses)):
    instance = []
    instance.append(factoid_questions[i+offset])
    if type(factoid_answers[i+offset][0]) != type("String lol"):
        instance.append(factoid_answers[i+offset][0][0])
    else:
        instance.append(factoid_answers[i+offset][0])
    instance.append(responses[i])
    output.append(instance)


with open("output/Llama-2-70B-BioASQ-training5b.json", "w") as outfile: 
    json.dump(output, outfile)

print("Time for batch inference: " + str(time.time() - start_time))