from src.model import ExLlama, ExLlamaCache, ExLlamaConfig
from src.tokenizer import ExLlamaTokenizer
from src.generator import ExLlamaGenerator
import os, glob
import json
import argparse
import re

import src.model_init as model_init
from src.llm import llm as llm

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
factoid_predictions = []
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


prompts = []
for question in factoid_questions[5:15]:
    prompts.append("You are an excellently helpful AI assistant. Given your training on biomedical data, you are an expert on questions related to biology and medicine, such as: <QUESTION>Orteronel was developed for treatment of which cancer?</QUESTION> <ANSWER>castration-resistant prostate cancer</ANSWER> You must now answer the following biomedical question AS SUCCINCTLY AS YOU CAN. Do not use more than 5 words\n <QUESTION>""" 
                   + question 
                   + "</QUESTION> <ANSWER>")

print("NOW WE'LL LET THE MODEL WORK ------------------------------------------")
raw_responses = []
responses = []
llm = llm(prompts, 30)
for line in llm:
    raw_responses.append(line)

pattern = r'<ANSWER>(.*?)</ANSWER>'

for raw_response in raw_responses:
    response = re.findall(pattern, raw_response, re.DOTALL)
    responses.append(response[1])

output = []
for i in range(len(responses)):
    instance = []
    print(factoid_questions[i+5])
    instance.append(factoid_questions[i+5])
    if type(factoid_answers[i+5][0]) != type("String lol"):
        print("Answer: " + str(factoid_answers[i+5][0][0]))
        instance.append(factoid_answers[i+5][0][0])
    else:
        print("Answer: " + str(factoid_answers[i+5][0]))
        instance.append(factoid_answers[i+5][0])
    print("Prediction: " + str(responses[i]))
    instance.append(responses[i])
    print("\n")
    output.append(instance)


with open("output/Llama-2-70B-BioASQ-training5b.json", "w") as outfile: 
    json.dump(output, outfile)