from src.model import ExLlama, ExLlamaCache, ExLlamaConfig
from src.tokenizer import ExLlamaTokenizer
from src.generator import ExLlamaGenerator
import os, glob
import json
import argparse

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
print(num)
combo = zip(factoid_questions, factoid_answers)
combo = list(combo)
# for item in combo[0:4]:
#     print(item)

prompts = []
for question in factoid_questions[0:3]:
    prompts.append("You are an AI chatbot that answers questions. Given your training on biomedical data, you are an expert on all topics related to biology and medicine. You must now answer the following biomedical question in 5 words or less: " + question + ". Answer:")

print("NOW WE'LL LET THE MODEL WORK ------------------------------------------")

llm = llm(prompts, 100)
for line in llm:
        print(line)