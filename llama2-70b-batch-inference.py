from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob
import json
import argparse

import model_init

# Directory containing model, tokenizer, generator

model_directory =  "../../models/Llama-2-70B-chat-GPTQ"

# Locate files we need within that directory

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]

with open('datasets/BioASQ-training5b/BioASQ-trainingDataset5b.json', 'rb') as json_file:
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

# Batched prompts

prompts = [
    "Once upon a time",
    "I don't like to",
    "A turbo encabulator is a",
    "In the words of Mark Twain"
]

combo = zip(factoid_questions, factoid_answers)
combo = list(combo)
for item in combo[0:4]:
    print(item)

prompts = []
for question in factoid_questions[0:4]:
    prompts.append("You are an AI chatbot that answers questions. Given your training on biomedical data, you are an expert on all topics related to biology and medicine. You must now answer the following biomedical question in 5 words or less: " + question + ". Answer:")

print("NOW WE'LL LET THE MODEL WORK ------------------------------------------")


#this code is taken from "test_benchmark_inference.py" and adapted

# Parse arguments
parser = argparse.ArgumentParser(description = "Benchmark tests for ExLlama")
model_init.add_args(parser)
parser.add_argument("-lora", "--lora", type = str, help = "Path to LoRA binary to use during benchmark")
parser.add_argument("-loracfg", "--lora_config", type = str, help = "Path to LoRA config to use during benchmark")
parser.add_argument("-ld", "--lora_dir", type = str, help = "Path to LoRA config and binary. to use during benchmark")

args = parser.parse_args()
print("args are as follows: " + str(args))
args.directory = model_directory

model_init.post_parse(args)
model_init.get_model_files(args)

# Globals
model_init.set_globals(args)

# Instantiate model
config = model_init.make_config(args)



config.model_path = model_path                          # supply path to model weights file

model = ExLlama(config)                                 # create ExLlama instance and load the weights
tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

cache = ExLlamaCache(model, batch_size = len(prompts))  # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

# Configure generator

generator.disallow_tokens([tokenizer.eos_token_id])

generator.settings.token_repetition_penalty_max = 1.2
generator.settings.temperature = 0.01
generator.settings.top_p = 0.65
generator.settings.top_k = 100
generator.settings.typical = 0.5

# Generate, batched

for line in prompts:
    print(line)

output = generator.generate_simple(prompts, max_new_tokens = 20)

for line in output:
    print("---")
    print(line)
