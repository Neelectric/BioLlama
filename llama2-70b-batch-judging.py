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

#time before batch inference

start_time = time.time()

# Directory containing model, tokenizer, generator

model_directory =  "../models/Llama-2-70B-chat-GPTQ"

# Locate files we need within that directory

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]

with open('output/Llama-2-70B-BioASQ-training5b.json', 'rb') as json_file:
    json_data = json_file.read().decode('utf-8')

data = json.loads(json_data)
print("data is of type " + str(type(data)))
prompts = []

for instance in data:
    prompts.append("For the following scenario, answer only with the word \"correct\" or \"incorrect\". You are an excellently helpful AI assistant. You have been trained on vast amounts of biomedical data, and are an expert on questions related to biology and medicine. Your task is to mark student responses to biomedical questions. Here are two examples of the question format: \n<QUESTION>What is the mode of inheritance of Wilson's disease?</QUESTION> <MARKING_SCHEME>autosomal recessive</MARKING_SCHEME> <STUDENT_RESPONSE>Autosomal recessive disorder</STUDENT_RESPONSE>\n In this example, the student has answered the question correctly, even if the response is not an exact match to the marking scheme.\n<QUESTION>What is the structural fold of bromodomain proteins?</QUESTION> <MARKING_SCHEME>All-alpha-helical fold</MARKING_SCHEME> <STUDENT_RESPONSE>Beta-alpha-beta structural fold.</STUDENT RESPONSE>\n In this example, the student has answered the question incorrectly, even though the response is similar to the marking scheme.\n Given this context, you must say whether the student response is correct or incorrect. Use ONLY the words CORRECT or INCORRECT. <QUESTION>" + instance[0] + "</QUESTION> <MARKING_SCHEME>" + instance[1] + "</MARKING_SCHEME> <STUDENT_RESPONSE>" + instance[2] + "</STUDENT_RESPONSE>. Is the student response correct or incorrect?")
                   
print("NOW WE'LL LET THE MODEL WORK ------------------------------------------")



def raw_llm_inference(prompts, max_new_tokens):
    llm_output = []

    llm_generator = llm(prompts, max_new_tokens)
    for line in llm_generator:
        llm_output.append(line)
    return llm_output

raw_responses = []

for i in range(len(prompts)//10):
    temp_prompts = list(prompts[i*10:(i+1)*10])
    raw_responses += raw_llm_inference(temp_prompts, 8)
    print("Performed batch inference on prompts " + str(i*10) + " to " + str((i+1)*10) + ".")

print("We have generated " + str(len(raw_responses)) + " responses.")


# Initialize counters
correct_count = 0
incorrect_count = 0

# Define regular expressions
correct_pattern = r'\bcorrect\b'
incorrect_pattern = r'\bincorrect\b'

responses = []
total_num_correct = 0
total_num_incorrect = 0
for raw_response in raw_responses:
    count_correct = len(re.findall(correct_pattern, raw_response, flags=re.IGNORECASE))
    count_incorrect = len(re.findall(incorrect_pattern, raw_response, flags=re.IGNORECASE))
    if count_correct > count_incorrect:
        total_num_correct += 1
    elif count_correct < count_incorrect:
        total_num_incorrect += 1
    else:
        print("Weird response: " + raw_response)

print("Total number correct: " + str(total_num_correct))
print("Total number of questions asked: " + str(len(raw_responses)))
print("Accuracy: " + str(total_num_correct/len(raw_responses)))

    


# with open("output/JUDGING-Llama-2-70B-BioASQ-training5b.json", "w") as outfile: 
#     json.dump(output, outfile)

print("Time for batch inference: " + str(time.time() - start_time))