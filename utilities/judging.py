# Part of the BioLlama library
# Written by Neel Rajani
# Implements judging of LLM output on benchmarks given the marking scheme, using Llama2 70B as a judge

import json
import argparse
from tqdm import tqdm
import re
import time
from utilities.llm import llm as llm
from utilities.old_prompts import promptify_for_judging

#time before batch inference

start_time = time.time()

# Directory containing model, tokenizer, generator
model_directory =  "../models/Llama-2-70B-chat-GPTQ"

#which model and benchmark combo to mark
model_to_mark = 'Llama-2-7B-chat-GPTQ'
benchmark_to_mark = "MedMCQA"

with open('output/' + model_to_mark + '-' + benchmark_to_mark + '.json', 'rb') as json_file:
    json_data = json_file.read().decode('utf-8')

data = json.loads(json_data)
prompts = []

for instance in data:
    prompts.append(promptify_for_judging(instance[0], instance[1], instance[2]))

start_index = 0
end_index = 10

prompts = prompts[start_index:end_index]
                   
print(f"--------Start of Llama-2-70B marking {model_to_mark} on {benchmark_to_mark}--------")

def raw_llm_inference(prompts, max_new_tokens):
    llm_output = []
    llm_generator = llm(model_directory, prompts, max_new_tokens)
    for line in llm_generator:
        llm_output.append(line)
    return llm_output

raw_responses = []

for i in tqdm(range(len(prompts)//10), desc="Batch Inference"):
    temp_prompts = list(prompts[i*10:(i+1)*10])
    raw_responses += raw_llm_inference(temp_prompts, 8)
    #print("Performed batch inference on prompts " + str(i*10) + " to " + str((i+1)*10) + ".")

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
total_num_weird = 0
judging_output = []
for raw_response in raw_responses:
    count_correct = len(re.findall(correct_pattern, raw_response, flags=re.IGNORECASE))
    count_incorrect = len(re.findall(incorrect_pattern, raw_response, flags=re.IGNORECASE))
    if count_correct > count_incorrect:
        total_num_correct += 1
    elif count_correct < count_incorrect:
        total_num_incorrect += 1
    else:
        judging_output.append(raw_response)
        total_num_weird += 1

correct_string = "Total number correct: " + str(total_num_correct)
incorrect_string = "Total number incorrect: " + str(total_num_incorrect)
weird_string = "Total number weird: " + str(total_num_weird)
total_string = "Total number of questions asked: " + str(len(raw_responses))
accuracy_string = "Accuracy: " + str(total_num_correct/len(raw_responses))


print(correct_string)
print(incorrect_string)
print(weird_string)
print(total_string)
print(accuracy_string)

judging_output.insert(0, [correct_string, incorrect_string, weird_string, total_string, accuracy_string])
    
with open("output/judging-output-" + model_to_mark + "-" + benchmark_to_mark + ".json", "w") as outfile: 
    json.dump(judging_output, outfile)

print("Written output to output/judging-output-" + model_to_mark + "-" + benchmark_to_mark + ".json")

print("Time for batch inference: " + str(time.time() - start_time))