# Part of the BioLlama library
# Written by Neel Rajani
# Implements judging of LLM output on benchmarks given the marking scheme, using Llama2 70B as a judge

import json
import argparse
from tqdm import tqdm
import re
import time
from utilities.llm import llm as llm
from utilities.prompts2 import promptify_for_judging

#time before batch inference
def llm_as_judge(model_to_mark='Llama-2-70B-chat-GPTQ', benchmark_to_mark="bioASQ_no_snippet"):
    start_time = time.time()

    # Directory containing the judger model, tokenizer, generator
    model_directory =  "../models/Llama-2-70B-chat-GPTQ"

    print("opening output/" + model_to_mark + "-" + benchmark_to_mark + ".json")
    with open('output/' + model_to_mark + '-' + benchmark_to_mark + '.json', 'rb') as json_file:
        json_data = json_file.read().decode('utf-8')

    data = json.loads(json_data)
    prompts = []
    if benchmark_to_mark == "bioASQ_no_snippet":
        for instance in data:
            prompts.append(promptify_for_judging(instance[0], instance[1], instance[2]))
    elif benchmark_to_mark == "bioASQ_with_snippet":
        for instance in data:
            prompts.append(promptify_for_judging(instance[0][1], instance[1], instance[2]))
    else:
        #throw an error complaining we are neither using bioASQ_no_snippet nor bioASQ_with_snippet
        print("Error: Benchmark not supported for judging")
        return
    start_index = 0
    end_index = 1000

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

    print("We have generated " + str(len(raw_responses)) + " responses.")


    # correct_count = 0
    # incorrect_count = 0
    correct_pattern = r'\bcorrect\b'
    incorrect_pattern = r'\bincorrect\b'
    # responses = []
    total_num_correct = 0
    total_num_incorrect = 0
    total_num_weird = 0
    judging_output = []
    for raw_response in raw_responses:
        print("\n")
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
    return total_num_correct / len(raw_responses)