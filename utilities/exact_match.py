# Part of the BioLlama library
# Written by Neel Rajani
# Implements building a FAISS index on text files, loading this index at inference and performing retrieval

import json
import re
import argparse

def standardize_string_PubMedQA(string):
    return re.sub(r'[. ]+', '', string.strip()).lower()

def mark_MedQA(model, benchmark, input):
    #if model ends in "finetune", save this as a bool
    finetune = False
    if model[-8:] == "finetune" or model[-8:] == "BioLlama":
        finetune = True

    num_correct = 0
    num_total = 0
    if finetune:
        for i in range(len(input)):
            marking_scheme = input[i][1]
            student_response = input[i][2]
            bracket_index = student_response.find("(")
            student_response = student_response[bracket_index:]
            num_total += 1
            if marking_scheme[0:3] == student_response[0:3]:
                num_correct += 1
    else:
        for i in range(len(input)):
            num_total += 1
            if input[i][1] == input[i][2]:
                num_correct += 1

    accuracy = num_correct/num_total

    print(f"Marking model {model} performance on benchmark {benchmark}")
    print("Accuracy is " + str(accuracy) + " with a total of " + str(num_total) + " responses.")
    return accuracy

def mark_PubMedQA(model, benchmark, input):
    valid_answers = ["yes","maybe","no"]
    num_correct = 0
    num_incorrect = 0
    num_total = 0
    num_invalid = 0
    for i in range(len(input)):
        num_total += 1
        marking_scheme = input[i][1]
        student_response = standardize_string_PubMedQA(input[i][2])
        if marking_scheme == student_response:
            num_correct += 1
        elif student_response in valid_answers:
            num_incorrect += 1
        else:
            num_invalid += 1

    accuracy = num_correct/num_total
    print(f"Marking model {model} performance on benchmark {benchmark}")
    print(f"Out of {num_total}, accuracy is {num_correct/num_total} with a total of {num_correct} correct, {num_incorrect} incorrect and {num_invalid} invalid responses.")
    return accuracy

def mark_MedMCQA(model, benchmark, input):
    num_correct = 0
    num_incorrect = 0
    num_total = 0

    for i in range(len(input)):
        input[i][2] = input[i][2].strip()
        num_total += 1
        if input[i][1] == input[i][2]:
            num_correct += 1
        else:
            num_incorrect += 1

    accuracy = num_correct/num_total
    print(f"Marking model {model} performance on benchmark {benchmark}")
    print(f"Out of {num_total}, accuracy is {num_correct/num_total} with a total of {num_correct} correct and {num_incorrect} incorrect responses.")
    return accuracy

def exact_match(model, benchmark):
    output_file = "output/" + model + "-" + benchmark + ".json"
    with open(output_file, 'rb') as json_file:
        json_data = json_file.read().decode('utf-8')
    data = json.loads(json_data)
    if(benchmark == "MedQA"):
        accuracy = mark_MedQA(model, benchmark, data)
    elif(benchmark == "PubMedQA"):
        accuracy = mark_PubMedQA(model, benchmark, data)
    elif(benchmark == "MedMCQA"):
        accuracy = mark_MedMCQA(model, benchmark, data)
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, help="Name of the model that answered questions.")
    parser.add_argument('-b', type=str, help="Name of the benchmark that was used.")
    args = parser.parse_args()

    model = args.m
    benchmark = args.b
    exact_match(model, benchmark)
