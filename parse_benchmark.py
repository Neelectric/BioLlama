import json
import argparse
from collections import Counter

def parse_bioASQ_no_snippet(version="5b"):
    #read in raw benchmark
    with open('benchmarks/BioASQ-training5b/BioASQ-trainingDataset5b.json', 'rb') as json_file:
        json_data = json_file.read().decode('utf-8')
    data = json.loads(json_data)
    num_factoids = 0
    num_non_factoid = 0
    benchmark_questions = []
    benchmark_answers = []
    question_types = {}
    
    #for every question in raw data, add to output if type is factoid
    for question in data["questions"]:
        if question["type"] not in question_types:
            question_types[question["type"]] = 1
        else:
            question_types[question["type"]] += 1
        if question["type"] == "factoid":
            benchmark_questions.append(question['body'])
            benchmark_answers.append(question['exact_answer'])
            num_factoids += 1
        else:
            num_non_factoid += 1
    print(question_types)
    print(f"Benchmark contains {num_factoids + num_non_factoid} questions, made up of {question_types}")
    return benchmark_questions, benchmark_answers

def parse_bioASQ_with_snippet(version="5b"):
    #read in raw benchmark
    with open('benchmarks/BioASQ-training5b/BioASQ-trainingDataset5b.json', 'rb') as json_file:
        json_data = json_file.read().decode('utf-8')
    data = json.loads(json_data)
    num_factoids = 0
    num_non_factoid = 0
    benchmark_questions = []
    benchmark_answers = []
    question_types = {}

    #for every question in raw data, add to output if type is factoid. benchmark_questions consists of 5 snippets and the question
    for question in data["questions"]:
        if question["type"] not in question_types:
            question_types[question["type"]] = 1
        else:
            question_types[question["type"]] += 1
        if question["type"] == "factoid":
            snippet_index = min(10,len(question["snippets"]))
            snippets = question["snippets"][0:snippet_index]
            snippets = [snippet['text'] for snippet in snippets]
            benchmark_questions.append([snippets,question['body']])
            benchmark_answers.append(question['exact_answer'])
            num_factoids += 1
        else:
            num_non_factoid += 1
    print(f"Benchmark contains {num_factoids + num_non_factoid} questions, made up of {question_types}")
    return benchmark_questions, benchmark_answers

def parse_MedQA(version="US"):
    #load raw data from benchmarks/MedQA-USMLE/US/train.jsonl, which has a dictionary on each line
    data = []
    with open('benchmarks/MedQA-USMLE/US/train.jsonl', 'r') as file:
        for line in file:
            record = json.loads(line)
            data.append(record)
    print("Loading Benchmark from MedQA-USMLE/US/train.jsonl")
    benchmark_questions = []
    benchmark_answers = []
    num_questions_with_5_options = 0
    num_questions_with_non_5_options = 0
    
    #for every question, ensure that there are 5 options, and add to output
    for instance in data:
        MCQ_question = instance["question"]
        if len(instance["options"].keys())== 5:
            num_questions_with_5_options += 1
        else:
            num_questions_with_non_5_options += 1
        for option in instance["options"].keys():
            MCQ_question += "\n" + "(" + option + ") " + instance["options"][option]
        MCQ_answer = "(" + instance['answer_idx'] + ") " + instance["answer"]
        benchmark_questions.append(MCQ_question)
        benchmark_answers.append(MCQ_answer)
    print("Benchmark contains " + str(len(data)) + " questions, made up of " + str(num_questions_with_5_options) + " with 5 options and " + str(num_questions_with_non_5_options) + " with non-5 options")
    return benchmark_questions, benchmark_answers

def parse_PubMedQA(version=""):
    #not yet supported
    return None, None

def parse_MedMCQA(version=""):
    #not yet supported
    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=str, help="Name of the benchmark to parse.")
    args = parser.parse_args()
    
    #call correct parsing function based on argument
    if(args.b == "bioASQ_no_snippet"):
        parse_bioASQ_no_snippet("5b")
    if(args.b == "bioASQ_with_snippet"):
        parse_bioASQ_with_snippet("5b")
    elif(args.b == "MedQA_US"):
        parse_MedQA("US")