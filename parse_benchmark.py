import json
import argparse
from collections import Counter

def parse_bioASQ(version="5b"):
    with open('benchmarks/BioASQ-training5b/BioASQ-trainingDataset5b.json', 'rb') as json_file:
        json_data = json_file.read().decode('utf-8')
    data = json.loads(json_data)
    num_factoids = 0
    num_non_factoid = 0
    benchmark_questions = []
    benchmark_answers = []
    for question in data["questions"]:
        if question["type"] == "factoid":
            benchmark_questions.append(question['body'])
            benchmark_answers.append(question['exact_answer'])
            num_factoids += 1
        else:
            num_non_factoid += 1
    print("Benchmark contains " + str(num_factoids + num_non_factoid) + " questions, made up of " + str(num_factoids) + " factoid questions and " + str(num_non_factoid) + " non-factoid questions")
    return benchmark_questions, benchmark_answers

def parse_MedQA(version="US"):
    #load data from benchmarks/MedQA-USMLE/US/train.jsonl, which has a dictionary on each line
    data = []
    with open('benchmarks/MedQA-USMLE/US/train.jsonl', 'r') as file:
        for line in file:
            # Load each line as a JSON object (dictionary)
            record = json.loads(line)
            data.append(record)
    print("Loading Benchmark from MedQA-USMLE/US/train.jsonl")
    benchmark_questions = []
    benchmark_answers = []
    num_questions_with_5_options = 0
    num_questions_with_non_5_options = 0
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=str, help="Name of the benchmark to parse.")
    args = parser.parse_args()
    if(args.b == "bioASQ"):
        parse_bioASQ("5b")
    elif(args.b == "MedQA_US"):
        parse_MedQA("US")