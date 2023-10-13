import json
import argparse
from collections import Counter

def parse_bioASQ(version):

    with open('benchmarks/BioASQ-training5b/BioASQ-trainingDataset5b.json', 'rb') as json_file:
        json_data = json_file.read().decode('utf-8')
    data = json.loads(json_data)
    num_factoids = 0
    num_non_factoid = 0
    benchmark = []
    for question in data["questions"]:
        if question["type"] == "factoid":
            factoid_question = [question['body'], question['exact_answer']]
            benchmark.append(benchmark)
            num_factoids += 1
        else:
            num_non_factoid += 1
    print("processed " + str(num_factoids) + " factoid questions")
    print("processed " + str(num_non_factoid) + " non-factoid questions")
    print("this adds up to " + str(num_factoids + num_non_factoid) + " questions")
    return ["factoid", benchmark]

def parse_MedQA(version):
    #load data from benchmarks/MedQA-USMLE/US/train.jsonl, which has a dictionary on each line
    data = []
    with open('benchmarks/MedQA-USMLE/US/train.jsonl', 'r') as file:
        for line in file:
            # Load each line as a JSON object (dictionary)
            record = json.loads(line)
            data.append(record)
    print("loaded " + str(len(data)) + " questions from MedQA-USMLE/US/train.jsonl")
    #print(data[0])
    benchmark_questions = []
    benchmark_answers = []
    for instance in data:
        MCQ_question = instance["question"]
        for option in instance["options"].keys():
            MCQ_question += "\n" + "(" + option + ") " + instance["options"][option]
        MCQ_answer = "(" + instance['answer_idx'] + ") " + instance["answer"]
        benchmark_questions.append(MCQ_question)
        benchmark_answers.append(MCQ_answer)

    return benchmark_questions, benchmark_answers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=str, help="Name of the benchmark to parse.")
    args = parser.parse_args()
    if(args.b == "bioASQ"):
        parse_bioASQ("5b")
    elif(args.b == "MedQA_US"):
        parse_MedQA("US")