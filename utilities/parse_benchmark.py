# Part of the BioLlama library
# Written by Neel Rajani
# Creates support for parsing of benchmark questions from JSON files, for use in inference

import json
import argparse
import glob
import os

def parse_bioASQ_no_snippet(version="5b"):
    #read in raw benchmark
    with open('benchmarks/BioASQ/BioASQ-trainingDataset5b.json', 'rb') as json_file:
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

def parse_BioASQ_with_snippet(version="5b"):
    #read in raw benchmark
    with open('benchmarks/BioASQ/BioASQ-trainingDataset5b.json', 'rb') as json_file:
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
    
    # the following would support for the GoldenEnriched questions
    # thing is, this is the test set. so we dont really want to train on this!
    # filename_list = os.listdir('benchmarks/BioASQ/Task5BGoldenEnriched/')
    # enriched_data = []
    # for filename in filename_list:
    #     with open('benchmarks/BioASQ/Task5BGoldenEnriched/'+ filename, 'rb') as json_file:
    #         temp_data = json_file.read().decode('utf-8')
    #         enriched_data.append(json.loads(temp_data))

    # for set in enriched_data:
    #     for question in set["questions"]:
    #         if question["type"] not in question_types:
    #             question_types[question["type"]] = 1
    #         else:
    #             question_types[question["type"]] += 1
    #         if question["type"] == "factoid":
    #             snippet_index = min(10,len(question["snippets"]))
    #             snippets = question["snippets"][0:snippet_index]
    #             snippets = [snippet['text'] for snippet in snippets]
    #             benchmark_questions.append([snippets,question['body']])
    #             benchmark_answers.append(question['exact_answer'])
    #             num_factoids += 1
    #         else:
    #             num_non_factoid += 1
    

    print(f"Benchmark contains {num_factoids + num_non_factoid} questions, made up of {question_types}")
    return benchmark_questions, benchmark_answers

def parse_MedQA_4(version="US"):
    #load raw data from benchmarks/MedQA-USMLE/US/train.jsonl, which has a dictionary on each line
    data = []
    # txt_files = glob.glob("../benchmarks")
    # print(txt_files)
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    with open('benchmarks/MedQA-4-option/test.jsonl', 'r') as file:
        for line in file:
            record = json.loads(line)
            data.append(record)
    print("Loading Benchmark from MedQA-4-option/test.jsonl")
    benchmark_questions = []
    benchmark_answers = []
    num_questions_with_4_options = 0
    num_questions_with_non_4_options = 0
    
    #for every question, ensure that there are 5 options, and add to output
    for instance in data:
        MCQ_question = instance["question"]
        if len(instance["options"].keys())== 4:
            num_questions_with_4_options += 1
        else:
            num_questions_with_non_4_options += 1
        for option in instance["options"].keys():
            MCQ_question += "\n (" + option + ") " + instance["options"][option]
        MCQ_answer = "(" + instance['answer_idx'] + ") " + instance["answer"]
        benchmark_questions.append(MCQ_question)
        benchmark_answers.append(MCQ_answer)
    print("Benchmark contains " + str(len(data)) + " questions, made up of " + str(num_questions_with_4_options) + " with 4 options and " + str(num_questions_with_non_4_options) + " with non-4 options")
    return benchmark_questions, benchmark_answers

def parse_MedQA_5(version="US"):
    #load raw data from benchmarks/MedQA-USMLE/US/train.jsonl, which has a dictionary on each line
    data = []
    # txt_files = glob.glob("../benchmarks")
    # print(txt_files)
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    with open('benchmarks/MedQA-USMLE/US/test.jsonl', 'r') as file:
        for line in file:
            record = json.loads(line)
            data.append(record)
    print("Loading Benchmark from MedQA-USMLE/US/test.jsonl")
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
            MCQ_question += "\n (" + option + ") " + instance["options"][option]
        MCQ_answer = "(" + instance['answer_idx'] + ") " + instance["answer"]
        benchmark_questions.append(MCQ_question)
        benchmark_answers.append(MCQ_answer)
    print("Benchmark contains " + str(len(data)) + " questions, made up of " + str(num_questions_with_5_options) + " with 5 options and " + str(num_questions_with_non_5_options) + " with non-5 options")
    return benchmark_questions, benchmark_answers

def parse_PubMedQA(version=""):
    #read in raw benchmark
    with open('benchmarks/PubMedQA/ori_pqal.json', 'rb') as json_file:
        json_data = json_file.read().decode('utf-8')
    data = json.loads(json_data)
    #print(len(data.keys()))
    benchmark_questions = []
    benchmark_answers = []
    for key, val in data.items():
        benchmark_questions.append([val["CONTEXTS"],val["QUESTION"]])
        benchmark_answers.append(val["final_decision"])
    return benchmark_questions, benchmark_answers

def parse_MedMCQA(version="train.json"):
    #load raw data from benchmarks/MedMCQA/ + version
    if version == None: version = "train.json"
    data = []
    with open('benchmarks/MedMCQA/'+ version, 'r') as file:
        for line in file:
            # Parse the JSON object from the line and append it to the list
            json_obj = json.loads(line)
            data.append(json_obj)
    print(f"Loading Benchmark from MedMCQA/{version}.json")
    benchmark_questions = []
    benchmark_answers = []
    num_questions_single = 0
    num_questions_multiple = 0
    subject_names = {}

    for instance in data:
        #add topic name to dictionary
        if instance["subject_name"] not in subject_names:
            subject_names[instance["subject_name"]] = 1
        else:
            subject_names[instance["subject_name"]] += 1        
        question_output = instance["question"]
        if instance["choice_type"] == "single":
            num_questions_single += 1
        else:
            num_questions_multiple += 1
            
        question_output += "\n (1) " + instance["opa"]
        question_output += "\n (2) " + instance["opb"]
        question_output += "\n (3) " + instance["opc"]
        question_output += "\n (4) " + instance["opd"]
        benchmark_questions.append(question_output)

        #the answer index starts with 1, not with 0; all correct options should be in the range of [ 1, 2, 3, 4 ]
        benchmark_answers.append(str(instance["cop"]))
    num_total = num_questions_single + num_questions_multiple
    #divide num_questions_single by num_total to 2 decimal places
    percent_single = round(num_questions_single/num_total, 2) *100
    percent_multiple = round(num_questions_multiple/num_total, 2) *100
    print(f"Benchmark contains {len(data)} questions, made up to {percent_single}% with single answers and {percent_multiple}% with multiple answers")
    print(f"Only adding the {num_questions_single} questions with single answers to the benchmark")
    #uncomment the following to display the number of questions per subject
    # for subject in subject_names:
    #     print(f"{subject}: {subject_names[subject]}")
    return benchmark_questions, benchmark_answers

def parse_benchmark(benchmark, version = None):
    #call correct parsing function based on argument
    if(benchmark == "bioASQ_no_snippet"):
       benchmark_questions, benchmark_answers = parse_bioASQ_no_snippet("5b")
    elif(benchmark == "bioASQ_with_snippet"):
        benchmark_questions, benchmark_answers = parse_BioASQ_with_snippet("5b")
    elif(benchmark == "MedQA-4"):
        benchmark_questions, benchmark_answers = parse_MedQA_4("US")
    elif(benchmark == "MedQA-5"):
        benchmark_questions, benchmark_answers = parse_MedQA_5("US")
    elif(benchmark == "PubMedQA"):
        benchmark_questions, benchmark_answers = parse_PubMedQA()
    elif(benchmark == "MedMCQA"):
        benchmark_questions, benchmark_answers = parse_MedMCQA(version)
    return benchmark_questions, benchmark_answers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=str, help="Name of the benchmark to parse.")
    parser.add_argument('-r', type=str, help="Whether a random sample of 100 questions should be written to file.")
    args = parser.parse_args()
    
    #call correct parsing function based on argument
    if(args.b == "bioASQ_no_snippet"):
       benchmark_questions, benchmark_answers = parse_bioASQ_no_snippet("5b")
    if(args.b == "bioASQ_with_snippet"):
        benchmark_questions, benchmark_answers = parse_BioASQ_with_snippet("5b")
        max_length = 0
        max_snippet_num = 0
        for benchmark_question in benchmark_questions:
            snippets = benchmark_question[0]
            question = benchmark_question[1]
            length = len(question)
            if len(snippets) > max_snippet_num:
                max_snippet_num = len(snippets)
            for snippet in snippets:
                length += len(snippet)
            print(length)
            if length > max_length:
                max_length = length
        print(f"max length is {max_length}")
        print(f"max snippet num is {max_snippet_num}")
    elif(args.b == "MedQA_US"):
        benchmark_questions, benchmark_answers = parse_MedQA_5("US")
    elif(args.b == "PubMedQA"):
        benchmark_questions, benchmark_answers = parse_PubMedQA()
    elif(args.b == "MedMCQA"):
        benchmark_questions, benchmark_answers = parse_MedMCQA()
        for i in range(1):
            print(benchmark_questions[i])
            print(benchmark_answers[i])
        
#add r random benchmark questions to benchmarks/benchmark_random_samples
    if(args.r):
        r = int(args.r)
        import random
        random.seed(42)
        random_indices = random.sample(range(len(benchmark_questions)), r)
        random_questions = [benchmark_questions[i] for i in random_indices]
        random_answers = [benchmark_answers[i] for i in random_indices]
        random_sample = {"questions":random_questions, "answers":random_answers}
        #create a new json file with the random sample, for this specific benchmark
        with open(f'benchmarks/benchmark_random_samples/{args.b}_random_sample.json', 'w') as outfile:
            json.dump(random_sample, outfile)
        print(f"Wrote random sample of {r} questions to benchmarks/benchmark_random_samples/{args.b}_random_sample.json")

        
