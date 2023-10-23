import json
import re
import argparse

def standardize_string_PubMedQA(string):
    return re.sub(r'[. ]+', '', string.strip()).lower()

def mark_MedQA(input):
    num_correct = 0
    num_total = 0
    for i in range(len(input)):
        num_total += 1
        if input[i][1] == input[i][2]:
            num_correct += 1

    print("Accuracy is " + str(num_correct/num_total) + " with a total of " + str(num_total) + " questions.")

def mark_PubMedQA(input):
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
            #print(student_response + "\n")
    print(f"Marking model {model_to_grade} performance on benchmark {benchmark_to_grade}")
    print(f"Out of {num_total}, accuracy is {num_correct/num_total} with a total of {num_correct} correct, {num_incorrect} incorrect and {num_invalid} invalid responses.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, help="Name of the model that answered questions.")
    parser.add_argument('-b', type=str, help="Name of the benchmark that was used.")
    args = parser.parse_args()

    model_to_grade = args.m
    benchmark_to_grade = args.b
    output_file = "output/" + model_to_grade + "-" + benchmark_to_grade + ".json"

    with open(output_file, 'rb') as json_file:
        json_data = json_file.read().decode('utf-8')
    data = json.loads(json_data)    
    #call correct parsing function based on argument
    if(args.b == "MedQA"):
        mark_MedQA(data)
    elif(args.b == "PubMedQA"):
        mark_PubMedQA(data)
