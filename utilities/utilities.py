# Part of the BioLlama library
# Written by Neel Rajani
# Intended as primary place to keep utility functions like write_to_readme etc

import box
import yaml
import glob
import faiss
import json
import pandas as pd
from io import StringIO
import datetime
import pytz

#retired method
def load_benchmark(benchmark_filepath, type):
    with open('benchmarks/' + benchmark_filepath, 'rb') as json_file:
        json_data = json_file.read().decode('utf-8')
    data = json.loads(json_data)
    num = 0
    questions = []
    exact_answers = []
    for question in data['questions']:
        if question['type'] == type:
            num += 1
            questions.append(question['body'])
            exact_answers.append(question['exact_answer'])
    print("Returning " + str(num) + " questions.")
    return questions, exact_answers

def write_to_readme(model, benchmark, result, db_name, retrieval_text_mode, top_k, num_questions):
    if model == "GTE" or model == "MedCPT":
        model += "-Llama"
    with open('README.md', 'r') as file:
        readme = file.read()
    before_table, table, after_table = readme.split("<!-- table -->")
    if benchmark == "bioASQ_with_snippet":
        benchmark = "BioASQ5b (snippets)"
    elif benchmark == "bioASQ_no_snippet":
        print("Error! BioASQ no snippet not supported yet.")
        return
    
    #ensure that result only has 2 decimal places
    result = round(result, 2)

    #read table into a dataframe, delete first row & excess whitespace in column/row strings
    df = pd.read_csv(StringIO(table), sep='|')
    df = df.iloc[1:]
    df.columns = df.columns.str.strip()
    df['Model'] = df['Model'].str.strip()

    #take note of old value of cell, then change and print it
    old_result = df.loc[df['Model'] == model, benchmark].values[0]
    #strip whitespace from old result
    old_result = old_result.strip()
    df.loc[df['Model'] == model, benchmark] = result
    print("Changed " + str(old_result) + " to " + str(result) + " for " + model + " on " + benchmark)
    df = df.drop(columns=['Unnamed: 0'])
    df = df.drop(columns=['Unnamed: 8'])

    #reconvert dataframe to markdown
    new_table = df.to_markdown(index=False)
    before_changelog_after_table, changelog, after_changelog_after_table = after_table.split("<!-- changelog -->")
    machine_timezone = pytz.timezone(pytz.country_timezones['DE'][0])
    now = datetime.datetime.now(machine_timezone)
    strftime = now.strftime("%H:%M:%S, %d.%m.%Y")
    if model == "GTE-Llama" or model == "MedCPT-Llama":
        new_change = f" * {strftime} | {model} | {benchmark} | {old_result} --> {result} ({top_k}*{retrieval_text_mode} {db_name}), {num_questions} questions\n"
    else: 
        new_change = f" * {strftime} | {model} | {benchmark} | {old_result} --> {result}, {num_questions} questions\n"
    changelog = new_change + changelog
    after_table = before_changelog_after_table + '<!-- changelog -->\n' + changelog + "\n<!-- changelog -->"+ after_changelog_after_table
    new_readme = before_table + '<!-- table -->\n' + new_table + "\n<!-- table -->"+ after_table

    with open('README.md', 'w') as file:
        file.write(new_readme)
    return

if __name__ == "__main__":
    # write_to_readme("BioLlama", "PubMedQA", 99.99 )
    write_to_readme("BioLlama", "PubMedQA", 99.99 )