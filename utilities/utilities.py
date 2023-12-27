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

# with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
#     cfg = box.Box(yaml.safe_load(ymlfile))

#retired method
def load_knowledge_db(knowledge_db_name):
    print("THIS METHOD SEEMS TO BE PROBLEMATIC. FOR NOW ITS FUNCTIONALITY IS COMMENTED OUT")
    print("Attempting to load FAISS index for " + cfg.DATABASE_AS_FAISS_PATH + knowledge_db_name + '.index')
    #this is what usually goes instea of the None 
    # faiss.read_index(cfg.DATABASE_AS_FAISS_PATH + knowledge_db_name + '.index')
    return None

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

def write_to_readme(model, benchmark, result, db_name, retrieval_text_mode):
    with open('README.md', 'r') as file:
        readme = file.read()
    before_table, table, after_table = readme.split("<!-- table -->")

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
    df = df.drop(columns=['Unnamed: 7'])

    #reconvert dataframe to markdown
    new_table = df.to_markdown(index=False)

    #prepare combinations & new changelog, then write result
    before_changelog_after_table, changelog, after_changelog_after_table = after_table.split("<!-- changelog -->")
    #print all three components
    # print("before changelog: " + before_changelog_after_table)
    # print("changelog: " + changelog)
    # print("after changelog: " + after_changelog_after_table)

    machine_timezone = pytz.timezone(pytz.country_timezones['DE'][0])

    now = datetime.datetime.now(machine_timezone)
    new_change = " * " + now.strftime("%H:%M:%S, %d.%m.%Y") + " | " + model + " | " + benchmark + " | " + str(old_result) + " --> " + str(result) + " (1*" + retrieval_text_mode + " " + db_name + ")\n"
    changelog = new_change + changelog
    after_table = before_changelog_after_table + '<!-- changelog -->\n' + changelog + "\n<!-- changelog -->"+ after_changelog_after_table
    new_readme = before_table + '<!-- table -->\n' + new_table + "\n<!-- table -->"+ after_table
    # print(new_table)
    # print(changelog)
    with open('README.md', 'w') as file:
        file.write(new_readme)
    return

if __name__ == "__main__":
    write_to_readme("BioLlama", "PubMedQA", 99.99 )