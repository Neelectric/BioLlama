import box
import yaml
import glob
import faiss
import json

import pandas as pd
from io import StringIO
import datetime

with open('../config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def load_knowledge_db(knowledge_db_name):
    print("THIS METHOD SEEMS TO BE PROBLEMATIC. FOR NOW ITS FUNCTIONALITY IS COMMENTED OUT")
    print("Attempting to load FAISS index for " + cfg.DATABASE_AS_FAISS_PATH + knowledge_db_name + '.index')
    #this is what usually goes instea of the None faiss.read_index(cfg.DATABASE_AS_FAISS_PATH + knowledge_db_name + '.index')
    return None

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

def write_to_readme(model, benchmark, result):
    #read README file
    with open('../README.md', 'r') as file:
        readme = file.read()
    
    #split README into parts
    before_table, table, after_table = readme.split("<!-- table -->")
    print(table)

    #read table into a dataframe
    df = pd.read_csv(StringIO(table), sep='|')
    #delete the first row containing a bunch of -s
    df = df.iloc[1:]
    #print(df)

    #some column strings have excess whitespace, remove it
    df.columns = df.columns.str.strip()
    #same for row strings
    df['Model'] = df['Model'].str.strip()

    #edit cell where row 'Model' = model and column 'Benchmark' = benchmark
    df.loc[df['Model'] == model, benchmark] = result
    #print this change
    print(df.loc[df['Model'] == model, benchmark])
    
    #convert dataframe back to a markdown table
    df = df.drop(columns=['Unnamed: 0'])
    df = df.drop(columns=['Unnamed: 7'])
    new_table = df.to_markdown(index=False)
    #
    print(new_table)

    #
    before_changelog_after_table, changelog, after_changelog_after_table = after_table.split("<!-- changelog -->")

    #current date and time
    now = datetime.datetime.now()
    changelog += " * " + now.strftime("%Y-%m-%d %H:%M:%S") + " | " + model + " | " + benchmark + " | " + str(result) + "\n"

    after_table = before_changelog_after_table + '<!-- changelog -->\n' + changelog + "\n<!-- changelog -->"+ after_changelog_after_table

    #combine parts
    new_readme = before_table + '<!-- table -->\n' + new_table + "\n<!-- table -->"+ after_table
    print(new_readme)

    #write to README
    with open('../README.md', 'w') as file:
        file.write(new_readme)

    return

write_to_readme("BioLlama", "PubMedQA", 99.99 )

