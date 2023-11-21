import box
import yaml
import glob
import faiss
import json

with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
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


def write_to_readme(result):
    return

