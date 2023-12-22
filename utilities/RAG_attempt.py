# Part of the BioLlama library
# Written by Neel Rajani
# File not really in use but consists of initial attempts at RiP

import box
import yaml
import faiss
import json
import numpy as np
from utilities.utilities import load_benchmark, load_knowledge_db

from sentence_transformers import SentenceTransformer

with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


model = "Llama-2-70B-q4"
benchmark = "BioASQ-training5b/BioASQ-trainingDataset5b.json"
question_type = "factoid"
database_name = "train.txt"
k = 5

questions, exact_answers = load_benchmark(benchmark, question_type)
print("Loaded benchmark" + benchmark + " with " + str(len(questions)) + " questions and " + str(len(exact_answers)) + " answers.")
#knowledge_db = load_knowledge_db(database_name)
embedding_model = SentenceTransformer("thenlper/gte-large")
responses = []

# JSON_file_path = cfg.DATABASE_AS_JSON_PATH + database_name + '.json'
# print(JSON_file_path)
# with open(JSON_file_path, "r") as json_file:
#     # Load the JSON data into a Python dictionary
#     knowledge_db_as_JSON = json.load(json_file)

for question in questions[0:4]:
    print(question)
    question_embedding = np.array([embedding_model.encode(question)])
    # distances, indices = knowledge_db.search(question_embedding, k)
    # distances = distances.flatten()
    # indices = indices.flatten()
    # print("Distances shape: " + str(distances.shape))

    # for i in range(len(distances)):
    #     print("Distance: " + str(distances[i]) + " Index: " + str(indices[i]))
    #     print(knowledge_db_as_JSON[int(indices[i])])
    prompt = question + " " + "something"
    print("\n")
    