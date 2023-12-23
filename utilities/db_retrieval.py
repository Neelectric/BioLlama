# Part of the BioLlama library
# Adapted from https://github.com/kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference, which has an MIT License
# Implements building a FAISS index on text files, loading this index at inference and performing retrieval

import box
import yaml
import glob
import argparse
import time
import json
import faiss
import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def read_chunks(db_name):
    chunk = ""
    chunks = []
    n_chunks = 0
    n_lines = 0
    chunks_dict = {}
    txt_files = glob.glob("../database/*.txt")
    print(txt_files)
    if db_name in txt_files:
        file_name = db_name
        print("Found the requested file, reading chunks")
    else:
        file_name = txt_files[0]
    if len(txt_files) != 1:
        raise Exception("There should be exactly one .txt file in the data directory.")
    with open(file_name, 'r') as file:
        for line in tqdm(file, desc="Processing chunks",):
            n_lines += 1
            if not line.strip():
                chunks.append(chunk)
                chunks_dict[n_chunks] = chunk
                n_chunks += 1
                chunk = ""
            else:
                chunk += line
    return chunks, chunks_dict

def prepare_folders(db_name, embedding_model):
    #omit ".txt" from db_name
    db_name = db_name[0:-4]
    #in case the folders for the two datastores don't exist yet, we create them here:
    faiss_folder_path = "../vectorstores/" + db_name + "/" + embedding_model + "/db_faiss/"
    if not os.path.exists(faiss_folder_path):
        os.makedirs(faiss_folder_path)
    json_folder_path = "../vectorstores/" + db_name + "/" + embedding_model + "/db_JSON/"
    if not os.path.exists(json_folder_path):
        os.makedirs(json_folder_path)
    return faiss_folder_path, json_folder_path

#builds a FAISS index for a given database
def build_index_gte(db_name):
    print("Building index with gte-large")
    chunks, chunks_dict = read_chunks(db_name)
    faiss_folder_path, json_folder_path = prepare_folders(db_name, "gte-large")
    db_name = db_name[0:-4]
    #build index
    print("len(chunks): " + str(len(chunks)))
    time_before_index = time.time()
    embedding_model = SentenceTransformer("thenlper/gte-large")
    chunk_embeddings = embedding_model.encode(chunks)
    print("Embeddings shape: " + str(chunk_embeddings.shape))
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    print("Saving index to " + faiss_folder_path + db_name + '.index')
    faiss.write_index(index, faiss_folder_path + db_name + '.index')
    time_after_index = time.time()
    print("Time to build index: " + str(time_after_index - time_before_index) + " seconds.")

    #save as JSON objects too for convenience 
    json_file_path = json_folder_path + db_name + '.json'
    with open(json_file_path, "w") as json_file:
        json.dump(chunks_dict, json_file, indent=4)

def build_index_medcpt(db_name):
    print("Building index with MedCPT")
    chunks, chunks_dict = read_chunks(db_name)
    faiss_folder_path, json_folder_path = prepare_folders(db_name, "medcpt")
    db_name = db_name[0:-4]
    #build index
    print("len(chunks): " + str(len(chunks)))
    time_before_index = time.time()
    embedding_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
    embedding_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    #medcpt article encoder has 768 dimensions
    index = faiss.IndexFlatL2(768)

    #we get too many embeddings to do everything in RAM, so we do it in batches
    batch_size = 100
    for i in tqdm(range(len(chunks)//batch_size), desc="Batch Inference"):
        temp_chunks = list(chunks[i*batch_size:(i+1)*batch_size])
        with torch.no_grad():
            # tokenize the articles
            encoded = embedding_tokenizer(
                temp_chunks, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=512,
            )
            # encode the queries (use the [CLS] last hidden states as the representations)
            chunk_embeddings = embedding_model(**encoded).last_hidden_state[:, 0, :]
            index.add(chunk_embeddings)

    print("Saving index to " + faiss_folder_path + db_name + '.index')
    faiss.write_index(index, faiss_folder_path + db_name + '.index')
    time_after_index = time.time()
    print("Time to build index: " + str(time_after_index - time_before_index) + " seconds.")

    #save as JSON objects too for convenience 
    json_file_path = json_folder_path + db_name + '.json'
    with open(json_file_path, "w") as json_file:
        json.dump(chunks_dict, json_file, indent=4)


def load_db(embedding_model, db_name):
    index_path_faiss = "vectorstores/" + db_name + "/" + embedding_model +"/db_faiss/" + db_name + '.index'
    index_path_json = "vectorstores/" + db_name + "/" + embedding_model + "/db_JSON/" + db_name + '.json'
    print("Attempting to load FAISS index for " + index_path_faiss)
    with open(index_path_json, "r") as json_file:
        knowledge_db_as_JSON = json.load(json_file)
    return faiss.read_index(index_path_faiss), knowledge_db_as_JSON

def medcpt_FAISS_retrieval(questions, db_name):
    db_faiss, db_json = load_db("medcpt", db_name)
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    #i will first try embedding of question and retrieval on a question by question basis, and time it
    #this is with arbitrary choices: k=1, max_length (how many tokens are in input i think?) = 480
    time_before_retrieval = time.time()
    k = 1
    chunk_list = []
    for question in questions:
        chunks = []
        with torch.no_grad():
            # tokenize the queries
            encoded = tokenizer(
                question, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=480,
            )
            # encode the queries (use the [CLS] last hidden states as the representations)
            embeds = model(**encoded).last_hidden_state[:, 0, :]
        #question_embedding = np.array([embeds])
        distances, indices = db_faiss.search(embeds, k)
        distances = distances.flatten()
        indices = indices.flatten()
        print("Distances shape: " + str(distances.shape))
        for i in range(len(distances)):
            print("Distance: " + str(distances[i]) + " Index: " + str(indices[i]))
            chunks.append(db_json[str(indices[i]+1)])
        chunk_list.append(chunks)   
    time_after_retrieval = time.time()
    print("Time to retrieve chunks: " + str(time_after_retrieval - time_before_retrieval) + " seconds.")
    return chunk_list
    
def gte_FAISS_retrieval(questions, db_name):
    db_faiss, db_json = load_db("gte-large", db_name)
    #if db is a faiss index, print that
    print("db: " + str(db_faiss))
    k = 1
    chunk_list = []
    embedding_model = SentenceTransformer("thenlper/gte-large")
    for question in questions:
        chunks = []
        question_embedding = np.array([embedding_model.encode(question)])
        distances, indices = db_faiss.search(question_embedding, k)
        distances = distances.flatten()
        indices = indices.flatten()
        print("Distances shape: " + str(distances.shape))
        for i in range(len(distances)):
            print("Distance: " + str(distances[i]) + " Index: " + str(indices[i]))
            chunks.append(db_json[str(indices[i]+1)])
        chunk_list.append(chunks)        
    return chunk_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default="RCT20ktrain.txt", help="Name of the database to build index for.")
    parser.add_argument('--embedding_type', type=str, default="gte-large", help='Type of embedding to use.')
    args = parser.parse_args()
    if args.embedding_type == "gte-large":
        build_index_gte(args.db_name)
    elif args.embedding_type =="medcpt":
        build_index_medcpt(args.db_name)
