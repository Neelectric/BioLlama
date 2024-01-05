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
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
local_transformers = False
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def read_documents(db_name, mode):
    document = ""
    documents = []
    n_documents = 0
    n_lines = 0
    documents_dict = {}
    txt_files = glob.glob("../database/*.txt")
    print(txt_files)
    if "../database/" + db_name in txt_files:
        file_name = "../database/" + db_name
        print("Found the requested file, reading chunks")
    else:
        file_name = txt_files[0]
    if len(txt_files) != 1:
        #raise Exception("There should be exactly one .txt file in the data directory.")
        print("FYI: Found more than one .txt file in the data directory. Using file " + file_name)

    #for read_documents, input segmentation behaves the same as bomrc
    if mode == "input_segmentation":
        print("mode is input_segmentation, changing to bomrc")
        mode = "bomrc"
    full_lengths = []
    bomrc_lengths = []
    brc_lengths = []
    bc_lengths = []

    with open(file_name, 'r') as file:
        id = ""
        dict = {}
        for line in tqdm(file, desc="Processing chunks"):
            n_lines += 1
            if id == "": 
                id = line.strip()
                #background, objective, methods, results, conclusions
                dict = {'ID': id,
                        'BACKGROUND': '',
                        'OBJECTIVE': "",
                        'METHODS': "",
                        'RESULTS': "",
                        'CONCLUSIONS': ""}
            elif line.startswith('BACKGROUND'):
                dict['BACKGROUND'] += line[10:-2].strip() + ". "
            elif line.startswith('OBJECTIVE'):
                dict['OBJECTIVE'] += line[9:-2].strip() + ". "
            elif line.startswith('METHODS'):
                dict['METHODS'] += line[7:-2].strip() + ". "
            elif line.startswith('RESULTS'):
                dict['RESULTS'] += line[7:-2].strip() + ". "
            elif line.startswith('CONCLUSIONS'):
                dict['CONCLUSIONS'] += line[11:-2].strip() + ". "
            if not line.strip():
                b_o_m_r_c = dict['BACKGROUND'] + dict['OBJECTIVE'] + dict['METHODS'] + dict['RESULTS'] + dict['CONCLUSIONS']
                b_r_c = dict['BACKGROUND'] + dict['RESULTS'] + dict['CONCLUSIONS']
                b_c = dict['BACKGROUND'] + dict['CONCLUSIONS']
                #quick study on effect of removing extra words with dict
                #and then on only taking background, results, conclusions
                full_lengths.append(len(document))
                bomrc_lengths.append(len(b_o_m_r_c))
                brc_lengths.append(len(b_r_c))  
                bc_lengths.append(len(b_c)) 

                if mode == "full":
                    documents.append(document)
                    documents_dict[n_documents] = document
                elif mode == "bomrc" and b_o_m_r_c != "":
                    documents.append(b_o_m_r_c)
                    documents_dict[n_documents] = b_o_m_r_c
                elif mode == "brc" and b_r_c != "":
                    documents.append(b_r_c)
                    documents_dict[n_documents] = b_r_c
                elif mode == "bc" and b_c != "":
                    documents.append(b_c)
                    documents_dict[n_documents] = b_c
                n_documents += 1
                document = ""
                id = ""
            else:
                document += line
    
    #find average lengths
    full_avg = np.mean(full_lengths)
    bomrc_avg = np.mean(bomrc_lengths)
    brc_avg = np.mean(brc_lengths)
    print("full_avg: " + str(full_avg))
    print("bomrc_avg: " + str(bomrc_avg))
    print("brc_avg: " + str(brc_avg))
    print("bc_avg: " + str(np.mean(bc_lengths)))
    return documents, documents_dict

def prepare_folders(db_name, embedding_model, mode, chunk_length=None):
    #omit ".txt" from db_name
    db_name = db_name[0:-4]
    #in case the folders for the two datastores don't exist yet, we create them here:
    if mode == "input_segmentation":
        faiss_folder_path = "../vectorstores/" + db_name + "/" + embedding_model + "/" + mode + "/db_faiss_" + str(chunk_length) + "/"
        if not os.path.exists(faiss_folder_path):
            os.makedirs(faiss_folder_path)
        json_folder_path = "../vectorstores/" + db_name + "/" + embedding_model + "/" + mode + "/db_JSON_" + str(chunk_length) + "/"
        if not os.path.exists(json_folder_path):
            os.makedirs(json_folder_path)
    else:
        faiss_folder_path = "../vectorstores/" + db_name + "/" + embedding_model + "/" + mode + "/db_faiss/"
        if not os.path.exists(faiss_folder_path):
            os.makedirs(faiss_folder_path)
        json_folder_path = "../vectorstores/" + db_name + "/" + embedding_model + "/" + mode + "/db_JSON/"
        if not os.path.exists(json_folder_path):
            os.makedirs(json_folder_path)
    return faiss_folder_path, json_folder_path

def build_index_gte(db_name, mode):
    print("Building index with gte-large")
    chunks, chunks_dict = read_documents(db_name, "gte-large")
    faiss_folder_path, json_folder_path = prepare_folders(db_name, "gte-large", mode)
    db_name = db_name[0:-4]
    #build index
    print("len(chunks): " + str(len(chunks)))
    time_before_index = time.time()
    embedding_model = SentenceTransformer("thenlper/gte-large")
    chunk_embeddings = embedding_model.encode(chunks)
    print("Embeddings shape: " + str(chunk_embeddings.shape))
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)

    faiss_file_path = faiss_folder_path + db_name + '.index'
    print("Saving index to " + faiss_file_path)
    faiss.write_index(index, faiss_file_path)
    time_after_index = time.time()
    print("Time to build index: " + str(time_after_index - time_before_index) + " seconds.")

    #save as JSON objects too for convenience 
    json_file_path = json_folder_path + db_name + '.json'
    with open(json_file_path, "w") as json_file:
        json.dump(chunks_dict, json_file, indent=4)

def build_index_medcpt(db_name, mode, chunk_length):
    print("Building index with MedCPT")
    documents, documents_dict = read_documents(db_name, mode)
    faiss_folder_path, json_folder_path = prepare_folders(db_name, "medcpt", mode, chunk_length=chunk_length)
    db_name = db_name[0:-4]
    #build index
    print("len(documents): " + str(len(documents)))
    time_before_index = time.time()
    # if local_transformers:
    #     from ..finetuning.cti.transformers.transformers.src.transformers.models.auto import AutoTokenizer, AutoModel
    embedding_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
    embedding_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    #medcpt article encoder has 768 dimensions
    index = faiss.IndexFlatL2(768)

    if mode != "input_segmentation":
        #we get too many embeddings to do everything in RAM, so we do it in batches
        batch_size = 100
        for doc_num in tqdm(range(len(documents)//batch_size), desc="Batch Inference"):
            temp_documents = list(documents[doc_num*batch_size:(doc_num+1)*batch_size])
            with torch.no_grad():
                # tokenize the articles
                encoded = embedding_tokenizer(
                    temp_documents, 
                    truncation=True, 
                    padding=True, 
                    return_tensors='pt', 
                    max_length=512,
                )
                # encode the queries (use the [CLS] last hidden states as the representations)
                chunk_embeddings = embedding_model(**encoded).last_hidden_state[:, 0, :]
                index.add(chunk_embeddings)
    else:
        documents_dict = {}
        dict_counter = 0
        print("Performing input segmentation with chunk size " + str(chunk_length) + " tokens.")
        for doc_num in tqdm(range(len(documents))):
            # segment input chunks into chunks of length chunk_length, override json file
            document = documents[doc_num]
            encoded = embedding_tokenizer(
                document, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=512,
            )
            #print the number of tokens the tokenizer spits out
            # print("Number of tokens: " + str(len(encoded['input_ids'][0])))
            m = chunk_length
            #we want to split the document into m chunks, padding with empty strings if necessary
            # print("length of document: " + str(len(document)))
            #now that we have the tokens, we want to split them into chunks of length m. if the last chunk is less than m, we pad it with empty strings
            normal_iterations = len(encoded['input_ids'][0])//m
            # print("tokens divided by m: " + str(normal_iterations))

            #consider as many chunks of size m as possible
            for i in range(normal_iterations):
                # print("i: " + str(i))
                #now we consider tokens from i*m to (i+1)*m, and we want to get the corresponding text from document
                #print("encoded = " + str(encoded))
                chunk_tokens = encoded['input_ids'][0][i*m:(i+1)*m]
                #print number of tokens we have in chunk
                # print("chunk length: " + str(len(chunk_tokens)))
                chunk_text = embedding_tokenizer.decode(chunk_tokens)
                documents_dict[dict_counter] = chunk_text
                # print("chunk = " + str(chunk_text))
                chunk_embeddings = embedding_model(chunk_tokens.unsqueeze(0)).last_hidden_state.mean(dim=1).detach().numpy()
                index.add(chunk_embeddings)
                dict_counter += 1
            #check if there are tokens left over:
            if(len(encoded['input_ids'][0])%m != 0):
                #consider last chunk
                last_chunk_tokens = encoded['input_ids'][0][normal_iterations*m:]
                # print("last chunk length: " + str(len(last_chunk_tokens)))
                last_chunk_text = embedding_tokenizer.decode(last_chunk_tokens)
                # print("last chunk = " + str(last_chunk_text))
                last_chunk_embeddings = embedding_model(last_chunk_tokens.unsqueeze(0)).last_hidden_state.mean(dim=1).detach().numpy()
                index.add(last_chunk_embeddings)
                documents_dict[dict_counter] = chunk_text
                dict_counter += 1

    faiss_file_path = faiss_folder_path + db_name + '.index'
    json_file_path = json_folder_path + db_name + '.json'

    print("Saving index to " + faiss_file_path)
    faiss.write_index(index, faiss_file_path)
    time_after_index = time.time()
    print("Time to build index: " + str(time_after_index - time_before_index) + " seconds.")

    #save as JSON objects too for convenience 
    with open(json_file_path, "w") as json_file:
        json.dump(documents_dict, json_file, indent=4)

def load_db(embedding_model, db_name, retrieval_text_mode, chunk_length=None):
    if retrieval_text_mode == "input_segmentation":
        index_path_faiss = "vectorstores/" + db_name + "/" + embedding_model + "/"+ retrieval_text_mode + "/db_faiss_" + str(chunk_length) + "/" + db_name + '.index'
        index_path_json = "vectorstores/" + db_name + "/" + embedding_model + "/" + retrieval_text_mode + "/db_JSON_" + str(chunk_length) + "/" + db_name + '.json'
    else:
        index_path_faiss = "vectorstores/" + db_name + "/" + embedding_model + "/"+ retrieval_text_mode + "/db_faiss/" + db_name + '.index'
        index_path_json = "vectorstores/" + db_name + "/" + embedding_model + "/" + retrieval_text_mode + "/db_JSON/" + db_name + '.json'
    print("Attempting to load FAISS index for " + index_path_faiss)
    print(os.getcwd())
    with open(index_path_json, "r") as json_file:
        knowledge_db_as_JSON = json.load(json_file)
    return faiss.read_index(index_path_faiss), knowledge_db_as_JSON

def medcpt_FAISS_retrieval(questions, db_name, retrieval_text_mode, chunk_length=None):
    #print("questions we are given: " + str(questions))
    db_faiss, db_json = load_db("medcpt", db_name, retrieval_text_mode, chunk_length=chunk_length)
    # print(db_json["0"])
    # if local_transformers:
        # from ..finetuning.cti.transformers.transformers.src.transformers.models.auto import AutoTokenizer, AutoModel
        # from ..finetuning.cti.transformers.transformers.src.transformers.models.auto import AutoTokenizer, AutoModel
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

    #i will first try embedding of question and retrieval on a question by question basis, and time it
    #this is with arbitrary choices: k=1, max_length (how many tokens are in input i think?) = 480
    time_before_retrieval = time.time()
    k = 5
    top_k = 1
    chunk_list = []
    retrieval_quality = []
    #use tqdm here to get a progress bar
    for question in tqdm(questions, desc="Retrieving chunks"):
        # print(question)
        chunks = []
        with torch.no_grad():
            # tokenize the queries
            encoded = tokenizer(
                question, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=512,
            )
            # encode the queries (use the [CLS] last hidden states as the representations)
            embeds = model(**encoded).last_hidden_state[:, 0, :]
        distances, indices = db_faiss.search(embeds, k)
        distances = distances.flatten()
        indices = indices.flatten()
        # print(indices)

        if k>1:
            #reranking step
            # if local_transformers:
            #     from ..finetuning.cti.transformers.transformers.src.transformers.models.auto import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
            rerank_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
            rerank_model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")
            chunks = [db_json[str(indices[i])] for i in range(len(distances))]
            # print("chunks: " + str(chunks))
            pairs = [[question, chunk] for chunk in chunks]
            with torch.no_grad():
                encoded = rerank_tokenizer(
                    pairs, 
                    truncation=True, 
                    padding=True, 
                    return_tensors='pt', 
                    max_length=512,
                )
                # encode the queries (use the [CLS] last hidden states as the representations)
                logits = rerank_model(**encoded).logits.squeeze(dim =1).numpy()
            # print("logits: " + str(logits))
            #in logits, print the index of the last score which is positive
            
            #"logits" now gives us relevance scores. we want to use to resort the chunks array
            #by the relevance scores in logits, where higher relevance should be first
            #we can do this by sorting the indices of logits, and then using those indices to sort chunks
            sorted_scores = sorted(zip(chunks, logits), key=lambda x: x[1], reverse=True)
            sorted_indices = np.array([x[1] for x in sorted_scores])
            # print("logits after: " + str(sorted_indices))
            # print("last positive score: ")
            if np.where(sorted_indices>0)[0].size>0:
                last_positive = np.where(sorted_indices>0)[0][-1]
            else:
                # print("None")
                last_positive = 0
                pass
            # print(last_positive)
            chunks = [x[0] for x in sorted_scores]
            print(chunks[0:5])
            top_chunk = chunks[0]
            # print(chunks)
            retrieval_quality.append(sorted_indices[0])

        chunk_list.append(top_chunk)   
    time_after_retrieval = time.time()
    retrieval_quality = np.array(retrieval_quality)
    print(retrieval_quality)
    avg_retrieval_quality = np.mean(retrieval_quality)
    print("Avg retrieval quality: " + str(avg_retrieval_quality))
    print("Time to retrieve chunks: " + str(time_after_retrieval - time_before_retrieval) + " seconds.")
    return chunk_list
    
def gte_FAISS_retrieval(questions, db_name, retrieval_text_mode):
    db_faiss, db_json = load_db("gte-large", db_name, retrieval_text_mode)
    #if db is a faiss index, print that
    print("db: " + str(db_faiss))
    time_before_retrieval = time.time()
    k = 5
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
            print("Chunk: " + str(db_json[str(indices[i])]))
            chunks.append(db_json[str(indices[i])])
        chunk_list.append(chunks)        
    time_after_retrieval = time.time()
    print("Time to retrieve chunks: " + str(time_after_retrieval - time_before_retrieval) + " seconds.")
    return chunk_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default="RCT20ktrain.txt", help="Name of the database to build index for.")
    parser.add_argument('--embedding_type', type=str, help='Type of embedding to use.')
    parser.add_argument('--mode', type=str, default="full", help='Whether to embed full text or combo of background, objective, methods, results, conclusions.')
    parser.add_argument('--chunk_length', type=int, default=16, help='Length of chunks to embed.')
    args = parser.parse_args()
    if args.embedding_type == "gte-large":
        build_index_gte(args.db_name, args.mode)
    elif args.embedding_type =="medcpt":
        build_index_medcpt(args.db_name, args.mode, args.chunk_length)
        # read_chunks("RCT200ktrain.txt", "medcpt", "sixteen")
    elif args.embedding_type =="debug":
        questions = ["Which is the main calcium pump of the sarcoplasmic reticulum?"]
        medcpt_FAISS_retrieval(questions=questions, db_name="RCT200ktrain", retrieval_text_mode="input_segmentation", chunk_length=16)
# questions = ["Which is the main calcium pump of the sarcoplasmic reticulum?"]
# medcpt_FAISS_retrieval(questions=questions, db_name="RCT200ktrain", retrieval_text_mode="input_segmentation", chunk_length=16)