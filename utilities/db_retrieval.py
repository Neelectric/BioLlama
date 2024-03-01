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
import autofaiss

local_transformers = False
if local_transformers:
    from .finetuning.cti.transformers.transformers.src.transformers.models.auto import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
    )
else:
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
    )
import numpy as np
from tqdm import tqdm


def read_documents(db_name, mode, return_full_dict=False):
    document = ""
    documents = []
    n_documents = 0
    n_lines = 0
    documents_dict = {}
    # print(os.getcwd())
    correct_prefix = None
    # correct_prefix = "../"
    if correct_prefix != None:
        database_directory = correct_prefix + "database/"
        database_directory_with_suffix = database_directory + "*.txt"
    else:
        database_directory = "database/"
        database_directory_with_suffix = "database/*.txt"

    txt_files = glob.glob(database_directory_with_suffix)
    # print(txt_files)

    if database_directory + db_name + ".txt" in txt_files:
        file_name = database_directory + db_name + ".txt"
        # print("Found the requested file, reading chunks")
    else:
        file_name = txt_files[0]
    if len(txt_files) != 1:
        # raise Exception("There should be exactly one .txt file in the data directory.")
        print(
            "FYI: Found more than one .txt file in the data directory. Using file "
            + file_name
        )

    # for read_documents, input segmentation behaves the same as bomrc
    if mode == "input_segmentation":
        # print("mode is input_segmentation, changing to bomrc")
        mode = "bomrc"
    full_lengths = []
    bomrc_lengths = []
    brc_lengths = []
    bc_lengths = []

    list_of_dicts = []
    # the barebones version of this iteration (ie adding lines to a given chunk until an empty line is reached) was written by Copilot
    # the specifics, however, ie using the dict and then computing averages at the end, was written by me (Neel Rajani)
    with open(file_name, "r") as file:
        id = ""
        dict = {}
        for line in tqdm(file, desc="Processing chunks"):
            n_lines += 1
            if id == "":
                id = line.strip()
                # background, objective, methods, results, conclusions
                dict = {
                    "ID": id,
                    "BACKGROUND": "",
                    "OBJECTIVE": "",
                    "METHODS": "",
                    "RESULTS": "",
                    "CONCLUSIONS": "",
                }
            elif line.startswith("BACKGROUND"):
                dict["BACKGROUND"] += line[10:-2].strip() + ". "
            elif line.startswith("OBJECTIVE"):
                dict["OBJECTIVE"] += line[9:-2].strip() + ". "
            elif line.startswith("METHODS"):
                dict["METHODS"] += line[7:-2].strip() + ". "
            elif line.startswith("RESULTS"):
                dict["RESULTS"] += line[7:-2].strip() + ". "
            elif line.startswith("CONCLUSIONS"):
                dict["CONCLUSIONS"] += line[11:-2].strip() + ". "
            if not line.strip():
                b_o_m_r_c = (
                    dict["BACKGROUND"]
                    + dict["OBJECTIVE"]
                    + dict["METHODS"]
                    + dict["RESULTS"]
                    + dict["CONCLUSIONS"]
                )
                b_r_c = dict["BACKGROUND"] + dict["RESULTS"] + dict["CONCLUSIONS"]
                b_c = dict["BACKGROUND"] + dict["CONCLUSIONS"]
                full_lengths.append(len(document))
                bomrc_lengths.append(len(b_o_m_r_c))
                brc_lengths.append(len(b_r_c))
                bc_lengths.append(len(b_c))

                if return_full_dict:
                    list_of_dicts.append(dict)

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
    if return_full_dict:
        documents_dict = list_of_dicts

    # find average lengths
    full_avg = np.mean(full_lengths)
    bomrc_avg = np.mean(bomrc_lengths)
    brc_avg = np.mean(brc_lengths)
    # print("full_avg: " + str(full_avg))
    # print("bomrc_avg: " + str(bomrc_avg))
    # print("brc_avg: " + str(brc_avg))
    # print("bc_avg: " + str(np.mean(bc_lengths)))
    return documents, documents_dict


def prepare_folders(db_name, embedding_model, mode, chunk_length=None):
    # omit ".txt" from db_name
    db_name = db_name[0:-4]
    # in case the folders for the two datastores don't exist yet, we create them here:
    if mode == "input_segmentation":
        faiss_folder_path = (
            "../vectorstores/"
            + db_name
            + "/"
            + embedding_model
            + "/"
            + mode
            + "/db_faiss_"
            + str(chunk_length)
            + "/"
        )
        if not os.path.exists(faiss_folder_path):
            os.makedirs(faiss_folder_path)
        json_folder_path = (
            "../vectorstores/"
            + db_name
            + "/"
            + embedding_model
            + "/"
            + mode
            + "/db_JSON_"
            + str(chunk_length)
            + "/"
        )
        if not os.path.exists(json_folder_path):
            os.makedirs(json_folder_path)
    else:
        faiss_folder_path = (
            "../vectorstores/"
            + db_name
            + "/"
            + embedding_model
            + "/"
            + mode
            + "/db_faiss/"
        )
        if not os.path.exists(faiss_folder_path):
            os.makedirs(faiss_folder_path)
        json_folder_path = (
            "../vectorstores/"
            + db_name
            + "/"
            + embedding_model
            + "/"
            + mode
            + "/db_JSON/"
        )
        if not os.path.exists(json_folder_path):
            os.makedirs(json_folder_path)
    return faiss_folder_path, json_folder_path


def build_index_gte(db_name, mode):
    print("Building index with gte-large")
    chunks, chunks_dict = read_documents(db_name, "gte-large")
    faiss_folder_path, json_folder_path = prepare_folders(db_name, "gte-large", mode)
    db_name = db_name[0:-4]
    # build index
    print("len(chunks): " + str(len(chunks)))
    time_before_index = time.time()
    embedding_model = SentenceTransformer("thenlper/gte-large")
    chunk_embeddings = embedding_model.encode(chunks)
    print("Embeddings shape: " + str(chunk_embeddings.shape))
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)

    faiss_file_path = faiss_folder_path + db_name + ".index"
    print("Saving index to " + faiss_file_path)
    faiss.write_index(index, faiss_file_path)
    time_after_index = time.time()
    print(
        "Time to build index: "
        + str(time_after_index - time_before_index)
        + " seconds."
    )

    # save as JSON objects too for convenience
    json_file_path = json_folder_path + db_name + ".json"
    with open(json_file_path, "w") as json_file:
        json.dump(chunks_dict, json_file, indent=4)


# a bunch of code in here is probably not really necessary
def build_lookupchart_medcpt(db_name, mode, chunk_length):
    print("Building index with MedCPT")
    documents, documents_dict = read_documents(db_name, mode)
    # faiss_folder_path, json_folder_path = prepare_folders(db_name, "medcpt", mode, chunk_length=chunk_length)
    db_name = db_name
    print("len(documents): " + str(len(documents)))
    time_before_index = time.time()

    embedding_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    documents_dict = {}
    dict_counter = 0
    print(
        "Performing input segmentation with chunk size "
        + str(chunk_length)
        + " tokens."
    )

    lookupchart = {}

    for doc_num in tqdm(range(len(documents))):
        document = documents[doc_num]
        encoded = embedding_tokenizer(
            document,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )
        m = chunk_length
        normal_iterations = len(encoded["input_ids"][0]) // m
        for i in range(normal_iterations):
            chunk_tokens = encoded["input_ids"][0][i * m : (i + 1) * m]
            chunk_text = embedding_tokenizer.decode(chunk_tokens)
            documents_dict[dict_counter] = chunk_text
            lookupchart[dict_counter] = doc_num
            dict_counter += 1
        if len(encoded["input_ids"][0]) % m != 0:
            # consider last chunk
            last_chunk_tokens = encoded["input_ids"][0][normal_iterations * m :]
            last_chunk_text = embedding_tokenizer.decode(last_chunk_tokens)
            documents_dict[dict_counter] = chunk_text
            lookupchart[dict_counter] = doc_num
            dict_counter += 1

    json_folder_path = (
        "../vectorstores/"
        + db_name
        + "/"
        + "medcpt"
        + "/"
        + mode
        + "/db_JSON_"
        + str(chunk_length)
        + "/"
    )
    json_file_path = json_folder_path[3:] + db_name + "lookupchart" + ".json"
    time_after_index = time.time()
    print(
        "Time to build index: "
        + str(time_after_index - time_before_index)
        + " seconds."
    )

    # save as JSON objects too for convenience
    with open(json_file_path, "w") as json_file:
        json.dump(lookupchart, json_file, indent=4)

    # lets take chunk index 999 and check which document it is from
    # chunk_id = 999
    # abstract_id = lookupchart[chunk_id]
    # abstract = documents_dict[abstract_id]
    # print(f"chunk with id 999 comes from abstract {abstract_id}")
    # print(f"its full text is {abstract}")


def build_index_medcpt(db_name, mode, chunk_length):
    print("Building index with MedCPT")
    documents, documents_dict = read_documents(db_name, mode)
    faiss_folder_path, json_folder_path = prepare_folders(
        db_name, "medcpt", mode, chunk_length=chunk_length
    )
    db_name = db_name[0:-4]
    # build index
    print("len(documents): " + str(len(documents)))
    time_before_index = time.time()

    embedding_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
    embedding_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    # medcpt article encoder has 768 dimensions
    index = faiss.IndexFlatL2(768)

    if mode != "input_segmentation":
        # we get too many embeddings to do everything in RAM, so we do it in batches
        batch_size = 100
        for doc_num in tqdm(
            range(len(documents) // batch_size), desc="Batch Inference"
        ):
            temp_documents = list(
                documents[doc_num * batch_size : (doc_num + 1) * batch_size]
            )
            with torch.no_grad():
                # tokenize the articles
                encoded = embedding_tokenizer(
                    temp_documents,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                # encode the queries (use the [CLS] last hidden states as the representations)
                chunk_embeddings = embedding_model(**encoded).last_hidden_state[:, 0, :]
                index.add(chunk_embeddings)
    else:
        documents_dict = {}
        dict_counter = 0
        print(
            "Performing input segmentation with chunk size "
            + str(chunk_length)
            + " tokens."
        )
        for doc_num in tqdm(range(len(documents))):
            # segment input chunks into chunks of length chunk_length, override json file
            document = documents[doc_num]
            encoded = embedding_tokenizer(
                document,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            # print the number of tokens the tokenizer spits out
            # print("Number of tokens: " + str(len(encoded['input_ids'][0])))
            m = chunk_length
            # we want to split the document into m chunks, padding with empty strings if necessary
            # print("length of document: " + str(len(document)))
            # now that we have the tokens, we want to split them into chunks of length m. if the last chunk is less than m, we pad it with empty strings
            normal_iterations = len(encoded["input_ids"][0]) // m
            # print("tokens divided by m: " + str(normal_iterations))

            # consider as many chunks of size m as possible
            for i in range(normal_iterations):
                # print("i: " + str(i))
                # now we consider tokens from i*m to (i+1)*m, and we want to get the corresponding text from document
                # print("encoded = " + str(encoded))
                chunk_tokens = encoded["input_ids"][0][i * m : (i + 1) * m]
                # print number of tokens we have in chunk
                # print("chunk length: " + str(len(chunk_tokens)))
                chunk_text = embedding_tokenizer.decode(chunk_tokens)
                documents_dict[dict_counter] = chunk_text
                # print("chunk = " + str(chunk_text))
                chunk_embeddings = (
                    embedding_model(chunk_tokens.unsqueeze(0))
                    .last_hidden_state.mean(dim=1)
                    .detach()
                    .numpy()
                )
                index.add(chunk_embeddings)
                dict_counter += 1
            # check if there are tokens left over:
            if len(encoded["input_ids"][0]) % m != 0:
                # consider last chunk
                last_chunk_tokens = encoded["input_ids"][0][normal_iterations * m :]
                # print("last chunk length: " + str(len(last_chunk_tokens)))
                last_chunk_text = embedding_tokenizer.decode(last_chunk_tokens)
                # print("last chunk = " + str(last_chunk_text))
                last_chunk_embeddings = (
                    embedding_model(last_chunk_tokens.unsqueeze(0))
                    .last_hidden_state.mean(dim=1)
                    .detach()
                    .numpy()
                )
                index.add(last_chunk_embeddings)
                documents_dict[dict_counter] = chunk_text
                dict_counter += 1

    faiss_file_path = faiss_folder_path + db_name + ".index"
    json_file_path = json_folder_path + db_name + ".json"

    print("Saving index to " + faiss_file_path)
    faiss.write_index(index, faiss_file_path)
    time_after_index = time.time()
    print(
        "Time to build index: "
        + str(time_after_index - time_before_index)
        + " seconds."
    )

    # save as JSON objects too for convenience
    with open(json_file_path, "w") as json_file:
        json.dump(documents_dict, json_file, indent=4)


def load_db(embedding_model, db_name, retrieval_text_mode, chunk_length=None):
    if retrieval_text_mode == "input_segmentation":
        index_path_faiss = ("vectorstores/" + db_name + "/" + embedding_model + "/" + retrieval_text_mode + "/db_faiss_" + str(chunk_length) + "/" + db_name + ".index")
        index_path_json = (
            "vectorstores/"
            + db_name
            + "/"
            + embedding_model
            + "/"
            + retrieval_text_mode
            + "/db_JSON_"
            + str(chunk_length)
            + "/"
            + db_name
            + ".json"
        )
    else:
        index_path_faiss = (
            "vectorstores/"
            + db_name
            + "/"
            + embedding_model
            + "/"
            + retrieval_text_mode
            + "/db_faiss/"
            + db_name
            + ".index"
        )
        index_path_json = (
            "vectorstores/"
            + db_name
            + "/"
            + embedding_model
            + "/"
            + retrieval_text_mode
            + "/db_JSON/"
            + db_name
            + ".json"
        )
    with open(index_path_json, "r") as json_file:
        knowledge_db_as_JSON = json.load(json_file)
    index = faiss.read_index(index_path_faiss)
    # index = index.to_gpu(gpu_id=0)
    return index, knowledge_db_as_JSON

def medcpt_FAISS_pubmed_retrieval(
    questions,
):
    k = 20
    index_path = "/home/service/BioLlama/vectorstores/PubMed/knn.index"
    pma = faiss.read_index(index_path)
    pma.metric_type = faiss.METRIC_INNER_PRODUCT

    query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    # rerank_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    # rerank_model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")

    for question in questions:
        with torch.no_grad(): # This code is taken directly from the MedCPT GitHub/HF tutorial
                # tokenize the queries
            encoded = query_tokenizer(
                question,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            # encode the queries (use the [CLS] last hidden states as the representations)
            # encoded.to("cuda:0")
            embeds = query_model(**encoded).last_hidden_state[:, 0, :]
            
        distances, indices = pma.search(embeds, k)
        distances = distances.flatten()
        indices = indices.flatten()
        
    print(indices)
    print(distances)
    #12375147

    return


def medcpt_FAISS_retrieval(
    questions,
    db_name,
    retrieval_text_mode,
    chunk_length=None,
    verbose=False,
    with_indices=False,
    query_tokenizer=None,
    query_model=None,
    rerank_tokenizer=None,
    rerank_model=None,
    top_k=1,
    k=5,
    db_faiss=None,
    db_json=None,
):
    time_start = time.time()
    if (db_faiss == None) or (db_json == None):
        db_faiss, db_json = load_db(
            "medcpt", db_name, retrieval_text_mode, chunk_length=chunk_length
        )
    time_after_loading_db = time.time()
    if (query_tokenizer == None) or (query_model == None):
        query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        print("had to load query model?")

    if (rerank_tokenizer == None) or (rerank_model == None):
        rerank_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        rerank_model = AutoModelForSequenceClassification.from_pretrained(
            "ncbi/MedCPT-Cross-Encoder"
        )
        print("had to load rerank model?")

    # i will first try embedding of question and retrieval on a question by question basis, and time it
    # this is with arbitrary choices: k=1, max_length (how many tokens are in input i think?) = 512
    chunk_list = []
    retrieval_quality = []
    disable = True
    # if "questions" is just a string, we make it a list so iteration is not character-wise
    if type(questions) == str:
        questions = [questions]
        disable = True
    time_after_loading_models = time.time()
    if verbose == True:
        disable = False

    for question in tqdm(questions, desc="Retrieving chunks", disable=disable):
        chunks = []
        with torch.no_grad(): # This code is taken directly from the MedCPT GitHub/HF tutorial
            # tokenize the queries
            encoded = query_tokenizer(
                question,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            # encode the queries (use the [CLS] last hidden states as the representations)
            # encoded.to("cuda:0")
            embeds = query_model(**encoded).last_hidden_state[:, 0, :]
        
        distances, indices = db_faiss.search(embeds, k)
        distances = distances.flatten()
        indices = indices.flatten()

        if k > 1:
            # reranking step
            # if local_transformers:
            #     from ..finetuning.cti.transformers.transformers.src.transformers.models.auto import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
            chunks = [db_json[str(indices[i])] for i in range(len(distances))]
            pairs = [[question, chunk] for chunk in chunks]
            with torch.no_grad():
                encoded = rerank_tokenizer(
                    pairs,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                # encode the queries (use the [CLS] last hidden states as the representations)
                logits = rerank_model(**encoded).logits.squeeze(dim=1).numpy()

            # "logits" now gives us relevance scores. we want to use to resort the chunks array by the relevance scores in logits, where higher relevance should be first
            # we can do this by sorting the indices of logits, and then using those indices to sort chunks
            sorted_scores = sorted(zip(chunks, logits), key=lambda x: x[1], reverse=True)
            sorted_chunkid_indices = sorted(zip(indices, logits), key=lambda x: x[1], reverse=True)
            sorted_indices = np.array([x[1] for x in sorted_scores])

            # can use the following to see "after which item do logits turn negative"
            if np.where(sorted_indices > 0)[0].size > 0:
                last_positive = np.where(sorted_indices > 0)[0][-1]
            else:
                last_positive = 0
                pass
            new_chunks = [x[0] for x in sorted_scores]
            top_chunk = new_chunks[0:top_k]
            top_index = [x[0] for x in sorted_chunkid_indices[0:top_k]]
            retrieval_quality.append(
                sorted_indices[0]
            )  # note that this is the score of the top chunk, not the average of the top k we return

        else:
            chunks = [db_json[str(indices[i])] for i in range(len(distances))]
            top_chunk = chunks[0]

        if with_indices:
            chunk_list.append([top_chunk, top_index])
        else:
            chunk_list.append(top_chunk)
    time_after_retrieval = time.time()
    retrieval_quality = np.array(retrieval_quality)
    avg_retrieval_quality = np.mean(retrieval_quality)
    # print(f"Avg retrieval quality: {str(avg_retrieval_quality)}")
    # print(f"Time to load db: {str(time_after_loading_db - time_start)}, then to load models: {str(time_after_loading_models - time_after_loading_db)}, to then retrieve chunks: {str(time_after_retrieval - time_after_loading_models)} seconds.")
    return chunk_list


def gte_FAISS_retrieval(questions, db_name, retrieval_text_mode):
    db_faiss, db_json = load_db("gte-large", db_name, retrieval_text_mode)
    # print("db: " + str(db_faiss))
    time_before_retrieval = time.time()
    k = 1
    chunk_list = []
    embedding_model = SentenceTransformer("thenlper/gte-large")
    for question in tqdm(questions, desc="Retrieving chunks"):
        chunks = []
        question_embedding = np.array([embedding_model.encode(question)])
        distances, indices = db_faiss.search(question_embedding, k)
        distances = distances.flatten()
        indices = indices.flatten()
        # print("Distances shape: " + str(distances.shape))
        for i in range(len(distances)):
            # print("Distance: " + str(distances[i]) + " Index: " + str(indices[i]))
            # print("Chunk: " + str(db_json[str(indices[i])]))
            chunks.append(db_json[str(indices[i])])
        chunk_list.append(chunks)
    time_after_retrieval = time.time()
    # print("Time to retrieve chunks: " + str(time_after_retrieval - time_before_retrieval) + " seconds.")
    return chunk_list


# finds the local abstract id of a chunk in the input_segmentation JSON files with sizes 16 and 32
def chunk_to_laid(chunk_id, chunk_length):
    file_path = (
        "vectorstores/RCT200ktrain/medcpt/input_segmentation/db_JSON_"
        + str(chunk_length)
        + "/RCT200ktrainlookupchart.json"
    )
    with open(file_path, "r") as f:
        lookupchart = json.load(f)
    return lookupchart[chunk_id]


def given_chunkids_find_sections(chunks, chunk_ids, chunk_length):
    import difflib

    documents, documents_dict = read_documents(
        "RCT200ktrain", "input_segmentation", return_full_dict=True
    )
    full_section_scores = {
        "BACKGROUND": 0,
        "OBJECTIVE": 0,
        "METHODS": 0,
        "RESULTS": 0,
        "CONCLUSIONS": 0,
    }
    for i in tqdm(range(len(chunk_ids)), desc="Assigning chunks to sections"):
        if (i > len(chunks) - 1) or (i > len(chunk_ids) - 1):
            print(
                f"i is {i} and this exceeds length? {len(chunks)} and {len(chunk_ids)}"
            )
            continue
        else:
            chunk = chunks[i]
            chunk_id = chunk_ids[i]
            local_abstract_id = chunk_to_laid(str(chunk_id), chunk_length)
            abstract = documents[local_abstract_id]
            abstract_as_dict = documents_dict[local_abstract_id]

            # now we run string comparison to see which section this chunk mostly belongs to
            # the following code was written by Bard
            section_scores = {}
            for section, text in abstract_as_dict.items():
                if section != "ID":
                    text = text.lower()
                    similarity = difflib.SequenceMatcher(None, chunk, text).ratio()
                    section_scores[section] = similarity
            best_match_section = max(section_scores, key=section_scores.get)
            best_match_score = section_scores[best_match_section]
            full_section_scores[best_match_section] += 1
    return full_section_scores


def section_distribution_stats(questions, chunk_length):
    retrieved_chunks = medcpt_FAISS_retrieval(
        questions=questions,
        db_name="RCT200ktrain",
        retrieval_text_mode="input_segmentation",
        chunk_length=chunk_length,
        with_indices=True,
    )
    chunks = []
    chunk_ids = []
    for i in range(len(retrieved_chunks)):
        chunk = retrieved_chunks[i][0]
        print(chunk)
        chunks.append(chunk)
        chunk_id = retrieved_chunks[i][1]
        # print(chunk_id)
        chunk_ids.append(chunk_id)
    section_scores = given_chunkids_find_sections(
        chunks=chunks, chunk_ids=chunk_ids, chunk_length=chunk_length
    )
    return section_scores


if __name__ == "__main__":
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_name", type=str, help="Name of the database to build index for."
    )
    parser.add_argument("--embedding_type", type=str, help="Type of embedding to use.")
    parser.add_argument(
        "--mode",
        type=str,
        help="Whether to embed full text or combo of background, objective, methods, results, conclusions.",
    )
    parser.add_argument("--chunk_length", type=int, help="Length of chunks to embed.")
    args = parser.parse_args()
    if args.embedding_type == "gte-large":
        build_index_gte(args.db_name, args.mode)
    elif args.embedding_type == "medcpt":
        build_index_medcpt(args.db_name, args.mode, args.chunk_length)
        # read_chunks("RCT200ktrain.txt", "medcpt", "sixteen")
    elif args.embedding_type == "debug":
        questions = ["Which is the main calcium pump of the sarcoplasmic reticulum?"]
        medcpt_FAISS_retrieval(
            questions=questions,
            db_name="RCT200ktrain",
            retrieval_text_mode="input_segmentation",
            chunk_length=16,
        )

# questions = ["The purpose of this randomized, double-blind parallel group study was to compare the safety, tolerability and acceptability of Easyhaler and Turbuhaler dry powder inhalers for the delivery of budesonide 800 microg day(-1) in adult asthmatic patients who had already been treated with inhaled corticosteroids for at least 6 months prior to the study Additionally the efficacy of the products was evaluated. The main objective was to evaluate the systemic safety of budesonide inhaled from Easyhaler (Giona Easyhaler, Orion Pharma, Finland) as determined by serum and urine cortisol measurements. The secondary objective was to compare the tolerability acceptability and efficacy of the two devices in the administration of budesonide."]
# medcpt_FAISS_pubmed_retrieval(questions=questions)

# neighbours = medcpt_FAISS_retrieval(questions=questions, db_name="RCT200ktrain", retrieval_text_mode="input_segmentation", chunk_length=32, verbose=True)
# print(neighbours)