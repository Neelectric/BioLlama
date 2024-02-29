# Part of the BioLlama library
# Written by Neel Rajani
# Builds a FAISS index of pma using autofaiss


import json
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
from autofaiss import build_index
from tqdm import tqdm

# build_index(embeddings="PubMed", index_path="vectorstores/PubMed/knn.index",
#             index_infos_path="vectorstores/PubMed/index_infos.json",
#             current_memory_available="250G")

# pmids_18 = json.load(open("PubMed/pmids_chunk_18.json"))
# pmid2content_18 = json.load(open("PubMed/pubmed_chunk_18.json"))
# print(pmids_18[:10])
# print(pmid2content_18["18000000"])
# # 18000000

# pmids_12 = json.load(open("PubMed/pmids_chunk_12.json"))
# pmid2content_12 = json.load(open("PubMed/pubmed_chunk_12.json"))
# print(pmids_12[:10])
# print(pmid2content_12["12375147"])
# # 18000000

# # 12375147

index_id = 0
raw_index_to_pmid = {}
# for i in tqdm(range(38)):

# i = 12
pmid2content_list = []
id2pmid_list = []
for i in tqdm(range(38)):
    pmid2content = json.load(open("PubMed/pubmed_chunk_" + str(i) + ".json"))
    pmid2content_list.append(pmid2content)

    id2pmid = json.load(open("PubMed/pmids_chunk_" + str(i) + ".json"))
    id2pmid_list.append(id2pmid)

    num_papers_in_file = len(pmid2content)
    keys = list(pmid2content.keys())
    for key in keys:
        raw_index_to_pmid[index_id] = key
        index_id += 1

raw_indices = [ 4412338, 24236582,  4206069, 17227622,  2530567]
        
faiss_result1 = 4437672
faiss_result2 = 25292480
str_faiss_result1 = str(faiss_result1)
str_faiss_result2 = str(faiss_result2)
print(f"it looks like raw_index of {faiss_result1} is {raw_index_to_pmid[faiss_result1]}")
print(f"it looks like raw_index of {faiss_result2} is {raw_index_to_pmid[faiss_result2]}")

for i in range(len(pmid2content_list)):
    id2pmid = id2pmid_list[i]
    pmid2content = pmid2content_list[i]
    print(f"first trying to query the pmid from id2pmid{i}")
    try:
        print(id2pmid[faiss_result1])
        print(id2pmid[faiss_result2])
    except:
        pass
    if str_faiss_result1 in id2pmid:
        print(f"{faiss_result1} is in id2pmid{i}")
        print(f"it is at index {id2pmid.index(str_faiss_result1)}")
    if str_faiss_result2 in id2pmid:
        print(f"{faiss_result2} is in id2pmid{i}")
    print(f"now trying to query the content from pmid2content{i}")
    try:
        print(pmid2content[str_faiss_result1])
    except:
        pass
    try:
        print(pmid2content[str_faiss_result2])
    except:
        pass