# the following is a high level overview of CCA
# soooo input comes in, we get this long sequence of tokens right

# chunk length = 32, so we'll split it into bits of 32 tokens... which should be easy given that every input_id refers to 1 token

# so say we have a prompt of length 96
# so sequence length (n) = 96, number of chunks (l) = 3, size (chunk_size) = 32
# we get 
# H1 = prompt[0:32]
# H2 = prompt[32:64]
# H3 = prompt[64:96]

# most of H1 just gets ignored lol

# we then perform retrieval: 
# for BioReader, training k performance is found to peak at 9, and eval k performance is found to be the same around 2<k<15
# for RETRO,     training k performance is found to peak at 2 (compared to 1 or 4), and eval k performance is found to be best at 2?

# for the first l-1 (2) chunks, find the 2 nearest neighbours and their continuations
# so for chunk
# H1, we find neighbours E1,1 and E1,2 with respective continuations and store in E1 = [E1,1 and E1,2]
# H2, we find neighbours E2,1 and E2,2 with respective continuations and store in E2 = [E2,1 and E2,2]

# then we prepare for cross attention, ie create l-1 (2) spliced chunks
# H1+ = prompt[31:63]
# H2+ = prompt[63:95] 

# then cross attention is performed:
# H1+_CA = CA(H1+, E1)
# H2+_CA = CA(H1+, E2)

# finally, we concatenate everything together:
# output = H1[0:-2] concatenated with H1+_CA concatenated with H2+_CA concatenated with H3[1:]
# this hopefully gives the output of size we desire....


print("CCA.py has been run")

import json
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
embeds = np.load("PubMed/embeds_chunk_18.npy")
print(embeds.shape)

# pmids = json.load(open("PubMed/pmids_chunk_18.json"))
# print(len(pmids))
# 940707
# pmid2content = json.load(open("PubMed/pubmed_chunk_18.json"))
# print(len(pmid2content))
# 940707
# print(pmids[:10])
# ['18000000', '18000001', '18000002', '18000003', '18000004', '18000005', '18000006', '18000007', '18000008', '18000009']
# print(pmid2content["18000000"])
# {'d': '20071115', 't': 'NONCODE v2.0: decoding the non-coding.', 'a': 'The NONCODE database is an integrated knowledge database designed for the analysis of non-coding RNAs (ncRNAs). Since NONCODE was first released 3 years ago, the number of known ncRNAs has grown rapidly, and there is growing recognition that ncRNAs play important regulatory roles in most organisms. In the updated version of NONCODE (NONCODE v2.0), the number of collected ncRNAs has reached 206 226, including a wide range of microRNAs, Piwi-interacting RNAs and mRNA-like ncRNAs. The improvements brought to the database include not only new and updated ncRNA data sets, but also an incorporation of BLAST alignment search service and access through our custom UCSC Genome Browser. NONCODE can be found under http://www.noncode.org or http://noncode.bioinfo.org.cn.', 'm': 'animals!|databases, nucleic acid!|databases, nucleic acid*|humans!|internet!|rna, untranslated!|rna, untranslated!chemistry|rna, untranslated!chemistry*|rna, untranslated!classification|rna, untranslated!genetics|user-computer interface!|'}


query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

rerank_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
rerank_model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")

query = "What is the role of non-coding RNA in humans?"
# with torch.no_grad():
# 	# tokenize the queries
# 	encoded = query_tokenizer(
# 		query, 
# 		truncation=True, 
# 		padding=True, 
# 		return_tensors='pt', 
# 		max_length=64,
# 	)
# 	embeds = query_model(**encoded).last_hidden_state[:, 0, :]

from autofaiss import build_index
build_index(embeddings="PubMed", index_path="vectorstores/PubMed/knn.index",
            index_infos_path="vectorstores/PubMed/index_infos.json",
            current_memory_available="250G")