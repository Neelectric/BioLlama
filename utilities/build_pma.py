# Part of the BioLlama library
# Written by Neel Rajani
# Builds a FAISS index of pma using autofaiss


import json
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
from autofaiss import build_index

build_index(embeddings="PubMed", index_path="vectorstores/PubMed/knn.index",
            index_infos_path="vectorstores/PubMed/index_infos.json",
            current_memory_available="250G")