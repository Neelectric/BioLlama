from autofaiss import build_index
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import faiss
import torch
import json

build_index(embeddings="vectorstores/pma_source", index_path="vectorstores/pma_target/pma_knn.index",
            index_infos_path="vectorstores/pma_target/pma_target_index_infos.json",
            current_memory_available="250G")