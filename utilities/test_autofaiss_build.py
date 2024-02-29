from autofaiss import build_index
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import faiss
import torch
import json

build_index(embeddings="vectorstores/pma_source", index_path="vectorstores/pma_target/pma_target_knn.index",
            index_infos_path="vectorstores/pma_target/pma_target_index_infos.json",
            current_memory_available="250G")

#i should go through and check the tokens per year. then do a cutoff for when it reaches 2 billion
#justify this somehow in background on corpora

#i guess i go through, check if paper date is after cutoff, check if paper has abstract, then add it to json file
#json file is gonna get huge lmao
#that gives me a full json file mapping PMIDs to abstracts, where all are after the cutoff
#then i need to take that and do the input_segmentation for chunk_length=32
#so id go through the json file, open an abstract, tokenize it with the llama-2 tokenizer into chunks of 32
#discard the last chunk if it's less than 32 (probably right)
#then i need to add the chunks to a new json file, which is gonna be huge
#then i need to go through that json file and create embeddings for each chunk, and then i call autofaiss...