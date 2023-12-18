#this code is taken directly from Kenneth Leung's repository
import box
import yaml
import glob
import argparse
import time
import json
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

with open('../config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def build_index(db_name):
    chunk = ""
    chunks = []
    n_chunks = 0
    n_lines = 0
    chunks_dict = {}

    txt_files = glob.glob("../" + cfg.DATABASE_PATH + "/*.txt")
    print(txt_files)
    if db_name in txt_files:
        file_name = db_name
        print("Found the requested file, building FAISS index")
    else:
        file_name = txt_files[0]
    if len(txt_files) != 1:
        raise Exception("There should be exactly one .txt file in the data directory.")
    
    with open(file_name, 'r') as file:
        for line in tqdm(file, desc="Processing chunks",):
            n_lines += 1
            #print(n_lines)
            if not line.strip():
                n_chunks += 1
                chunks.append(chunk)
                chunks_dict[n_chunks] = chunk
                chunk = ""
            else:
                chunk += line
            
    print("len(chunks): " + str(len(chunks)))
    time_before_index = time.time()
    embedding_model = SentenceTransformer("thenlper/gte-large")
    chunk_embeddings = embedding_model.encode(chunks)
    print("Embeddings shape: " + str(chunk_embeddings.shape))
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    print(cfg.DATABASE_AS_FAISS_PATH + db_name + '.index')
    faiss.write_index(index, cfg.DATABASE_AS_FAISS_PATH + db_name + '.index')
    time_after_index = time.time()
    print("Time to build index: " + str(time_after_index - time_before_index) + " seconds.")

    #save to JSON
    json_file_path = cfg.DATABASE_AS_JSON_PATH + db_name + '.json'
    with open(json_file_path, "w") as json_file:
        json.dump(chunks_dict, json_file, indent=4)

def simple_FAISS_retrieval(questions):
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default="train.txt", help="Name of the database to build index for.")
    args = parser.parse_args()
    build_index(args.db_name)
