import box
import yaml
import glob
import argparse
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def build_index(db_name):
    chunk = ""
    chunks = []
    n_lines = 0

    txt_files = glob.glob(cfg.DATABASE_PATH + "/*.txt")
    print(txt_files)
    if db_name in txt_files:
        file_name = db_name
    else:
        file_name = txt_files[0]
    # if len(txt_files) != 1:
    #     raise Exception("There should be exactly one .txt file in the data directory.")
    
    with open(file_name, 'r') as file:
        for line in tqdm(file, desc="Building index",):
            n_lines += 1
            #print(n_lines)
            if not line.strip():
                chunks.append(chunk)
            else:
                chunk += line
    print(len(chunks))
    embedding_model = SentenceTransformer("thenlper/gte-large")
    chunk_embeddings = embedding_model.encode(chunks)

    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    faiss.write_index(index, cfg.DATABASE_AS_FAISS_PATH + db_name + '.index')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name', type=str, help="Name of the database to build index for.")
    args = parser.parse_args()
    build_index(args.db_name)
