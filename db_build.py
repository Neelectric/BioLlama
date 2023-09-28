import box
import yaml
import glob
import faiss
from tqdm import tqdm

with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def build_index():
    chunk = ""
    chunks = []
    n_lines = 0
    txt_files = glob.glob(cfg.DATA_PATH + "/*.txt")
    if len(txt_files) != 1:
        raise Exception("There should be exactly one .txt file in the data directory.")
    
    with open(txt_files[0], 'r') as file:
        for line in tqdm(file, desc="Building index",):
            n_lines += 1
            #print(n_lines)
            if not line.strip():
                chunks.append(chunk)
            else:
                chunk += line
    print(len(chunks))
    

if __name__ == "__main__":
    build_index()
