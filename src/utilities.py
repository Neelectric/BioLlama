import box
import yaml
import glob
import faiss

with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def load_knowledge_db(knowledge_db_name):
    return faiss.read_index(cfg.DATABASE_AS_FAISS_PATH + 'faiss_index.index')

def load_benchmark(benchmark_name):
    return None

