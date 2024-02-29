# Part of the BioLlama library
# Written by Neel Rajani
# Small file to find out how many tokens in PubMed

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

global_num_tokens = 0

year_dict = {}

for i in tqdm(range(38)):
    # print(f"length of file PubMed/pubmed_chunk_" + str(i) + ".json:")
    pmid2content = json.load(open("PubMed/pubmed_chunk_" + str(i) + ".json"))
    num_papers_in_file = len(pmid2content)
    keys = list(pmid2content.keys())

    papers_with_titles = 0
    papers_with_abstracts = 0
    total_num_tokens = 0
    for key in keys:
        paper = pmid2content[key]
        year = paper["d"][0:4]
        if year not in year_dict:
            year_dict[year] = 1
        else:
            year_dict[year] += 1
        if len(paper["t"]) > 0:
            papers_with_titles += 1
        if len(paper["a"]) > 0:
            papers_with_abstracts += 1
            tokenized_abstract = tokenizer(paper["a"], return_tensors="pt")
            num_tokens = tokenized_abstract.input_ids.size()[1]
            total_num_tokens += num_tokens
    json.dump(year_dict, open("vectorstores/pma_years/year_dict" + str(i) + ".json", "w"))


    print(f"num papers in file: {num_papers_in_file}, papers with titles: {papers_with_titles}, papers with abstracts: {papers_with_abstracts}, total num tokens: {total_num_tokens}")
    global_num_tokens += total_num_tokens
print(f"global num tokens: {global_num_tokens}")

#dump year dict at vectorstores/year_dict.json
json.dump(year_dict, open("vectorstores/pma_years/total_year_dict.json", "w"))