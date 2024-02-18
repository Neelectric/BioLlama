#04.11.23 - Neel Rajani

#The purpose of this file is to provide a full pipeline for BioLlama
#It is called with a number of parameters, as outlined below
# -m : the model to use. Options are Llama2 (in 7B, 13B or 70B)
# -b : the benchmark to use. Options are BioASQ5b_factoid_with_snippet, MedQA, PubMedQA, MedMCQA
# -e : the evaluation method to use. Options are exact_match and judging

from utilities.inference import inference
from utilities.exact_match import exact_match
from utilities.utilities import write_to_readme
import torch

model =  "BioLlama-7B" # eg. "Llama-2-7B-chat-GPTQ", "Llama-2-13B-chat-GPTQ", "Llama-2-70B-chat-GPTQ", "Llama-2-7B-chat-finetune"
# model = "Llama-2-7B-chat-GPTQ"
torch_dtype = None
if model[:8] == "BioLlama":
    torch_dtype = torch.float32
benchmark = "MedMCQA" # eg. "MedQA", "PubMedQA", "MedMCQA"
db_name = "RCT200ktrain"
retrieval_model = None # eg. "gte-large", "medcpt"
retrieval_text_mode = None # eg. "full", "input_segmentation
chunk_length = None
top_k = 1
b_start = 10
num_questions = 1000
b_end = b_start + num_questions


inference(model=model,
        benchmark=benchmark,
        b_start=b_start,
        b_end=b_end,
        max_new_tokens=30,
        inference_mode="std",
        retrieval_model=retrieval_model,
        retrieval_text_mode=retrieval_text_mode,
        chunk_length=chunk_length,
        top_k=top_k,
        db_name=db_name,
        torch_dtype=torch_dtype)

if torch_dtype is not None:
    print(f"Used dtype {torch_dtype}")

if benchmark == "MedQA" or benchmark == "PubMedQA" or benchmark == "MedMCQA":
    accuracy = 100*exact_match(model=model, benchmark=benchmark)
    #convert accuracy to be up to 2 significant figures
    
if retrieval_model == "gte-large":
    model = "GTE"
elif retrieval_model == "medcpt":
    model = "MedCPT"
elif retrieval_model == "retro":
    model = "BioLlama"
if num_questions > 100:
    write_to_readme(model, benchmark, result=accuracy, db_name=db_name, retrieval_text_mode=retrieval_text_mode, top_k=top_k, num_questions=num_questions)