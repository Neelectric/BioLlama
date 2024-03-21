#04.11.23 - Neel Rajani
#The purpose of this file is to provide a full pipeline for BioLlama
#It is called with a number of parameters, as outlined below
# -m : the model to use. Options are Llama2 (in 7B, 13B or 70B)
# -b : the benchmark to use. Options are BioASQ5b_factoid_with_snippet, MedQA-5, PubMedQA, MedMCQA
# -e : the evaluation method to use. Options are exact_match and judging

from utilities.inference import inference
from utilities.exact_match import exact_match
from utilities.judging import llm_as_judge
from utilities.utilities import write_to_readme
import torch

model =  "Llama-2-13B-chat-GPTQ" # eg. "Llama-2-7B-chat-GPTQ", "Llama-2-7B-chat-finetune", "BioLlama-7B", "BioLlama-7B-finetune"
two_epochs = False
torch_dtype = None
zero_shot = True
if model[:11] == "BioLlama-7B": torch_dtype = torch.float32 # eg. torch.float32, torch.bfloat16 or "int4"
elif model[:12] == "BioLlama-13B": torch_dtype = torch.bfloat16 # eg. torch.float32, torch.bfloat16 or "int4"
elif model[:12] == "BioLlama-70B": torch_dtype = "int4" # eg. torch.float32, torch.bfloat16 or "int4"
benchmark = "MedQA-5" # eg. "MedQA-5", "PubMedQA", "MedMCQA", "bioASQ_no_snippet", "bioASQ_with_snippet"
db_name = "RCT200ktrain"
retrieval_model = None # eg. "gte-large", "medcpt"
retrieval_text_mode = None # eg. "full", "input_segmentation
chunk_length = None
top_k = 1
b_start = 10
num_questions = 20
b_end = b_start + num_questions

#if benchmark name starts with "bioASQ" set max_new_tokens to 40
max_new_tokens = 30
if benchmark[:6] == "bioASQ":
    max_new_tokens = 45  
    b_start = 0

if benchmark == "PubMedQA":
    max_new_tokens = 15
    b_start = 0
    b_end = b_start + num_questions
if benchmark == "MedQA-4" or benchmark == "MedQA-5":
    max_new_tokens = 30

if zero_shot:
    max_new_tokens = 35

inference(model=model,
        benchmark=benchmark,
        b_start=b_start,
        b_end=b_end,
        max_new_tokens=max_new_tokens,
        inference_mode="std",
        retrieval_model=retrieval_model,
        retrieval_text_mode=retrieval_text_mode,
        chunk_length=chunk_length,
        top_k=top_k,
        db_name=db_name,
        torch_dtype=torch_dtype,
        zero_shot=zero_shot,)

if torch_dtype is not None:
    print(f"Used dtype {torch_dtype}")

if benchmark == "MedQA-4" or benchmark == "MedQA-5" or benchmark == "PubMedQA" or benchmark == "MedMCQA":
    accuracy = 100*exact_match(model=model, benchmark=benchmark, zero_shot=zero_shot)

elif benchmark == "bioASQ_no_snippet" or benchmark == "bioASQ_with_snippet":
    # note that this runs out of memory for anything larger than 7B models (I think)
    # it should be easy to remedy, but I don't have the time to do it
    accuracy = 100*llm_as_judge(model_to_mark=model, benchmark_to_mark=benchmark) 
    
if retrieval_model == "gte-large":
    model = "GTE"
elif retrieval_model == "medcpt":
    model = "MedCPT"
elif retrieval_model == "retro":
    model = "BioLlama"
if two_epochs:
    model = model + "-2"
if zero_shot:
    model = model + "-0"
if num_questions > 100:
    write_to_readme(model, benchmark, result=accuracy, db_name=db_name, retrieval_text_mode=retrieval_text_mode, top_k=top_k, num_questions=num_questions)
