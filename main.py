#04.11.23 - Neel Rajani

#The purpose of this file is to provide a full pipeline for BioLlama
#It is called with a number of parameters, as outlined below
# -m : the model to use. Options are Llama2 (in 7B, 13B or 70B)
# -b : the benchmark to use. Options are BioASQ5b_factoid_with_snippet, MedQA, PubMedQA, MedMCQA
# -e : the evaluation method to use. Options are exact_match and judging

from utilities.inference import inference
from utilities.exact_match import exact_match
from utilities.utilities import write_to_readme

model =  "Llama-2-70B-chat-GPTQ"
benchmark = "bioASQ_no_snippet"
db_name = "RCT200ktrain"
retrieval_model = "gte-large"
retrieval_text_mode = "brc"

inference(model=model,
        benchmark=benchmark,
        b_start=13,
        b_end=15,
        max_new_tokens=30,
        inference_mode="std",
        retrieval_model=retrieval_model,
        retrieval_text_mode=retrieval_text_mode,
        db_name=db_name)

accuracy = 100*exact_match(model=model, benchmark=benchmark)
if retrieval_model == "gte-large":
    model = "RAGLlama"
elif retrieval_model == "medcpt":
    model = "RiPLlama"
elif retrieval_model == "retro":
    model = "BioLlama"
#write_to_readme(model=model, benchmark=benchmark, result=accuracy)