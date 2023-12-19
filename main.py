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
benchmark = "MedMCQA"
retrieval_mode = "faiss"

inference(inference_mode="std",
        retrieval_mode=retrieval_mode,
        model=model,
        benchmark=benchmark,
        b_start=10,
        b_end=20,
        max_new_tokens=30)

accuracy = 100*exact_match(model=model, benchmark=benchmark)
if retrieval_mode == "faiss":
    model = "RAGLlama"
elif retrieval_mode == "retro":
    model = "BioLlama"
write_to_readme(model=model, benchmark=benchmark, result=accuracy)