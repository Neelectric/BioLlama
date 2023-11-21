#04.11.23 - Neel Rajani

#The purpose of this file is to provide a full pipeline for BioLlama
#It is called with a number of parameters, as outlined below
# -m : the model to use. Options are Llama2 (in 7B, 13B or 70B)
# -b : the benchmark to use. Options are BioASQ5b_factoid_with_snippet, MedQA, PubMedQA, MedMCQA
# -e : the evaluation method to use. Options are exact_match and judging

from src.llm import llm
from src.prompts import promptify_MedMCQA_question
from inference import inference

# model_directory =  "../models/Llama-2-70B-chat-GPTQ/"

# question = "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma (1) Hyperplasia (2) Hyperophy (3) Atrophy (4) Dyplasia"
# marking_scheme = 3

# promptified_question = promptify_MedMCQA_question(question)
# print(f"promptified question: {promptified_question}")

# output = llm(model_directory, promptified_question, 35, generator_mode="alt")
# print(f"output: {output}")

inference(inference_mode="alt")