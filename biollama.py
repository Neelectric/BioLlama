# Part of the BioLlama library
# Written by Neel Rajani
# Main file to serve as an abstract entry point into BioLlama inference

def inference_biollama(model="Llama-2-70B-chat-GPTQ", 
              benchmark="MedMCQA", 
              b_start = 0, 
              b_end = 1, 
              max_new_tokens = 30,):

    #parse the given benchmark
    questions = ["Which is the main calcium pump of the sarcoplasmic reticulum?"]
    # marking_scheme = "Sarcoplasmic reticulum Ca(2+)-ATPase"

    prompts = [promptify(question) for question in questions]

    retrieved_chunks = []

    BioLlama = biollama()

    for prompt in prompts:
        biollama(prompts, retrieved_chunks)