# Part of the BioLlama library
# Written by Neel Rajani
# Primary method for creation of callable "llm" object, adapted from exllama's "test_benchmark_inference.py"

from src.model import ExLlama, ExLlamaCache, ExLlamaConfig
from src.tokenizer import ExLlamaTokenizer
from src.generator import ExLlamaGenerator
from src.alt_generator import ExLlamaAltGenerator
import os, glob
import json
import argparse
import src.model_init as model_init

#function that creates a callable "llm" object
def llm(model_directory, prompts, max_new_tokens, generator_mode="std"):
    # Locate files we need within that directory
    tokenizer_path = os.path.join(model_directory, "tokenizer.model")
    model_config_path = os.path.join(model_directory, "config.json")
    st_pattern = os.path.join(model_directory, "*.safetensors")
    model_path = glob.glob(st_pattern)[0]

    #this code is taken from "test_benchmark_inference.py" and adapted

    # Parse arguments
    parser = argparse.ArgumentParser(description = "Benchmark tests for ExLlama")
    model_init.add_args(parser)
    args = parser.parse_args()
    
    args.directory = model_directory
    if model_directory == "../models/Llama-2-13B-chat-GPTQ/":
        #args.gpu_split = "10,20"
        args.gpu_split = "4,20"
        #args.length = 1700
        args.gpu_peer_fix = True
    else:
        #args.gpu_split = "24,17"
        #args.gpu_peer_fix = True
        args.gpu_split = "17.2,24"
    model_init.post_parse(args)
    model_init.get_model_files(args)

    # Globals
    model_init.set_globals(args)

    # Instantiate model
    config = model_init.make_config(args)
    config.model_path = model_path                          # supply path to model weights file
    model = ExLlama(config)                                 # create ExLlama instance and load the weights
    tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
    
    # Configure generator
    if generator_mode == "std":
        if type(prompts) == str:
            prompts = [prompts]
        cache = ExLlamaCache(model, batch_size = len(prompts))  # create cache for inference
        generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator
        generator.disallow_tokens([tokenizer.eos_token_id])
        generator.settings.token_repetition_penalty_max = 1.2
        generator.settings.temperature = 0.00
        generator.settings.top_p = 0.65
        generator.settings.top_k = 100
        generator.settings.typical = 0.5
        
        # Generate, batched
        output = generator.generate_simple(prompts, max_new_tokens = max_new_tokens)

    elif generator_mode == "alt":
        cache = ExLlamaCache(model, batch_size = 1)  # create cache for inference
        generator = ExLlamaAltGenerator(model, tokenizer, cache)
        generator.settings.token_repetition_penalty_max = 1.2
        generator.settings.temperature = 0.00
        generator.settings.top_p = 0.65
        generator.settings.top_k = 100
        generator.settings.typical = 0.5

        #generator.settings.stop_strings = ["</ANSWER>"]
        stop_conditions = ["</ANSWER>"]
        output = generator.generate(prompts, max_new_tokens=max_new_tokens, gen_settings=generator.settings, stop_conditions=stop_conditions, encode_special_characters=False)
    return output