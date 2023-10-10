from src.model import ExLlama, ExLlamaCache, ExLlamaConfig
from src.tokenizer import ExLlamaTokenizer
from src.generator import ExLlamaGenerator
import os, glob
import json
import argparse

import src.model_init as model_init


#function that creates a callable "llm" object
def llm(prompts, max_new_tokens):

    # Directory containing model, tokenizer, generator
    model_directory =  "../models/Llama-2-70B-chat-GPTQ"

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
    #print("args are as follows: " + str(args))
    args.directory = model_directory
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

    cache = ExLlamaCache(model, batch_size = len(prompts))  # create cache for inference
    generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

    # Configure generator
    generator.disallow_tokens([tokenizer.eos_token_id])
    generator.settings.token_repetition_penalty_max = 1.2
    generator.settings.temperature = 0.01
    generator.settings.top_p = 0.65
    generator.settings.top_k = 100
    generator.settings.typical = 0.5

    # Generate, batched
    output = generator.generate_simple(prompts, max_new_tokens = max_new_tokens)

    # for line in output:
    #     print("---")
    #     print(line)
    return output