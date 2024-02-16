# Part of the BioLlama library
# Written by Neel Rajani
# Primary method for creation of callable "llm" object, adapted from exllama's "test_benchmark_inference.py"

from transformers import AutoModelForCausalLM, AutoTokenizer
import os, glob
import json
import argparse
import torch

from src.model import ExLlama, ExLlamaCache, ExLlamaConfig
from src.tokenizer import ExLlamaTokenizer
from src.generator import ExLlamaGenerator
from src.alt_generator import ExLlamaAltGenerator
import src.model_init as model_init
from utilities.biollama import BioLlama

#function that creates a callable "llm" object
def llm(model_directory, prompts, max_new_tokens, generator_mode, model_object, torch_dtype):
    if model_directory == "/home/service/BioLlama/utilities/finetuning/llama2_training_output/":
        output, model_object = finetuned_llama2(model_directory, prompts, max_new_tokens, model_object)
        return output, model_object
    elif model_directory[:10] == "meta-llama":
        output, model_object = finetuned_biollama(model_directory, prompts, max_new_tokens, model_object, torch_dtype)
        return output, model_object
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
        generator.settings.temperature = 0.01
        generator.settings.top_p = 0.65
        generator.settings.top_k = 100
        generator.settings.typical = 0.5
        
        # Generate, batched
        output = generator.generate_simple(prompts, max_new_tokens = max_new_tokens)

    elif generator_mode == "alt":
        cache = ExLlamaCache(model, batch_size = 1)  # create cache for inference
        generator = ExLlamaAltGenerator(model, tokenizer, cache)
        generator.settings.token_repetition_penalty_max = 1.2
        generator.settings.temperature = 0.01
        generator.settings.top_p = 0.65
        generator.settings.top_k = 100
        generator.settings.typical = 0.5

        #generator.settings.stop_strings = ["</ANSWER>"]
        stop_conditions = ["</ANSWER>"]
        output = generator.generate(prompts, max_new_tokens=max_new_tokens, gen_settings=generator.settings, stop_conditions=stop_conditions, encode_special_characters=False)
    return output, model_object

def finetuned_llama2(model_directory, prompts, max_new_tokens, model_object = None):
    if model_object is None:
        new_model = AutoModelForCausalLM.from_pretrained(model_directory, device_map="auto")
        new_model.new_tokenizer = AutoTokenizer.from_pretrained(model_directory)
    else:
        new_tokenizer = model_object.new_tokenizer
        new_model = model_object
    #set model temperature to 0.01
    new_model.config.temperature = 0.01
    generations = []
    for prompt in prompts:
        input_ids = new_model.new_tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to('cuda') # otherwise we get userwarning for input ids on CPU while model on GPU
        generated = new_model.generate(input_ids, max_new_tokens=35, do_sample=True, top_p=0.95, top_k=60)
        decoded_generated = new_model.new_tokenizer.decode(generated[0], skip_special_tokens=True)
        generations.append(decoded_generated)
    return generations, new_model

def finetuned_biollama(model_directory, prompts, max_new_tokens, model_object = None, torch_dtype = None):
    # override_directory = 'meta-llama/Llama-2-13b-chat-hf'
    # print(f"overriding model_id with {override_directory}, using torch_dtype {torch_dtype}")
    if model_object is None:
        chunk_length = 32
        new_model = BioLlama(model_id=model_directory, 
                             chunk_length=chunk_length, 
                             RETRO_layer_ids = [15], 
                             training=False, 
                             torch_dtype=torch_dtype)
    else:
        new_model = model_object
    #set model temperature to 0.01
    generations = []
    for prompt in prompts:
        # print(prompt)
        num_tokens, text = new_model.generate(prompt=prompt, max_new_tokens=20)
        generations.append(text)
    return generations, new_model