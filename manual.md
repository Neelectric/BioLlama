# User manual 

NOTE: This repository is entirely experimental and unlikely to run on another machine without many hours of laborious setup and debugging of my confusing code. Proceed at your own risk!

This directory contains the files of a complex and wide-reaching project. To replicate my approach and results, a workstation with 48Gb GPU VRAM is probably necessary, in addition to sufficient free storage space (~300Gb) and installation of the relevant CUDA/pytorch/transformers versions. requirements.txt was automatically produced by the wandb library and hopefully includes all necessary dependencies. Experiments were run with 
* nvidia-smi version 525.147.05
* driver version 525.147.05
* CUDA version 12.0
* torch version 2.1.2
on 2x NVIDIA GeForce RTX4090s.

The main driver file is main.py (main_biollama.py served as an entry-point for testing BioLlama's behaviour during development, but was not used for actual benchmarking). Inside of main.py, initial variables can be set, such as


model =  "Llama-2-13B-chat-GPTQ" # eg. "Llama-2-7B-chat-GPTQ", "Llama-2-7B-chat-finetune", "BioLlama-7B", "BioLlama-7B-finetune"
two_epochs = False # eg. True, False
torch_dtype = None # should be false but could theoretically be eg. torch.float32, torch.bfloat16 or "int4"
zero_shot = True # eg. True, False
benchmark = "MedQA-5" # eg. "MedQA-4", "MedQA-5", "PubMedQA", "MedMCQA", "bioASQ_no_snippet", "bioASQ_with_snippet"
db_name = "RCT200ktrain" # eg. "RCT200ktrain", in addition "RCT20ktrain" could work but support would need to be added manually
retrieval_model = None # eg. "gte-large", "medcpt", this variable was relevant for retrieval testing, should be None 
retrieval_text_mode = None # eg. "full", "input_segmentation",this variable was relevant for retrieval testing, should be None
chunk_length = None # this variable was relevant for retrieval testing, should be None
top_k = 1 # how many chunks to retrieve, this variable was relevant for retrieval testing, should be 1
b_start = 10 # from which point in the benchmark shall we start sampling questions, 10 recommended as questions 0-10 is where few-shot examples are from
num_questions = 1000 # how many questions to sample, 1000 is standard

These variables dictate how an experiment is run. A call to inference.py is then made, which takes care of running a benchmark as specified. There are essentially two model types: Llama-2, and BioLlama. Both have different backends. inference.py makes a call to llm.py, where the correct backend is identified and handled. If the model type is Llama-2, the exllama library with its optimized CUDA kernels handles the creation of a model object and the model inference itself. The relevant files are in /src/. If the model type is BioLlama, my custom BioLlama class in biollama.py handles the creation of a model object and model inference in a tight interplay with HuggingFace's Transformers library. After the respective backend returns the model outputs, these are saved by inference.py to a JSON file in /output/, named after the model and benchmark config (optionally also if it was zero-shot and if it was a finetuned model). main.py then uses exact_match.py or llm-as-a-judge.py to mark these outputs, and writes to the README if it was a properly sized training run (if less than 100 questions were involved in benchmarking, this is treated as a sanity test/debugging run and not recorded).

Note that this code was designed without the intention of thorough reuse by other people. If there are persistent issues, feel free to contact me at Neel.R@web.de, but I cannot make guarantees that any of this will work elsewhere/will not brick your computer.