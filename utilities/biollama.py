# Part of the BioLlama library
# Written by Neel Rajani
# Main file to serve as an abstract entry point into BioLlama inference
# Hugely aided by https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb

import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
from ..finetuning.cti.transformers.transformers.src.transformers.models.auto import AutoTokenizer, AutoModelForCausalLM
import time
from .db_retrieval import medcpt_FAISS_retrieval

class RETROWrapper(torch.nn.Module):
    def __init__(self, id, layer):
        super().__init__()
        print(f"Wrapping layer {id} with retro")
        self.layer = layer

class BioLlama:
    def __init__(self, model_id, chunk_length):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id,
                                                          device_map="auto")

        #here i change this part, so that im adding normal layers as normal
        #and retro layers differently
        # for i, layer in enumerate(self.model.model.layers):
        #     self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm)

        RETRO_layer_ids = [15]
        for i, layer in enumerate(self.model.model.layers):
            #switch decoder layer to be a RETRO layer
            if i in RETRO_layer_ids:
                self.model.model.layers[i] = RETROWrapper(id=i, layer=layer)

    def inference(self, questions, db_name, retrieval_text_mode):
        #generate neighbours
        neighbours = medcpt_FAISS_retrieval(questions=questions,db_name=db_name, retrieval_text_mode=retrieval_text_mode)

        #promptify questions with neighbours, few-shot?

        #batch inference

        #write output
        
    def generate(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]