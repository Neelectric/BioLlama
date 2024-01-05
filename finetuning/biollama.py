# Part of the BioLlama library
# Written by Neel Rajani
# Main file to serve as an abstract entry point into BioLlama inference
# Hugely aided by https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb

import torch
from cti.transformers.transformers.src.transformers.models.auto import AutoTokenizer, AutoModelForCausalLM
import time
# class RETROWrapper(torch.nn.Module):

class BioLlama:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-70b-Chat-GPTQ")
        self.model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-70b-Chat-GPTQ",
                                                          device_map="auto")

        #here i change this part, so that im adding normal layers as normal
        #and retro layers differently
        # for i, layer in enumerate(self.model.model.layers):
        #     self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm)

        # RETRO_layer_ids = [15]
        # for i, layer in enumerate(self.model.model.layers):
        #     #switch decoder layer to be a RETRO layer
        #     if i in RETRO_layer_ids:
        #         self.model.model.layers[i] = RETROWrapper(layer, self.model.lm_head, self.model.model.norm)

    def generate_text(self, prompt, max_length=30):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

prompt = "Tell me a good night story" 
time_before_setup = time.time()
BioLlama = BioLlama()
time_before_generation = time.time()
text = BioLlama.generate_text(prompt)
time_after = time.time()
print(text)
print(f"Time taken for setup: {time_before_generation - time_before_setup}")
print(f"Time taken for generation: {time_after - time_before_generation}")
print(f"Time total: {time_after - time_before_setup}")