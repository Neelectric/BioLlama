#hugely aided by https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb

import torch
from cti.transformers.transformers.src.transformers.models.auto import AutoTokenizer, AutoModelForCausalLM

# class RETROWrapper(torch.nn.Module):

class BioLlama:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13b-Chat-GPTQ")
        self.model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13b-Chat-GPTQ",
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
    
BioLlama = BioLlama()
prompt = "Tell me a good night story"
text = BioLlama.generate_text(prompt)
print(text)