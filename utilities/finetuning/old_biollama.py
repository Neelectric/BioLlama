# # Part of the BioLlama library
# # Written by Neel Rajani
# # Main file to serve as an abstract entry point into BioLlama inference
# # Hugely aided by https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb

# import torch
# from cti.transformers.transformers.src.transformers.models.auto import AutoTokenizer, AutoModelForCausalLM
# import time
# # class RETROWrapper(torch.nn.Module):

# class BioLlama:
#     def __init__(self, model_id, chunk_length):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForCausalLM.from_pretrained(model_id,
#                                                           device_map="auto")

#         #here i change this part, so that im adding normal layers as normal
#         #and retro layers differently
#         # for i, layer in enumerate(self.model.model.layers):
#         #     self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm)

#         # RETRO_layer_ids = [15]
#         # for i, layer in enumerate(self.model.model.layers):
#         #     #switch decoder layer to be a RETRO layer
#         #     if i in RETRO_layer_ids:
#         #         self.model.model.layers[i] = RETROWrapper(layer, self.model.lm_head, self.model.model.norm)

#     def inference(self, prompt, max_length=100):
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
#         return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# questions = ["Which is the main calcium pump of the sarcoplasmic reticulum?"]
# # neighbours = 
# # marking_scheme = "Sarcoplasmic reticulum Ca(2+)-ATPase"
# prompt = questions[0]
# model_id = "TheBloke/Llama-2-7b-chat-GPTQ"
# chunk_length = 16

# time_before_setup = time.time()
# BioLlama = BioLlama(model_id=model_id, chunk_length=chunk_length)
# time_before_generation = time.time()
# text = BioLlama.inference(prompt)
# time_after = time.time()
# print(text)
# print(f"Time taken for setup: {time_before_generation - time_before_setup}")
# print(f"Time taken for generation: {time_after - time_before_generation}")
# print(f"Time total: {time_after - time_before_setup}")