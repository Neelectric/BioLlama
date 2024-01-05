# Part of the BioLlama library
# Written by Neel Rajani
# Main file to serve as an abstract entry point into BioLlama inference
# Hugely aided by https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb

import torch
local_transformers = False
if local_transformers:
    from .finetuning.cti.transformers.transformers.src.transformers.models.auto import AutoTokenizer, AutoModelForCausalLM
    from .finetuning.cti.transformers.transformers.src.transformers.models.llama.modeling_llama import LlamaRMSNorm
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaRMSNorm

import time
# from .db_retrieval import medcpt_FAISS_retrieval

class CCA(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()

    def forward(self, *args, **kwargs):
        # we perform chunked cross attention at every decoding step. 
        # we do this with sequences that are 16 tokens long
        # the tokens come in
        output = None
        return output

# We wrap layers, who's ids are designated as RETRO blocks, with this RETROLayer class
class RETROLayer(torch.nn.Module):
    def __init__(self, id, layer, config):
        super().__init__()
        print(f"Wrapping layer {id} with retro")
        self.layer = layer
        self.training = False # apparently by default it thinks we're training
        self.RETRO_id = id # tagging the RETRO layer with its id to identify it later

        #adding the two new components that differentiate BioLlama from vanilla Llama2
        self.pre_CCA_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # this gets initiated with hidden_size
        self.CCA = CCA()
        
    
    def forward(self, *args, **kwargs):
        #intermediate decoding implementation
        # outputs = self.layer(*args, **kwargs)

        #huggingface transformers implementation
        hidden_states = args[0] # usually a tuple where the first element is what we want
        attention_mask = kwargs["attention_mask"] # should be torch.FloatTensor with  `(batch_size, 1,query_sequence_length, key_sequence_length)`
        position_ids = kwargs["position_ids"]
        past_key_value = kwargs["past_key_value"]
        output_attentions = kwargs["output_attentions"]
        use_cache=kwargs["use_cache"]

        residual = hidden_states
        hidden_states = self.layer.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            # **kwargs,
        )
        hidden_states = residual + hidden_states

        # If RETRO, 
        # residual = hidden_states
        # hidden_states = self.pre_CCA_layernorm(hidden_states)
        # hidden_states = self.CCA(hidden_states)
        # hidden_states = residual + hidden_states


        # Fully Connected
        residual = hidden_states
        hidden_states = self.layer.post_attention_layernorm(hidden_states)
        hidden_states = self.layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

class BioLlama:
    def __init__(self, model_id, chunk_length):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

        RETRO_layer_ids = [15]
        for i, layer in enumerate(self.model.model.layers):
            #switch pre-specified decoder layers to be a RETRO layers
            if i in RETRO_layer_ids:
                self.model.model.layers[i] = RETROLayer(id=i, layer=layer, config=self.model.config)

    def inference(self, questions, db_name, retrieval_text_mode):
        #generate neighbours
        # neighbours = medcpt_FAISS_retrieval(questions=questions,db_name=db_name, retrieval_text_mode=retrieval_text_mode)

        #promptify questions with neighbours, few-shot?

        #batch inference

        #write output
        return
        
    def generate(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]