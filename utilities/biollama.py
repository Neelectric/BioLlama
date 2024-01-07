# Part of the BioLlama library
# Written by Neel Rajani
# Main file to serve as an abstract entry point into BioLlama inference
# Hugely aided by https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb

import torch
local_transformers = True
if local_transformers:
    from .finetuning.cti.transformers.transformers.src.transformers.models.auto import AutoTokenizer, AutoModelForCausalLM
    from .finetuning.cti.transformers.transformers.src.transformers.models.llama.modeling_llama import LlamaRMSNorm
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # from transformers import LlamaRMSNorm
from .db_retrieval import medcpt_FAISS_retrieval
    

if local_transformers == False:
    class LlamaRMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            """
            LlamaRMSNorm is equivalent to T5LayerNorm
            """
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

import time
# from .db_retrieval import medcpt_FAISS_retrieval


#randomly initialize Retro encoder and crossattention? according to InstructRetro paper
class CCA(torch.nn.Module):
    def __init__(self, model, layer):
        super().__init__()
        self.training = False # apparently by default it thinks we're training
        self.pre_CCA_layernorm = None
        self.model = model
        self.config = model.model.config
        self.embed_tokens = torch.nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.config.pad_token_id)
        self.layer = layer

    def forward(self, input_ids, attention_mask, position_ids, past_key_value, output_attentions,use_cache):
        # we perform chunked cross attention at every decoding step, with sequences that are 16 tokens long?

        # first we use the llama2 tokenizer to decode the input_ids
        # tokens = self.model.tokenizer.decode(input_ids)
        print(f"input_ids has len {len(input_ids)}")
        tokens = self.model.tokenizer.decode(input_ids)[4:]
        
        # with this unencoded sequence, we then do medCPT FAISS retrieval, returning a chunk
        #retrieved_chunk = medcpt_FAISS_retrieval(tokens, db_name="RCT200ktrain", retrieval_text_mode="input_segmentation", chunk_length=16)
        retrieved_chunk = '[CLS] sarcoplasmic reticulum ( sr ) ca ( 2 + ) - handling proteins play' #hardcoded while transformers bugs me

        # we then use the llama2 tokenizer to encode this chunk
        encoded_chunk = self.model.tokenizer(retrieved_chunk, return_tensors="pt")
        chunk_input_ids = encoded_chunk.input_ids

        print(f"chunk_input_ids has size {chunk_input_ids.size()}")

        # the input sequence/context, which was originally given to model has size 22
        # our chunk has size 27. so we need to prune it to make the residual connection work
        # im pruning the last tokens off...
        print(type(chunk_input_ids))
        unnested_chunk_input_ids = torch.unbind(chunk_input_ids, dim=0)[0]
        sliced_chunk_input_ids = unnested_chunk_input_ids[0:22]
        chunk_input_ids = sliced_chunk_input_ids.reshape((1,22))
        print(f"chunk_input_ids now has size {chunk_input_ids.size()}")
        print("so far everything has worked")
        # then embed them
        inputs_embeds = self.embed_tokens(chunk_input_ids)
        embeds_shape = inputs_embeds.shape
        hidden_states = inputs_embeds

        # then do pre_CCA_layernorm
        hidden_states = self.pre_CCA_layernorm(hidden_states)

        # it complains "expected scalar type Float but found Half"
        # i changed transformers qlinear_cuda_old implementation:
        # before matmul iterations, if weights is torch.float16 and x is torch.float32, then weights gets put to torch.float32
        

        # if use_cache is true, this causes "Attention mask should be of size (1, 1, 22, 44), but is torch.Size([1, 1, 22, 22])"
        # i think that (1,1,22,22) is actually correct: (1,1,22,44) comes from 
        # (bsz, 1, q_len, kv_seq_len) --> kv_seq_len is too large, because self_attn on retrieved chunks messes up kv cache
        # so we set use_cache to false for this attention. but on the whole it stays true
        # note: even when use_cache gets passed as false, the forward pass on the qlinear still processes kv_seq_length
        # so i added an if statement there

        # then self attention
        hidden_states, self_attn_weights, present_key_value = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=False,
            # **kwargs,
        )
        
        return hidden_states

# We wrap layers, who's ids are designated as RETRO blocks, with this RETROLayer class
class RETROLayer(torch.nn.Module):
    def __init__(self, id, layer, config, model):
        super().__init__()
        print(f"Wrapping layer {id} with retro")
        self.layer = layer
        self.training = False # apparently by default it thinks we're training
        self.RETRO_id = id # tagging the RETRO layer with its id to identify it later
        self.model = model
        
        #adding the two new components that differentiate BioLlama from vanilla Llama2
        self.pre_CCA_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # this gets initiated with hidden_size
        self.CCA = CCA(model, layer)
        self.CCA.pre_CCA_layernorm = self.pre_CCA_layernorm
        
    
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
        input_ids = self.model.input_ids

        residual = hidden_states
        hidden_states = self.layer.input_layernorm(hidden_states)

        # print(f"Before Self Attention, hidden_states has shape {hidden_states.shape} and len {len(hidden_states)}")
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

        print(f"Before CCA, hidden_states has shape {hidden_states.shape} and len {len(hidden_states)}")

        # Chunked Cross Attention
        #lets think this through step by step: this comes in at this point with shape [1, 1, 4096]
        # ill let residual be the normal hidden_states, and then 
        # hidden_states is just self.attention on chunks
        # and then i add again

        # preparing the retrieved chunks for attention:
        # I need to call self.tokenizer(prompt, return_tensors="pt")

        residual = hidden_states
        hidden_states = self.CCA.forward(input_ids=input_ids, 
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        past_key_value=past_key_value,
                                        output_attentions=output_attentions,
                                        use_cache=use_cache)
        hidden_states = residual + hidden_states


        # Fully Connected
        residual = hidden_states
        hidden_states = self.layer.post_attention_layernorm(hidden_states)
        hidden_states = self.layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        # print(f"After FFW, hidden_states has shape {hidden_states.shape} and len {len(hidden_states)}")
        outputs = (hidden_states,)
        

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        # potential issue: returned as dtype: torch.float16?
        return outputs

class BioLlama:
    def __init__(self, model_id, chunk_length):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        self.input_ids = None
        RETRO_layer_ids = [15]
        for i, layer in enumerate(self.model.model.layers):
            #switch pre-specified decoder layers to be a RETRO layers
            if i in RETRO_layer_ids:
                self.model.model.layers[i] = RETROLayer(id=i, layer=layer, config=self.model.config, model=self)

        self.model.config.use_cache = False # testing this in hopes of less cca isszes
    def inference(self, questions, db_name, retrieval_text_mode):
        #generate neighbours
        # neighbours = medcpt_FAISS_retrieval(questions=questions,db_name=db_name, retrieval_text_mode=retrieval_text_mode)

        #promptify questions with neighbours, few-shot?

        #batch inference

        #write output
        return
        
    def generate(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        self.input_ids = [int(element) for element in inputs.input_ids[0]]

        encoded = self.tokenizer.encode(prompt)
        print(f"encoded has size {len(encoded)}")
        decoded = self.tokenizer.decode(encoded)
        tokenized = self.tokenizer.tokenize(prompt)
        input_ids = inputs["input_ids"][0]
        print(f"tensor has size {len(input_ids)}")
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]