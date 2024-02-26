# Part of the BioLlama library
# Written by Neel Rajani
# Main file to serve as an abstract entry point into BioLlama inference
# Hugely aided by https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb

import torch
import time
import math
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaRMSNorm, apply_rotary_pos_emb, repeat_kv
from transformers.modeling_utils import load_state_dict
from .db_retrieval import medcpt_FAISS_retrieval, load_db

# New method that allows us to replace the forward pass on the LlamaForCausalLm object
def model_new_forward(self, *args, **kwargs):
    if "input_ids" in kwargs:
        self.input_ids_biollama = kwargs["input_ids"]
        output = self.old_forward(*args, **kwargs)
    elif "labels" in kwargs:
        self.input_ids_biollama = kwargs["labels"]
        kwargs["input_ids"] = kwargs["labels"]
        output = self.old_forward(*args, **kwargs)
    else:
        raise Exception("input_ids or labels not found in kwargs")
    return output

# Cross Chunked Attention (temporary, altered version), currently retired
# def cca_forward(self, input_ids):
#     embed_tokens = self.biollama.model.base_model.embed_tokens
#     input_ids = [int(element) for element in input_ids[0]]
#     chunk_length = self.biollama.chunk_length # pruning input_ids to be the last chunk_length tokens
#     if len(input_ids) > chunk_length: pass #  issue with this: difference in num tokens given by MedCPT query tokenizer vs llama2 tokenizer
#     input_ids = input_ids[-chunk_length:]
#     tokens = self.biollama.tokenizer.decode(input_ids)
#     if tokens[:4] == "<s> ": tokens = tokens[4:] # without the leading "<s> "
#     retrieved_chunk = medcpt_FAISS_retrieval( # example 16: '[CLS] sarcoplasmic reticulum ( sr ) ca ( 2 + ) - handling proteins play'
#         tokens, # example 32: "and stimulation of sarcoplasmic reticulum calcium atpase. we examined the hemodynamic, echocardiographic, and neurohormonal effects of intravenous istaroxime in patients hospitalized with"
#         db_name="RCT200ktrain",
#         retrieval_text_mode="input_segmentation",
#         chunk_length=self.biollama.chunk_length,
#         query_tokenizer=self.biollama.query_tokenizer, # passed as a pre-loaded object to save time
#         query_model=self.biollama.query_model, # passed as a pre-loaded object to save time
#         rerank_tokenizer=self.biollama.rerank_tokenizer, # passed as a pre-loaded object to save time
#         rerank_model=self.biollama.rerank_model, # passed as a pre-loaded object to save time
#         top_k=1,
#         k=5,
#         db_faiss=self.biollama.db_faiss, # passed as a pre-loaded object to save time
#         db_json=self.biollama.db_json, # passed as a pre-loaded object to save time
#     )

#     if type(retrieved_chunk[0]) == list: retrieved_chunk = retrieved_chunk[0]
#     # print(f"tokens is:\n{tokens}")
#     # print(f"retrieved chunk is:\n{retrieved_chunk}")
    
#     encoded_chunk = self.biollama.tokenizer(retrieved_chunk, return_tensors="pt") # we then use the llama2 tokenizer to encode this chunk
#     chunk_input_ids = encoded_chunk.input_ids # get input_ids of tokens of the encoded chunk

#     # Here i prune the last tokens off, otherwise matmul fails
#     unnested_chunk_input_ids = torch.unbind(chunk_input_ids, dim=0)[0] # unnest the chunk_input_ids
#     cutoff = len(input_ids) # this is the number of tokens in the sequence with which we performed retrieval, ie 32
#     sliced_chunk_input_ids = unnested_chunk_input_ids[0:cutoff] # this slices chunk_input_ids to length chunk_length
#     chunk_input_ids = sliced_chunk_input_ids.reshape((1, cutoff))

#     # Next, they are embedded using the model's embed_tokens layer
#     inputs_embeds = embed_tokens(chunk_input_ids)
#     embeds_shape = inputs_embeds.shape
#     hidden_states = inputs_embeds
#     position_ids = torch.arange(embeds_shape[-2], dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

#     # Before passing to cca_attn, we perform pre_cca_layernorm
#     hidden_states = self.pre_cca_layernorm(hidden_states)

#     # Finally, we perform cca_attn, which is currently just self-attetnion on the retrieved chunk
#     hidden_states, self_attn_weights, present_key_value = self.cca_attn(  #when input_ids or hidden_states has shape [1,33] this line explodes
#         hidden_states=hidden_states,
#         position_ids=position_ids,
#         use_cache=False,
#     )
#     return hidden_states

def ca(self, hidden_states, e): # The following combines the HF Transformers LlamaSdpaAttention and RETRO code
    embed_tokens = self.biollama.model.base_model.embed_tokens
    cca_attn = self.cca_attn
    if type(e) == list: 
        e_0 = e[0] 
    else: 
        e_0 = e
    self.biollama.tokenizer.padding_side = 'left'  # this line and the following can be necessary to prevent padding bugs
    self.biollama.tokenizer.pad_token = self.biollama.tokenizer.eos_token
    e_encoded = self.biollama.tokenizer(e_0, return_tensors="pt", max_length=32, padding="max_length") # e can come out as shorter than 32, in which case we pad
    e_input_ids = e_encoded.input_ids
    e_input_ids = e_input_ids[:,0:32] # in case e is longer than 32 tokens, we truncate
    e_encoded_and_embedded = embed_tokens(e_input_ids)
    if e_encoded_and_embedded.device != cca_attn.q_proj.weight.device: # sometimes it complains about tensors not being on same device
        e_encoded_and_embedded = e_encoded_and_embedded.to(cca_attn.q_proj.weight.device)

    
    bsz, q_len, _ = hidden_states.size()
    position_ids = torch.arange(hidden_states.shape[-2], dtype=torch.long, device=hidden_states.device).unsqueeze(0)

    query_states = cca_attn.q_proj(hidden_states)
    key_states = cca_attn.k_proj(e_encoded_and_embedded)
    value_states = cca_attn.v_proj(e_encoded_and_embedded)

    query_states = query_states.view(bsz, q_len, cca_attn.num_heads, cca_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, cca_attn.num_key_value_heads, cca_attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, cca_attn.num_key_value_heads, cca_attn.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    cos, sin = cca_attn.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    key_states = repeat_kv(key_states, cca_attn.num_key_value_groups)
    value_states = repeat_kv(value_states, cca_attn.num_key_value_groups)
    
    attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=cca_attn.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=cca_attn.is_causal and q_len > 1,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = cca_attn.o_proj(attn_output)
    return attn_output


# Cross Chunked Attention (true version, following DeepMind's suggestion as closely as possible)
def cca_forward_true(self, input_ids, hidden_states):
    if (hidden_states.device != self.pre_cca_layernorm.weight.device):
        hidden_states = hidden_states.to(self.pre_cca_layernorm.weight.device)
    hidden_states = self.pre_cca_layernorm(hidden_states)
    input_ids = [int(element) for element in input_ids[0]]
    n = len(input_ids)
    m = self.biollama.chunk_length
    l = math.ceil(n/m)
    H_list = [] # For splitting the input into l chunks of size m (the last one will probably be shorter)
    H_list_decoded = [] # Keeping a decoded version for retrieval lookup
    for i in range(l): 
        if (i+1)*m < n:
            H_temp = input_ids[i*m:(i+1)*m]
            H_temp_decoded = self.biollama.tokenizer.decode(H_temp)
            H_list.append(H_temp)
            H_list_decoded.append(H_temp_decoded)
        else:
            H_temp = input_ids[i*m:]
            H_temp_decoded = self.biollama.tokenizer.decode(H_temp)
            H_list.append(H_temp)
            H_list_decoded.append(H_temp_decoded)
    # print(f"decoded is {H_list_decoded}")

    Hplus_list = [] # every chunk here consists of last token of preceding chunk + chunk itself (minus last token)
    num_spliced_chunks = (n-(m-1)) // m 
    for i in range(m-1, num_spliced_chunks * m, m): # note: this for loop iterates differently than the one above
        Hplus_temp = hidden_states[:,i:i+m:,:]
        Hplus_list.append(Hplus_temp)

    if self.biollama.retrieved_chunk_storage == None: # If this is first decoding step, retrieve neighbours and store them
        E_no_continuations = medcpt_FAISS_retrieval(
            H_list_decoded[0:l-1], # we do not retrieve for the last chunk, following RETRO
            db_name="RCT200ktrain",
            retrieval_text_mode="input_segmentation",
            chunk_length=self.biollama.chunk_length,
            verbose=False,
            query_tokenizer=self.biollama.query_tokenizer, # passed as a pre-loaded object to save time
            query_model=self.biollama.query_model, # passed as a pre-loaded object to save time
            rerank_tokenizer=self.biollama.rerank_tokenizer, # passed as a pre-loaded object to save time
            rerank_model=self.biollama.rerank_model, # passed as a pre-loaded object to save time
            top_k=1, # normally retrieve top 2, following RETRO
            k=1,
            db_faiss=self.biollama.db_faiss, # passed as a pre-loaded object to save time
            db_json=self.biollama.db_json, # passed as a pre-loaded object to save time
        )    
        self.biollama.retrieved_chunk_storage = E_no_continuations
    elif ((n-31) % 32 == 0) or (len(Hplus_list) > len(self.biollama.retrieved_chunk_storage)): # if this is not first decoding step, but we've generated enough to consider a new chunk, retrieve new neighbours and update store
        E_no_continuations = medcpt_FAISS_retrieval(
            H_list_decoded[0:l-1], # we do not retrieve for the last chunk, following RETRO
            db_name="RCT200ktrain",
            retrieval_text_mode="input_segmentation",
            chunk_length=self.biollama.chunk_length,
            verbose=False,
            query_tokenizer=self.biollama.query_tokenizer, # passed as a pre-loaded object to save time
            query_model=self.biollama.query_model, # passed as a pre-loaded object to save time
            rerank_tokenizer=self.biollama.rerank_tokenizer, # passed as a pre-loaded object to save time
            rerank_model=self.biollama.rerank_model, # passed as a pre-loaded object to save time
            top_k=1, # retrieve top 2, following RETRO
            k=1,
            db_faiss=self.biollama.db_faiss, # passed as a pre-loaded object to save time
            db_json=self.biollama.db_json, # passed as a pre-loaded object to save time
        )    
        self.biollama.retrieved_chunk_storage = E_no_continuations
    else: # otherwise, we do not need retrieval (as it would just retrieve the same as we already have stored)
        E_no_continuations = self.biollama.retrieved_chunk_storage

    ca_list = None
    for i in range(len(Hplus_list)): # for these spliced chunks in Hplus_list, calculate cross attentions with neighbours
        # print(f"performing cross-attention of")
        # print(f"chunk:     {H_list_decoded[i+1]}")
        # print(f"neighbour: {E_no_continuations[i]}")
        Hplus_ca = ca(self, Hplus_list[i], E_no_continuations[i])
        if ca_list == None:
            ca_list = Hplus_ca
        else:
            ca_list = torch.cat((ca_list, Hplus_ca), dim=1)
    
    # concatenate together, following RETRO
    last_tokens_offset = (m-1) + num_spliced_chunks*m
    prefix_and_ca_tensors = torch.cat((hidden_states[:,0:m-1], ca_list), dim=1)
    output = torch.cat((prefix_and_ca_tensors, hidden_states[:,last_tokens_offset:]), dim=1)
    return output

# Altered forward pass to replace the LlamaForwardPass of RETRO layers
def RETRO_layer_forward(self, *args, **kwargs): #this combines insights from the intermediate decoding implementation, and the HF transformers implementation
    # Preparation
    hidden_states = args[0]  # usually a tuple where the first element is what we want
    if (len(kwargs) == 0 and len(args) == 1):
        attention_mask = None
        position_ids = torch.arange(args[0].shape[1], dtype=torch.long).unsqueeze(0)
        past_key_value = None
        output_attentions = False
        use_cache = False
    elif (len(kwargs) == 0 and len(args) == 6):
        attention_mask = args[1]
        position_ids = args[2]
        past_key_value = args[3]
        output_attentions = args[4]
        use_cache = args[5]
    else:
        attention_mask = kwargs["attention_mask"]  # should be torch.FloatTensor with  `(batch_size, 1,query_sequence_length, key_sequence_length)`
        position_ids = kwargs["position_ids"]
        past_key_value = kwargs["past_key_value"]
        output_attentions = kwargs["output_attentions"]
        use_cache = kwargs["use_cache"]
    input_ids = self.biollama.model.input_ids_biollama

    # RMS Norm
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    if (hidden_states.device != residual.device): hidden_states = hidden_states.to(residual.device)
    
    # Self-Attention 
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        # **kwargs,
    )
    if (hidden_states.device != residual.device): hidden_states = hidden_states.to(residual.device)
    # Residual Connection
    hidden_states = residual + hidden_states

    # Chunked Cross Attention
    residual = hidden_states
    if hidden_states.shape[0] > 1: # if we are processing prompts in a batch (for eg during training), adapt CCA
        first_prompts_states = cca_forward_true(self, input_ids, hidden_states[0].unsqueeze(0))
        for i in range(1,hidden_states.shape[0]):
            next_prompt_states = cca_forward_true(self, input_ids, hidden_states[i].unsqueeze(0))
            first_prompts_states = torch.cat((first_prompts_states, next_prompt_states), dim=0)
        hidden_states = first_prompts_states
    else:
        hidden_states = cca_forward_true(self, input_ids, hidden_states)
        # hidden_states = cca_forward(self, input_ids)
    hs_shape = hidden_states.shape
    rs_shape = residual.shape
    size_difference = rs_shape[1] - hs_shape[1]
    if size_difference > 0:
        prefix = residual[:,0:size_difference,:]
        if prefix.device != hidden_states.device:  #before we concatenate, check if they are on the same device. if not, move prefix to the same device as hidden_states
            prefix = prefix.to(hidden_states.device) #without this, there have been issues in the past
        hidden_states = torch.cat((prefix, hidden_states), dim=1)
    if (hidden_states.device != residual.device): hidden_states = hidden_states.to(residual.device)
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if (hidden_states.device != residual.device): hidden_states = hidden_states.to(residual.device)
    hidden_states = residual + hidden_states
    outputs = (hidden_states,)
    return outputs

# Helper method that RETRO-fits the passed layer
def RETROfit_layer(layer, layer_id, biollama, training, torch_dtype):
    config = biollama.model.config
    layer.biollama = biollama
    layer.cca_attn = LlamaSdpaAttention(config=config, layer_idx=layer_id)
    layer.pre_cca_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # this gets initiated with hidden_size
    if torch_dtype != torch.float32:
        layer.cca_attn.to(torch_dtype)
        layer.pre_cca_layernorm.to(torch_dtype)        
    layer.cca_attn.to(biollama.device)
    layer.pre_cca_layernorm.to(biollama.device)
    layer.forward = RETRO_layer_forward.__get__(layer)
    return

def load_RETRO_weights(model, RETRO_layer_ids, CCA_state_dict):
    for i, layer in enumerate(model.model.layers): # switch pre-specified decoder layers to be a RETRO layer
            if i in RETRO_layer_ids:#load cca_attn weights
                k_string = "model.layers." + str(i) + ".cca_attn.k_proj.weight"
                k_weight = CCA_state_dict[k_string].clone()
                layer.cca_attn.k_proj.weight.data = k_weight

                q_string = "model.layers." + str(i) + ".cca_attn.q_proj.weight"
                q_weight = CCA_state_dict[q_string].clone()
                layer.cca_attn.q_proj.weight.data = q_weight

                v_string = "model.layers." + str(i) + ".cca_attn.v_proj.weight"
                v_weight = CCA_state_dict[v_string].clone()
                layer.cca_attn.v_proj.weight.data = v_weight

                o_string = "model.layers." + str(i) + ".cca_attn.o_proj.weight"
                o_weight = CCA_state_dict[o_string].clone()
                layer.cca_attn.o_proj.weight.data = o_weight

                pre_cca_layernorm_string = "model.layers." + str(i) + ".pre_cca_layernorm.weight"
                pre_cca_layernorm_weight = CCA_state_dict[pre_cca_layernorm_string].clone()
                layer.pre_cca_layernorm.weight.data = pre_cca_layernorm_weight
                layer = layer.to("cuda")
    return

class BioLlama:
    def __init__(self, model_id, chunk_length, RETRO_layer_ids=[15], training=False, torch_dtype=torch.float32):
        # If quantization is lower than float32 (ie int8 or int4), we need a BitsAndBytesConfig
        if (torch_dtype == "int4"):
            if training: raise Exception("Cannot train with quantization, train unquantized and quantize after training")
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=False)
            torch_dtype=torch.bfloat16
        elif (torch_dtype == torch.int8):
            if training: raise Exception("Cannot train with quantization, train unquantized and quantize after training")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            torch_dtype=torch.ffloat16
        else:
            bnb_config = None

        # Model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch_dtype, quantization_config=bnb_config)
        self.model.config.use_cache = False  # testing this in hopes of less cca issues
        self.model.generation_config.temperature = 0.01
        self.state_dict = self.model.state_dict()

        #RETRO-fit and retrieval preparation
        self.RETRO_layer_ids = RETRO_layer_ids
        self.model.input_ids_biollama = None
        self.chunk_length = chunk_length
        for i, layer in enumerate(self.model.model.layers): # switch pre-specified decoder layers to be a RETRO layer
            if i in RETRO_layer_ids:
                RETROfit_layer(layer, i, self, training, torch_dtype)   
        if not training and model_id.startswith("/home/service/"):
            print(f"LOADING THE 7B BIOLLAMA WEIGHTS FOR CCA")
            CCA_state_dict = load_state_dict('/home/service/BioLlama/utilities/finetuning/biollama_training_output/7/model-00002-of-00003.safetensors')
            load_RETRO_weights(self.model, RETRO_layer_ids, CCA_state_dict)
            del CCA_state_dict

        self.model.old_forward = self.model.forward
        self.model.forward = model_new_forward.__get__(self.model)
        self.query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        # self.query_tokenizer.to("cuda")
        self.query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        # self.query_model.to("cuda:0")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        # self.rerank_tokenizer.to("cuda")
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        # self.rerank_model.to("cuda:0")
        db_faiss, db_json = load_db("medcpt", "RCT200ktrain", "input_segmentation", chunk_length=chunk_length)
        self.db_faiss = db_faiss
        self.db_json = db_json
        self.retrieved_chunk_storage = None

    def generate(self, prompt, max_new_tokens=100):
        if (type(prompt) == list) and (len(prompt)>1):
            padding = True
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
        else: 
            padding = False
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=padding)
        self.model.input_ids_biollama = inputs["input_ids"]
        self.model.prompt_biollama = prompt
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_new_tokens=max_new_tokens, use_cache=False)
        # print(f"generate_ids is {generate_ids}")
        num_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
        return (num_tokens, self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)[0],)
