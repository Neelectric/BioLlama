# Part of the BioLlama library
# Written by Neel Rajani
# Main file to serve as an abstract entry point into BioLlama inference
# Hugely aided by https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb

import torch
import time

local_transformers = False
if local_transformers:
    from .finetuning.cti.transformers.transformers.src.transformers.models.auto import (
        AutoTokenizer,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
    )
    from .finetuning.cti.transformers.transformers.src.transformers.models.llama.modeling_llama import (
        LlamaRMSNorm,
    )
else:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
    )
    from transformers.models.llama.modeling_llama import LlamaSdpaAttention
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

from .db_retrieval import medcpt_FAISS_retrieval, load_db

# InstructRetro Paper suggests random initialisation of RETRO CCA layer weights
class CCA(torch.nn.Module):
    def __init__(self, biollama, layer, training):
        super().__init__()
        self.training = training  # apparently by default it thinks we're training, but does it even have an effect??
        self.pre_CCA_layernorm = None
        self.biollama = biollama
        self.config = biollama.model.config
        self.layer = layer
        self.layer.CCA_attn = LlamaSdpaAttention(config=self.config, layer_idx=self.biollama.RETRO_layer_ids[0])
        #move it to gpu
        self.layer.CCA_attn.to(biollama.device)
        if training:
            for module in self.layer.CCA_attn.modules():
                if isinstance(module, torch.nn.Linear):
                    #module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                    #module.bias.data.zero_()
                    # module.weight.data = torch.randn(module.weight.data.shape)
                    # may not need this random initialisation, because it already gets initialised randomly?
                    pass
                    # module.bias.data = torch.randn(module.bias.data.shape)
        # move to gpu
        else: 
            #load LlamaSdpaAttention weights
            state_dict = biollama.model.state_dict()
            # for key,val in state_dict.items():
            #     if key == 'model.layers.15.CCA_attn.q_proj.weight':
            #         print(val.device)
            #     if key == 'model.layers.15.CCA_attn.k_proj.weight':
            #         print(val.device)
            #     if key == 'model.layers.15.CCA_attn.v_proj.weight':
            #         print(val.device)
            #     if key == 'model.layers.15.CCA_attn.o_proj.weight':
            #         print(val.device)
            # self.layer.CCA_attn.load_state_dict("utilities/finetuning/biollama_training_output/model-00003-of-00006.safetensors")
        # self.layer.CCA_attn.to(biollama.device)

    def forward(self, input_ids, attention_mask, position_ids, past_key_value, output_attentions, use_cache,):
        embed_tokens = self.biollama.model.base_model.embed_tokens
        if input_ids.shape == torch.Size([32, 1024]):
            pass
        elif input_ids.shape == torch.Size([1024]):
            pass
        else:
            input_ids = [int(element) for element in input_ids[0]]
        chunk_length = self.biollama.chunk_length # pruning input_ids to be the last chunk_length tokens
        if len(input_ids) > chunk_length: pass #  issue with this: difference in num tokens given by MedCPT query tokenizer vs llama2 tokenizer
        input_ids = input_ids[-chunk_length:]
        tokens = self.biollama.tokenizer.decode(input_ids)
        if tokens[4:] == "<s> ": tokens = tokens[4:] # without the leading "<s> "
        retrieved_chunk = medcpt_FAISS_retrieval(
            tokens,
            db_name="RCT200ktrain",
            retrieval_text_mode="input_segmentation",
            chunk_length=self.biollama.chunk_length,
            query_tokenizer=self.biollama.query_tokenizer, # passed as a pre-loaded object to save time
            query_model=self.biollama.query_model, # passed as a pre-loaded object to save time
            rerank_tokenizer=self.biollama.rerank_tokenizer, # passed as a pre-loaded object to save time
            rerank_model=self.biollama.rerank_model, # passed as a pre-loaded object to save time
            top_k=1,
            db_faiss=self.biollama.db_faiss, # passed as a pre-loaded object to save time
            db_json=self.biollama.db_json, # passed as a pre-loaded object to save time
        )
        # retrieved_chunk = '[CLS] sarcoplasmic reticulum ( sr ) ca ( 2 + ) - handling proteins play' #hardcoded while transformers bugs me
        # retrieved_chunk = "and stimulation of sarcoplasmic reticulum calcium atpase. we examined the hemodynamic, echocardiographic, and neurohormonal effects of intravenous istaroxime in patients hospitalized with"

        if type(retrieved_chunk[0]) == list:
            retrieved_chunk = retrieved_chunk[0]
        print("tokens is:")
        print(tokens)
        print("retrieved chunk is:")
        print(retrieved_chunk)
        
        encoded_chunk = self.biollama.tokenizer(retrieved_chunk, return_tensors="pt") # we then use the llama2 tokenizer to encode this chunk
        chunk_input_ids = encoded_chunk.input_ids # get input_ids of tokens of the encoded chunk

        # here i prune the last tokens off, otherwise matmul fails
        unnested_chunk_input_ids = torch.unbind(chunk_input_ids, dim=0)[0] # unnest the chunk_input_ids
        cutoff = len(input_ids) # this is the number of tokens in the sequence with which we performed retrieval, ie 32
        sliced_chunk_input_ids = unnested_chunk_input_ids[0:cutoff] # this slices chunk_input_ids to length chunk_length
        # print(f"sliced_chunk_input_ids is of shape: {sliced_chunk_input_ids.shape}")
        chunk_input_ids = sliced_chunk_input_ids.reshape((1, cutoff))

        # Next, they are embedded using the model's embed_tokens layer
        inputs_embeds = embed_tokens(chunk_input_ids)
        embeds_shape = inputs_embeds.shape
        hidden_states = inputs_embeds
        position_ids = torch.arange(embeds_shape[-2], dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

        hidden_states, self_attn_weights, present_key_value = self.layer.CCA_attn(  #when input_ids or hidden_states has shape [1,33] this line explodes
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
    def __init__(self, id, layer, config, biollama, training):
        super().__init__()
        print(f"Wrapping layer {id} with retro")
        self.layer = layer
        self.training = training  # apparently by default it thinks we're training, but does it even have an effect?
        self.RETRO_id = id  # tagging the RETRO layer with its id to identify it later
        self.biollama = biollama
        self.CCA = CCA(biollama=biollama, layer=layer, training=training)
        self.pre_CCA_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # this gets initiated with hidden_size
        #move it to the gpu
        self.pre_CCA_layernorm.to(biollama.device)
        self.CCA.pre_CCA_layernorm = self.pre_CCA_layernorm

    def forward(self, *args, **kwargs):#this combines insights from the intermediate decoding implementation, and the HF transformers implementation
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

        #loading layernorms from layer 14 in hopes it fixes it
        layer_14_input_layernorm_weight = torch.nn.Parameter(self.biollama.state_dict["model.layers.14.input_layernorm.weight"])
        layer_14_post_attention_layernorm_weight = torch.nn.Parameter(self.biollama.state_dict["model.layers.14.post_attention_layernorm.weight"])
        self.layer.input_layernorm.weight = layer_14_input_layernorm_weight
        self.layer.post_attention_layernorm.weight = layer_14_post_attention_layernorm_weight

        # RMS Norm
        residual = hidden_states
        hidden_states = self.layer.input_layernorm(hidden_states)
        
        # Self-Attention 
        hidden_states, self_attn_weights, present_key_value = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            # **kwargs,
        )

        # Residual Connection
        hidden_states = residual + hidden_states

        # Chunked Cross Attention
        # print(f"Before CCA, hidden_states has shape {hidden_states.shape} and len {len(hidden_states)}")
        residual = hidden_states
        if hidden_states.shape == torch.Size([2, 1024, 4096]):
            hidden_states_0 = self.CCA.forward( # during training, hidden_states can have shape [32,1024,4096]
                input_ids=input_ids[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states_1 = self.CCA.forward( # during training, hidden_states can have shape [32,1024,4096]
                input_ids=input_ids[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = torch.cat((hidden_states_0, hidden_states_1), dim=0)
        else:
            hidden_states = self.CCA.forward( # during training, hidden_states can have shape [32,1024,4096]
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            # pass
        #at this point, hidden_states maxes out at size [1,32,4096]. but eventually, residual grows to sizes like [1,33,4096] and larger
        #so we take the [1,1,4096]th item of residual, and prepend it to hidden_states. very hacky solution, but it works!
        hs_shape = hidden_states.shape
        rs_shape = residual.shape
        size_difference = rs_shape[1] - hs_shape[1]
        if size_difference > 0:
            prefix = residual[:,0:size_difference,:]
            if prefix.device != hidden_states.device:  #before we concatenate, check if they are on the same device. if not, move prefix to the same device as hidden_states
                prefix = prefix.to(hidden_states.device) #without this, there have been issues in the past
            hidden_states = torch.cat((prefix, hidden_states), dim=1)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.layer.post_attention_layernorm(hidden_states)
        hidden_states = self.layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        return outputs

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

def cca_forward(self, input_ids, position_ids):
    embed_tokens = self.biollama.model.base_model.embed_tokens
    if input_ids.shape == torch.Size([32, 1024]):
        pass
    elif input_ids.shape == torch.Size([1024]):
        pass
    else:
        input_ids = [int(element) for element in input_ids[0]]
    chunk_length = self.biollama.chunk_length # pruning input_ids to be the last chunk_length tokens
    if len(input_ids) > chunk_length: pass #  issue with this: difference in num tokens given by MedCPT query tokenizer vs llama2 tokenizer
    input_ids = input_ids[-chunk_length:]
    tokens = self.biollama.tokenizer.decode(input_ids)
    if tokens[:4] == "<s> ": tokens = tokens[4:] # without the leading "<s> "
    retrieved_chunk = medcpt_FAISS_retrieval(
        tokens,
        db_name="RCT200ktrain",
        retrieval_text_mode="input_segmentation",
        chunk_length=self.biollama.chunk_length,
        query_tokenizer=self.biollama.query_tokenizer, # passed as a pre-loaded object to save time
        query_model=self.biollama.query_model, # passed as a pre-loaded object to save time
        rerank_tokenizer=self.biollama.rerank_tokenizer, # passed as a pre-loaded object to save time
        rerank_model=self.biollama.rerank_model, # passed as a pre-loaded object to save time
        top_k=1,
        k=5,
        db_faiss=self.biollama.db_faiss, # passed as a pre-loaded object to save time
        db_json=self.biollama.db_json, # passed as a pre-loaded object to save time
    )
    # retrieved_chunk = '[CLS] sarcoplasmic reticulum ( sr ) ca ( 2 + ) - handling proteins play' #hardcoded while transformers bugs me
    # retrieved_chunk = "and stimulation of sarcoplasmic reticulum calcium atpase. we examined the hemodynamic, echocardiographic, and neurohormonal effects of intravenous istaroxime in patients hospitalized with"

    if type(retrieved_chunk[0]) == list:
        retrieved_chunk = retrieved_chunk[0]
    print("tokens is:")
    print(tokens)
    print("retrieved chunk is:")
    print(retrieved_chunk)
    
    encoded_chunk = self.biollama.tokenizer(retrieved_chunk, return_tensors="pt") # we then use the llama2 tokenizer to encode this chunk
    chunk_input_ids = encoded_chunk.input_ids # get input_ids of tokens of the encoded chunk

    # here i prune the last tokens off, otherwise matmul fails
    unnested_chunk_input_ids = torch.unbind(chunk_input_ids, dim=0)[0] # unnest the chunk_input_ids
    cutoff = len(input_ids) # this is the number of tokens in the sequence with which we performed retrieval, ie 32
    sliced_chunk_input_ids = unnested_chunk_input_ids[0:cutoff] # this slices chunk_input_ids to length chunk_length
    # print(f"sliced_chunk_input_ids is of shape: {sliced_chunk_input_ids.shape}")
    chunk_input_ids = sliced_chunk_input_ids.reshape((1, cutoff))

    # Next, they are embedded using the model's embed_tokens layer
    inputs_embeds = embed_tokens(chunk_input_ids)
    embeds_shape = inputs_embeds.shape
    hidden_states = inputs_embeds
    position_ids = torch.arange(embeds_shape[-2], dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

    hidden_states = self.cca_attn(  #when input_ids or hidden_states has shape [1,33] this line explodes
        hidden_states=hidden_states,
        # attention_mask=attention_mask,
        position_ids=position_ids,
        # past_key_value=past_key_value,
        # output_attentions=output_attentions,
        use_cache=False,
        # **kwargs,
    )
    # the output of any attn forward pass is a tuple of (hidden_states, self_attn_weights, present_key_value), we only want the first one
    hidden_states = hidden_states[0]
    return hidden_states



# Altered forward pass to replace the LlamaForwardPass of RETRO layers
def RETRO_layer_forward(self, *args, **kwargs):
    #this combines insights from the intermediate decoding implementation, and the HF transformers implementation
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

    #loading layernorms from layer 14 in hopes it fixes it
    # layer_14_input_layernorm_weight = torch.nn.Parameter(self.biollama.state_dict["model.layers.14.input_layernorm.weight"])
    # layer_14_post_attention_layernorm_weight = torch.nn.Parameter(self.biollama.state_dict["model.layers.14.post_attention_layernorm.weight"])
    # self.input_layernorm.weight = layer_14_input_layernorm_weight
    # self.post_attention_layernorm.weight = layer_14_post_attention_layernorm_weight

    # RMS Norm
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
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

    # Residual Connection
    hidden_states = residual + hidden_states

    # Chunked Cross Attention
    # print(f"Before CCA, hidden_states has shape {hidden_states.shape} and len {len(hidden_states)}")
    residual = hidden_states
    if hidden_states.shape == torch.Size([2, 1024, 4096]):
        # hidden_states_0 = self.CCA.forward( # during training, hidden_states can have shape [32,1024,4096]
        #     input_ids=input_ids[0],
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=past_key_value,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        # )
        # hidden_states_1 = self.CCA.forward( # during training, hidden_states can have shape [32,1024,4096]
        #     input_ids=input_ids[1],
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=past_key_value,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        # )
        # hidden_states = torch.cat((hidden_states_0, hidden_states_1), dim=0)
        print(f"THIS NEEDS TO BE REVISITED!! YOU HAVEN'T SUFFICIENTLY IMPLEMENTED THIS")
    else:
        # hidden_states = self.CCA.forward( # during training, hidden_states can have shape [32,1024,4096]
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=past_key_value,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        # )
        hidden_states = cca_forward(self, input_ids, position_ids)
        # pass
    #at this point, hidden_states maxes out at size [1,32,4096]. but eventually, residual grows to sizes like [1,33,4096] and larger
    #so we take the [1,1,4096]th item of residual, and prepend it to hidden_states. very hacky solution, but it works!
    hs_shape = hidden_states.shape
    rs_shape = residual.shape
    size_difference = rs_shape[1] - hs_shape[1]
    if size_difference > 0:
        prefix = residual[:,0:size_difference,:]
        if prefix.device != hidden_states.device:  #before we concatenate, check if they are on the same device. if not, move prefix to the same device as hidden_states
            prefix = prefix.to(hidden_states.device) #without this, there have been issues in the past
        hidden_states = torch.cat((prefix, hidden_states), dim=1)
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    outputs = (hidden_states,)
    return outputs

# Helper method that RETRO-fits the passed layer
def RETROfit_layer(layer, layer_id, biollama, training):
    config = biollama.model.config
    layer.biollama = biollama
    layer.cca_attn = LlamaSdpaAttention(config=config, layer_idx=layer_id)
    layer.pre_cca_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # this gets initiated with hidden_size
    # layer.CCA = CCA(biollama=biollama, layer=layer, training=training)
    layer.cca_attn.to(biollama.device)
    layer.pre_cca_layernorm.to(biollama.device)
    layer.forward = RETRO_layer_forward.__get__(layer)
    return

class BioLlama:
    def __init__(self, model_id, chunk_length, RETRO_layer_ids=[15], training=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        self.RETRO_layer_ids = RETRO_layer_ids
        self.model.input_ids_biollama = None
        self.chunk_length = chunk_length
        for i, layer in enumerate(self.model.model.layers): # switch pre-specified decoder layers to be a RETRO layer
            # essentially what does a RETRO layer really need...
            # it needs CCA
            # it needs pre_CCA_layernorm
            if i in RETRO_layer_ids:
                # self.model.model.layers[i] = RETROLayer(id=i, layer=layer, config=self.model.config, biollama=self, training=training)
                
                RETROfit_layer(layer, i, self, training)

                # self.model.model.layers[i].CCA = CCA(biollama=self, layer=layer, training=training)
                # self.model.model.layers[i].pre_CCA_layernorm = LlamaRMSNorm(self.model.config.hidden_size, eps=self.model.config.rms_norm_eps)  # this gets initiated with hidden_size
                # self.model.model.layers[i].pre_CCA_layernorm.to(self.device)
                # self.model.model.layers[i].forward = RETRO_layer_forward.__get__(self.model.model.layers[i])
        
        self.model.old_forward = self.model.forward
        self.model.forward = model_new_forward.__get__(self.model)

        self.query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        self.model.config.use_cache = False  # testing this in hopes of less cca issues
        self.model.generation_config.temperature = 0.01
        db_faiss, db_json = load_db("medcpt", "RCT200ktrain", "input_segmentation", chunk_length=chunk_length)
        self.db_faiss = db_faiss
        self.db_json = db_json
        self.state_dict = self.model.state_dict()

    def generate(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        self.model.input_ids_biollama = inputs["input_ids"]
        self.model.prompt_biollama = prompt
        encoded = self.tokenizer.encode(prompt)
        decoded = self.tokenizer.decode(encoded)
        tokenized = self.tokenizer.tokenize(prompt)
        input_ids = inputs["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_new_tokens=max_new_tokens, use_cache=False)
        num_tokens = len(generate_ids)
        return (num_tokens, self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)[0],)
