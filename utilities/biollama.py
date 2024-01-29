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

    # from transformers import LlamaRMSNorm
from .db_retrieval import medcpt_FAISS_retrieval, load_db


if local_transformers == False:
    # the following class is copied directly from huggingface
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
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return self.weight * hidden_states.to(input_dtype)


# InstructRetro Paper suggests random initialisation of RETRO CCA layer weights
class CCA(torch.nn.Module):
    def __init__(self, biollama, layer):
        super().__init__()
        self.training = True  # apparently by default it thinks we're training, but does it even have an effect??
        self.pre_CCA_layernorm = None
        self.model = biollama
        self.config = biollama.model.config
        self.embed_tokens = torch.nn.Embedding(
            self.config.vocab_size, self.config.hidden_size, self.config.pad_token_id
        )
        self.layer = layer

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
    ):
        # we perform chunked cross attention at every decoding step, with sequences that are 16 tokens long?
        if input_ids.shape == torch.Size([32, 1024]):
            # print("input_ids has shape [32,1024]")
            pass
        elif input_ids.shape == torch.Size([1024]):
            pass
        else:
            input_ids = [int(element) for element in input_ids[0]]
        # print(f"input_ids has len {len(input_ids)}")

        # here i should probably prune this to be the last "chunk_length" tokens right?
        # but i guess i need to re-encode these tokens with the MedCPT query tokenizer...
        # because otherwise its using llama token lengths, not MedCPT token lengths
        chunk_length = self.model.chunk_length
        if len(input_ids) > chunk_length:
            # print("we are exceeding chunk_length")
            pass
        input_ids = input_ids[-chunk_length:]
        tokens = self.model.tokenizer.decode(input_ids)
        if tokens[4:] == "<s> ":
             # without the leading "<s> "
            tokens = tokens[4:]
        temp_tokens = self.model.tokenizer.decode(input_ids)
        # print(f"tokens is currently {tokens}")

        # with this unencoded sequence, we then do medCPT FAISS retrieval, returning a chunk
        # to save time, we pass the query tokenizer and model, as well as the pre-loaded db to the retrieval function
        retrieved_chunk = medcpt_FAISS_retrieval(
            tokens,
            db_name="RCT200ktrain",
            retrieval_text_mode="input_segmentation",
            chunk_length=self.model.chunk_length,
            query_tokenizer=self.model.query_tokenizer,
            query_model=self.model.query_model,
            rerank_tokenizer=self.model.rerank_tokenizer,
            rerank_model=self.model.rerank_model,
            top_k=1,
            db_faiss=self.model.db_faiss,
            db_json=self.model.db_json,
        )
        # retrieved_chunk = '[CLS] sarcoplasmic reticulum ( sr ) ca ( 2 + ) - handling proteins play' #hardcoded while transformers bugs me
        # retrieved_chunk = "and stimulation of sarcoplasmic reticulum calcium atpase. we examined the hemodynamic, echocardiographic, and neurohormonal effects of intravenous istaroxime in patients hospitalized with"

        if type(retrieved_chunk[0]) == list:
            retrieved_chunk = retrieved_chunk[0]
        # print(f"retrieved_chunk is currently {retrieved_chunk}")
        
        encoded_chunk = self.model.tokenizer(retrieved_chunk, return_tensors="pt") # we then use the llama2 tokenizer to encode this chunk
        chunk_input_ids = encoded_chunk.input_ids # get input_ids of tokens of the encoded chunk

        # the input sequence/context, which was originally given to model has size 22
        # our chunk has size 27. so we need to prune it to make the residual connection work
        # im pruning the last tokens off...
        unnested_chunk_input_ids = torch.unbind(chunk_input_ids, dim=0)[0] # unnest the chunk_input_ids
        cutoff = len(input_ids) # this is the number of tokens in the sequence with which we performed retrieval, ie 32
        sliced_chunk_input_ids = unnested_chunk_input_ids[0:cutoff] # this slices chunk_input_ids to length chunk_length
        chunk_input_ids = sliced_chunk_input_ids.reshape((1, cutoff))

        # Next, they are embedded using the model's embed_tokens layer
        inputs_embeds = self.embed_tokens(chunk_input_ids)
        embeds_shape = inputs_embeds.shape
        hidden_states = inputs_embeds

        
        # then do pre_CCA_layernorm
        hidden_states = self.pre_CCA_layernorm(hidden_states) # AM I SURE THIS CCA LAYERNORM WORKS? IT GETS INITIALISED TO NONE AT CREATION

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

        #at the moment, tensor a is the retrieved chunk, which is either its natural length (eg 52???), or chunk_length (eg 32)
        #tensor b is the input sequence, which starts at the length of input prompt (eg 19) and then grows
        # we need to make sure that position ids does not exceed our max length
        # so we need to make sure that position_ids is of length chunk_length

        position_ids = torch.arange(
            embeds_shape[-2], dtype=torch.long, device=inputs_embeds.device
        ).unsqueeze(0)
       
        hidden_states, self_attn_weights, present_key_value = self.layer.self_attn(  #when input_ids or hidden_states has shape [1,33] this line explodes
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
    def __init__(self, id, layer, config, biollama):
        super().__init__()
        print(f"Wrapping layer {id} with retro")
        self.layer = layer
        self.training = True  # apparently by default it thinks we're training, but does it even have an effect?
        self.RETRO_id = id  # tagging the RETRO layer with its id to identify it later
        self.biollama = biollama

        # adding the two new components that differentiate BioLlama from vanilla Llama2
        self.pre_CCA_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )  # this gets initiated with hidden_size
        self.CCA = CCA(biollama, layer)
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
        #at this point, hidden_states max out at size [1,32,4096]
        #but eventually, residual grows to sizes like [1,33,4096] and larger
        #so we take the [1,1,4096]th item of residual, and prepend it to hidden_states
        
        #very hacky solution, but it works!
        hs_shape = hidden_states.shape
        rs_shape = residual.shape
        size_difference = rs_shape[1] - hs_shape[1]
        if size_difference > 0:
            prefix = residual[:,0:size_difference,:]
            hidden_states = torch.cat((prefix, hidden_states), dim=1)


        hidden_states = residual + hidden_states

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

# New method that allows us to replace the forward pass on the LlamaForCausalLm object
def new_forward(self, *args, **kwargs):
    # print("in new forward")
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

class BioLlama:
    def __init__(self, model_id, chunk_length, RETRO_layer_ids=[15]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        
        self.model.input_ids_biollama = None
        self.chunk_length = chunk_length
        for i, layer in enumerate(self.model.model.layers):
            # switch pre-specified decoder layers to be a RETRO layers
            if i in RETRO_layer_ids:
                self.model.model.layers[i] = RETROLayer(
                    id=i, layer=layer, config=self.model.config, biollama=self
                )
        self.model.old_forward = self.model.forward
        self.model.forward = new_forward.__get__(self.model)
        # self.model.forward()

        self.query_tokenizer = AutoTokenizer.from_pretrained(
            "ncbi/MedCPT-Query-Encoder"
        )
        self.query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(
            "ncbi/MedCPT-Cross-Encoder"
        )
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            "ncbi/MedCPT-Cross-Encoder"
        )

        self.model.config.use_cache = False  # testing this in hopes of less cca issues
        self.model.generation_config.temperature = 1.0
        db_faiss, db_json = load_db(
            "medcpt", "RCT200ktrain", "input_segmentation", chunk_length=chunk_length
        )
        self.db_faiss = db_faiss
        self.db_json = db_json

    def generate(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        self.model.input_ids_biollama = inputs["input_ids"]
        self.model.prompt_biollama = prompt

        encoded = self.tokenizer.encode(prompt)
        decoded = self.tokenizer.decode(encoded)
        tokenized = self.tokenizer.tokenize(prompt)
        input_ids = inputs["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        generate_ids = self.model.generate(
            inputs.input_ids.to(self.device), max_new_tokens=max_new_tokens, use_cache=False
        )
        num_tokens = len(generate_ids)
        return (
            num_tokens,
            self.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0],
        )
