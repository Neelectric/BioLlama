# the following is a high level overview of CCA
# soooo input comes in, we get this long sequence of tokens right

# chunk length = 32, so we'll split it into bits of 32 tokens... which should be easy given that every input_id refers to 1 token

# so say we have a prompt of length 96
# so sequence length (n) = 96, number of chunks (l) = 3, size (chunk_size) = 32
# we get 
# H1 = prompt[0:32]
# H2 = prompt[32:64]
# H3 = prompt[64:96]

# most of H1 just gets ignored lol

# we then perform retrieval: 
# for BioReader, training k performance is found to peak at 9, and eval k performance is found to be the same around 2<k<15
# for RETRO,     training k performance is found to peak at 2 (compared to 1 or 4), and eval k performance is found to be best at 2?

# for the first l-1 (2) chunks, find the 2 nearest neighbours and their continuations
# so for chunk
# H1, we find neighbours E1,1 and E1,2 with respective continuations and store in E1 = [E1,1 and E1,2]
# H2, we find neighbours E2,1 and E2,2 with respective continuations and store in E2 = [E2,1 and E2,2]

# then we prepare for cross attention, ie create l-1 (2) spliced chunks
# H1+ = prompt[31:63]
# H2+ = prompt[63:95] 

# then cross attention is performed:
# H1+_CA = CA(H1+, E1)
# H2+_CA = CA(H1+, E2)

# finally, we concatenate everything together:
# output = H1[0:-2] concatenated with H1+_CA concatenated with H2+_CA concatenated with H3[1:]
# this hopefully gives the output of size we desire....

def ca(self, hidden_state, neighbour): # Cross-Attention
    query = self.W_query(hidden_state) # prepare query from hidden state
    keys = self.W_key(neighbour) # prepare keys from neighbour
    values = self.W_value(neighbour) # prepare values from neighbour
    keys = self.kview(keys).transpose(1, 2) # prepare keys for attention
    query = self.qview(query).transpose(1, 2) # prepare query for attention
    values = self.vview(values).transpose(1, 2) # prepare values for attention
    attn_output = self._attn(query, keys, values) # apply attention
    attn_output = self.reshape_output(attn_output) # reshape attention output
    return attn_output

def cca(self, hidden_states): # Chunked Cross-Attention
    neighbours = self.retrieval(hidden_states) # retrieve neighbours
    ee_neighbours = self.encode_and_embed(neighbours) # encode and embed neighbours
    first_hs = hidden_states[0:32] # first hidden state remains untouched
    sliced_hs = self.slice_hidden_states(hidden_states) # slice hidden states
    for hidden_state, neighbour in zip(sliced_hs, ee_neighbours):
        cross_attention = self.ca(hidden_state, neighbour) # apply cross attention
        first_hs = torch.cat((first_hs, cross_attention), 0) # concatenate hidden states
    output = self.add_last_chunk(first_hs, hidden_states) # add last chunk (if any)
    return output

def RETRO_layer_forward(self, *args, **kwargs): # .forward of RETRO layers
    # Standard RMS normalization, self-attention, and residual connections
    ...
    residual = hidden_states # prepare residual connection
    hiddden_states = self.pre_cca_layernorm(hidden_states) # normalize
    hidden_states = self.cca(hidden_states) # apply chunked cross-attention
    hidden_states = residual + hidden_states # add residual connection
    # Standard RMS normalization, feed-forward, and residual connections
    ...
    return hidden_states

class BioLlama: # BioLlama model
    def __init__(self, model_id, chunk_length, RETRO_layer_ids=15, training=False):
        setup_biollama(self, model_id, training) # setup the model and tokenizer
        # adds RMSNorm and SdpaAttention modules & customizes .forward()
        RETROfit_layers(self.model.layers, RETRO_layer_ids)
        # if benchmarking a trained model, load RETRO weights
        if not training: load_RETRO_weights(self.model, RETRO_layer_ids) 
        self.model.forward = generate_new_forward(self) # replace model .forward() 
        prepare_medCPT_and_db(self, chunk_length) # attach medCPT and db to model

    def generate(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt") # tokenize inputs
        generate_ids = self.model.generate(inputs.input_ids) # generate output
        return self.tokenizer.batch_decode(generate_ids)[0]