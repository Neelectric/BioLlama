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


