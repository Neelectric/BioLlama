from src.tokenizer import ExLlamaTokenizer
import os

def count_tokens(model_directory, sequence):
    tokenizer_path = os.path.join(model_directory, "tokenizer.model")
    tokenizer = ExLlamaTokenizer(tokenizer_path)
    return tokenizer.num_tokens(sequence)