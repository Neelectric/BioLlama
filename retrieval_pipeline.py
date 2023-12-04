from inference import inference
import faiss

inference(benchmark="MedMCQA", 
              b_start = 0, 
              b_end = 1, 
              max_new_tokens = 30,
              inference_mode = "std")