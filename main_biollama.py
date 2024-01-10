from utilities.biollama import BioLlama
import time

questions = ["Which is the main calcium pump of the sarcoplasmic reticulum? Answer:"]
#answers = ["Sarcoplasmic reticulum Ca(2+)-ATPase"] # or "SERCA","serca2"

db_name = "RCT200ktrain"
retrieval_text_mode = "input_segmentation"

prompt = questions[0]
model_id = "TheBloke/Llama-2-7b-chat-GPTQ"
chunk_length = 32

time_before_setup = time.time()
BioLlama = BioLlama(model_id=model_id, chunk_length=chunk_length)
time_before_generation = time.time()
num_tokens, text = BioLlama.generate(prompt=prompt, max_length=33)

time_after = time.time()

print("***Generating***")
print(text)
# actual_response = text[len(prompt):]
# print(actual_response)
# print(f"Actual response length: {len(actual_response)}")
print(f"Time taken for setup: {time_before_generation - time_before_setup}")
print(f"Time taken for generation: {time_after - time_before_generation}")
print(f"Tokens per second: {num_tokens/(time_after - time_before_generation)}")
print(f"Time total: {time_after - time_before_setup}")