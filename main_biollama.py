from utilities.biollama import BioLlama
import time
import torch

# questions = ["Which is the main calcium pump of the sarcoplasmic reticulum? Answer:"]
amended_questions = ["The main calcium pump of the sarcoplasmic reticulum is "]
questions = amended_questions
#answers = ["Sarcoplasmic reticulum Ca(2+)-ATPase"] # or "SERCA","serca2"

prompt  = '<QUESTION>A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? \n (A) Ampicillin\n (B) Ceftriaxone\n (C) Ciprofloxacin\n (D) Doxycycline\n (E) Nitrofurantoin</QUESTION>\n<ANSWER> '
prompt2 = '<QUESTION>A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died only after she awoke in the morning. No cause of death was determined based on the autopsy. Which of the following precautions could have prevented the death of the baby? \n (A) Placing the infant in a supine position on a firm mattress while sleeping\n (B) Routine postnatal electrocardiogram (ECG)\n (C) Keeping the infant covered and maintaining a high room temperature\n (D) Application of a device to maintain the sleeping position\n (E) Avoiding pacifier use during sleep</QUESTION>\n<ANSWER> '
prompt3 = "<QUESTION>A mother brings her 3-week-old infant to the pediatrician's office because she is concerned about his feeding habits. He was born without complications and has not had any medical problems up until this time. However, for the past 4 days, he has been fussy, is regurgitating all of his feeds, and his vomit is yellow in color. On physical exam, the child's abdomen is minimally distended but no other abnormalities are appreciated. Which of the following embryologic errors could account for this presentation? \n (A) Abnormal migration of ventral pancreatic bud\n (B) Complete failure of proximal duodenum to recanalize\n (C) Error in neural crest cell migration\n (D) Abnormal hypertrophy of the pylorus\n (E) Failure of lateral body folds to move ventrally and fuse in the midline</QUESTION>\n<ANSWER> "
questions = [prompt, prompt2, prompt3]

db_name = "RCT200ktrain"
retrieval_text_mode = "input_segmentation"

prompt = questions[1]
model_id = 'meta-llama/Llama-2-13b-chat-hf'
chunk_length = 32

time_before_setup = time.time()
BioLlama = BioLlama(model_id=model_id, RETRO_layer_ids=[15], chunk_length=chunk_length, training=True, torch_dtype=torch.float16)
time_before_generation = time.time()
num_tokens, text = BioLlama.generate(prompt=prompt, max_new_tokens=15)

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