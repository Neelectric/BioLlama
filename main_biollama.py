from utilities.biollama import BioLlama
import time
import torch
# from utilities.db_retrieval import medcpt_FAISS_retrieval

# chunk1 = "<s> Which is the main calcium p"
# chunk2 = "ump of the sarcoplasmic"
# chunk3 = "reticulum? Answer:"
# chunks = [chunk1, chunk2, chunk3]
# retrieved_chunks = medcpt_FAISS_retrieval( # example 16: '[CLS] sarcoplasmic reticulum ( sr ) ca ( 2 + ) - handling proteins play'
#         chunks,
#         db_name="RCT200ktrain",
#         retrieval_text_mode="input_segmentation",
#         chunk_length=32,
#         # query_tokenizer=self.biollama.query_tokenizer, # passed as a pre-loaded object to save time
#         # query_model=self.biollama.query_model, # passed as a pre-loaded object to save time
#         # rerank_tokenizer=self.biollama.rerank_tokenizer, # passed as a pre-loaded object to save time
#         # rerank_model=self.biollama.rerank_model, # passed as a pre-loaded object to save time
#         top_k=1,
#         k=5,
#         # db_faiss=self.biollama.db_faiss, # passed as a pre-loaded object to save time
#         # db_json=self.biollama.db_json, # passed as a pre-loaded object to save time
#     )
# print(chunks)
# print(retrieved_chunks)

# questions = ["Which is the main calcium pump of the sarcoplasmic reticulum? Answer:"]
amended_questions = ["The main calcium pump of the sarcoplasmic reticulum is "]
questions = amended_questions
#answers = ["Sarcoplasmic reticulum Ca(2+)-ATPase"] # or "SERCA","serca2"

prompt  = '<QUESTION>A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? \n (A) Ampicillin\n (B) Ceftriaxone\n (C) Ciprofloxacin\n (D) Doxycycline\n (E) Nitrofurantoin</QUESTION>\n<ANSWER> '
prompt2 = '<QUESTION>A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died only after she awoke in the morning. No cause of death was determined based on the autopsy. Which of the following precautions could have prevented the death of the baby? \n (A) Placing the infant in a supine position on a firm mattress while sleeping\n (B) Routine postnatal electrocardiogram (ECG)\n (C) Keeping the infant covered and maintaining a high room temperature\n (D) Application of a device to maintain the sleeping position\n (E) Avoiding pacifier use during sleep</QUESTION>\n<ANSWER> '
prompt3 = "<QUESTION>A mother brings her 3-week-old infant to the pediatrician's office because she is concerned about his feeding habits. He was born without complications and has not had any medical problems up until this time. However, for the past 4 days, he has been fussy, is regurgitating all of his feeds, and his vomit is yellow in color. On physical exam, the child's abdomen is minimally distended but no other abnormalities are appreciated. Which of the following embryologic errors could account for this presentation? \n (A) Abnormal migration of ventral pancreatic bud\n (B) Complete failure of proximal duodenum to recanalize\n (C) Error in neural crest cell migration\n (D) Abnormal hypertrophy of the pylorus\n (E) Failure of lateral body folds to move ventrally and fuse in the midline</QUESTION>\n<ANSWER> "
prompt_super_long = "You are an excellently helpful AI assistant that answers biomedical questions. You start all of your responses with <ANSWER> and end them with </ANSWER>, as shown in the following example: <QUESTION>A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? (A) Ampicillin (B) Ceftriaxone (C) Ciprofloxacin (D) Doxycycline (E) Nitrofurantoin</QUESTION><ANSWER> (E) Nitrofurantoin</ANSWER><QUESTION>A 30-year-old African American woman comes to the physician for the evaluation of a dry cough and chest discomfort for the past 3 days. During this period, the patient has had headaches, muscle aches, joint pain, fever, and chills. Ten days ago, she was hiking with her family in Mississippi. The patient has asthma that is treated with an albuterol inhaler. Her mother has a lung disease treated with methotrexate. The patient has smoked one pack of cigarettes daily for the past 10 years. Her temperature is 38°C (100.4°F). Physical examination shows slight wheezes throughout both lung fields. Laboratory studies and urinalysis are positive for polysaccharide antigen. Bronchoalveolar lavage using silver/PAS-staining shows macrophages filled with a dimorphic fungus with septate hyphae. Which of the following is the most likely cause of this patient's symptoms? (A) Legionella pneumophila infection (B) Aspergillus fumigatus infection (C) Pneumocystis pneumonia (D) Histoplasma capsulatum infection (E) Blastomyces dermatitidis infection</QUESTION><ANSWER> "
prompt_super_long_2 = "<QUESTION>A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? (A) Ampicillin (B) Ceftriaxone (C) Ciprofloxacin (D) Doxycycline (E) Nitrofurantoin</QUESTION><ANSWER> (E) Nitrofurantoin</ANSWER><QUESTION>A 59-year-old overweight woman presents to the urgent care clinic with the complaint of severe abdominal pain for the past 2 hours. She also complains of a dull pain in her back with nausea and vomiting several times. Her pain has no relation with food. Her past medical history is significant for recurrent abdominal pain due to cholelithiasis. Her father died at the age of 60 with some form of abdominal cancer. Her temperature is 37°C (98.6°F), respirations are 15/min, pulse is 67/min, and blood pressure is 122/98 mm Hg. Physical exam is unremarkable. However, a CT scan of the abdomen shows a calcified mass near her gallbladder. Which of the following diagnoses should be excluded first in this patient? (A) Acute cholecystitis (B) Gallbladder cancer (C) Choledocholithiasis (D) Pancreatitis (E) Duodenal peptic ulcer</QUESTION><ANSWER>"
questions = [prompt, prompt2, prompt3]

db_name = "RCT200ktrain"
retrieval_text_mode = "input_segmentation"

prompt = questions[0]
model_id = 'meta-llama/Llama-2-70b-chat-hf'
chunk_length = 32

time_before_setup = time.time()
BioLlama = BioLlama(model_id=model_id, 
                    RETRO_layer_ids=[15], 
                    chunk_length=chunk_length, 
                    training=False, 
                    torch_dtype="int4")
time_before_generation = time.time()
num_tokens, text = BioLlama.generate(prompt=prompt, max_new_tokens=50)

time_after = time.time()

print("***Generating***")
print(text)
# actual_response = text[len(prompt):]
# print(actual_response)
# print(f"Actual response length: {len(actual_response)}")
print(f"Number of tokens: {num_tokens}")
print(f"Time taken for setup: {time_before_generation - time_before_setup}")
print(f"Time taken for generation: {time_after - time_before_generation}")
print(f"Tokens per second: {num_tokens/(time_after - time_before_generation)}")
print(f"Time total: {time_after - time_before_setup}")