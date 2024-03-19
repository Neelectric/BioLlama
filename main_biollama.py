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
prompt_super_long = "You are an excellently helpful AI assistant that answers biomedical questions. You start all of your responses with <ANSWER> and end them with </ANSWER>, as shown in the following example: <QUESTION>A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? (A) Ampicillin (B) Ceftriaxone (C) Ciprofloxacin (D) Doxycycline (E) Nitrofurantoin</QUESTION><ANSWER> (E) Nitrofurantoin</ANSWER><QUESTION>A 30-year-old African American woman comes to the physician for the evaluation of a dry cough and chest discomfort for the past 3 days. During this period, the patient has had headaches, muscle aches, joint pain, fever, and chills. Ten days ago, she was hiking with her family in Mississippi. The patient has asthma that is treated with an albuterol inhaler. Her mother has a lung disease treated with methotrexate. The patient has smoked one pack of cigarettes daily for the past 10 years. Her temperature is 38°C (100.4°F). Physical examination shows slight wheezes throughout both lung fields. Laboratory studies and urinalysis are positive for polysaccharide antigen. Bronchoalveolar lavage using silver/PAS-staining shows macrophages filled with a dimorphic fungus with septate hyphae. Which of the following is the most likely cause of this patient's symptoms? (A) Legionella pneumophila infection (B) Aspergillus fumigatus infection (C) Pneumocystis pneumonia (D) Histoplasma capsulatum infection (E) Blastomyces dermatitidis infection</QUESTION><ANSWER> "
prompt_super_long_2 = "<QUESTION>A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? (A) Ampicillin (B) Ceftriaxone (C) Ciprofloxacin (D) Doxycycline (E) Nitrofurantoin</QUESTION><ANSWER> (E) Nitrofurantoin</ANSWER><QUESTION>A 59-year-old overweight woman presents to the urgent care clinic with the complaint of severe abdominal pain for the past 2 hours. She also complains of a dull pain in her back with nausea and vomiting several times. Her pain has no relation with food. Her past medical history is significant for recurrent abdominal pain due to cholelithiasis. Her father died at the age of 60 with some form of abdominal cancer. Her temperature is 37°C (98.6°F), respirations are 15/min, pulse is 67/min, and blood pressure is 122/98 mm Hg. Physical exam is unremarkable. However, a CT scan of the abdomen shows a calcified mass near her gallbladder. Which of the following diagnoses should be excluded first in this patient? (A) Acute cholecystitis (B) Gallbladder cancer (C) Choledocholithiasis (D) Pancreatitis (E) Duodenal peptic ulcer</QUESTION><ANSWER>"
prompt_ultra_long = "Using the following text snippets, answer the question that follows.\n<SNIPPETS>\nTo investigate the association between primary systemic vasculitis (PSV) and environmental risk factors.\nSeventy-five PSV cases and 273 controls (220 nonvasculitis, 19 secondary vasculitis, and 34 asthma controls) were interviewed using a structured questionnaire. Factors investigated were social class, occupational and residential history, smoking, pets, allergies, vaccinations, medications, hepatitis, tuberculosis, and farm exposure in the year before symptom onset (index year). The Standard Occupational Classification 2000 and job-exposure matrices were used to assess occupational silica, solvent, and metal exposure. Stepwise multiple logistic regression was used to calculate the odds ratio (OR) and 95% confidence interval (95% CI) adjusted for potential confounders. Total PSV, subgroups (47 Wegener's granulomatosis [WG], 12 microscopic polyangiitis, 16 Churg-Strauss syndrome [CSS]), and antineutrophil cytoplasmic antibody (ANCA)-positive cases were compared with control groups.\nFarming in the index year was significantly associated with PSV (OR 2.3 [95% CI 1.2-4.6]), with WG (2.7 [1.2-5.8]), with MPA (6.3 [1.9-21.6]), and with perinuclear ANCA (pANCA) (4.3 [1.5-12.7]). Farming during working lifetime was associated with PSV (2.2 [1.2-3.8]) and with WG (2.7 [1.3-5.7]). Significant associations were found for high occupational silica exposure in the index year (with PSV 3.0 [1.0-8.4], with CSS 5.6 [1.3-23.5], and with ANCA 4.9 [1.3-18.6]), high occupational solvent exposure in the index year (with PSV 3.4 [0.9-12.5], with WG 4.8 [1.2-19.8], and with classic ANCA [cANCA] 3.9 [1.6-9.5]), high occupational solvent exposure during working lifetime (with PSV 2.7 [1.1-6.6], with WG 3.4 [1.3-8.9], and with cANCA 3.3 [1.0-10.8]), drug allergy (with PSV 3.6 [1.8-7.0], with WG 4.0 [1.8-8.7], and with cANCA 4.7 [1.9-11.7]), and allergy overall (with PSV 2.2 [1.2-3.9], with WG 2.7 [1.4-5.7]). No other significant associations were found.\n</SNIPPETS>\nYou start all of your responses with <ANSWER> and end them with </ANSWER>, as shown in the following example: \n<QUESTION>Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?</QUESTION>\n<ANSWER> Yes</ANSWER>\nDo not justify your response, respond with only Yes, Maybe or No.\n<QUESTION>Are environmental factors important in primary systemic vasculitis?</QUESTION>\n<ANSWER>"
prompt_medmcqa = """
You start all of your responses with <ANSWER> and end them with </ANSWER>, as shown in the following example: 
<QUESTION> Which of the following is not true for myelinated nerve fibers: 
(1) Impulse through myelinated fibers is slower than non-myelinated fibers 
(2) Membrane currents are generated at nodes of Ranvier 
(3) Saltatory conduction of impulses is seen 
(4) Local anesthesia is effective only when the nerve is not covered by myelin sheath</QUESTION>
<ANSWER> 3</ANSWER>
Select the correct choice for the following question. State nothing other than the index of the correct choice, without brackets.
<QUESTION>Lymph vessel which drain the posterior 1/3 rd of the tongue:
 (1) Basal vessel.
 (2) Marginal vessel.
 (3) Central vessel.
 (4) Lateral vessel.</QUESTION>
<ANSWER>"""
prompt_medmcqa_2 = """You start all of your responses with <ANSWER> and end them with </ANSWER>, as shown in the following example: 
<QUESTION> Which of the following is not true for myelinated nerve fibers: 
(1) Impulse through myelinated fibers is slower than non-myelinated fibers 
(2) Membrane currents are generated at nodes of Ranvier 
(3) Saltatory conduction of impulses is seen 
(4) Local anesthesia is effective only when the nerve is not covered by myelin sheath</QUESTION>
<ANSWER> 3</ANSWER>
Select the correct choice for the following question. State nothing other than the index of the correct choice, without brackets.
<QUESTION>Treatment of choice in traumatic facial nerve injury is:
 (1) Facial sling
 (2) Facial nerve repair
 (3) Conservative management
 (4) Systemic corticosteroids</QUESTION>
<ANSWER>"""
questions = [prompt, prompt2, prompt3, prompt_ultra_long, prompt_medmcqa_2]

db_name = "RCT200ktrain"
retrieval_text_mode = "input_segmentation"

prompt = questions[4]
# model_id = 'meta-llama/Llama-2-7b-chat-hf'
# model_id = "/home/service/BioLlama/utilities/finetuning/biollama_training_output/7/"
model_directory = "/home/service/BioLlama/utilities/finetuning/biollama_training_output/MedQA-4/13/"
chunk_length = 32

time_before_setup = time.time()
BioLlama = BioLlama(model_id=model_directory, 
                    RETRO_layer_ids=[19], 
                    chunk_length=chunk_length, 
                    training=False,
                    torch_dtype=torch.bfloat16)

time_before_generation = time.time()
num_tokens, text = BioLlama.generate(prompt=prompt, max_new_tokens=10)

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