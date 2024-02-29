from autofaiss import build_index
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import faiss
import torch
import json
from tqdm import tqdm

candidate_chunk_0 = { # its 21 btw
        "d": "19750901",
        "t": "[Biochemical studies on camomile components/III. In vitro studies about the antipeptic activity of (--)-alpha-bisabolol (author's transl)].",
        "a": "(--)-alpha-Bisabolol has a primary antipeptic action depending on dosage, which is not caused by an alteration of the pH-value. The proteolytic activity of pepsin is reduced by 50 percent through addition of bisabolol in the ratio of 1/0.5. The antipeptic action of bisabolol only occurs in case of direct contact. In case of a previous contact with the substrate, the inhibiting effect is lost.",
        "m": "dose-response relationship, drug!|hemoglobins!|hemoglobins!metabolism|hydrogen-ion concentration!|in vitro techniques!|methods!|pepsin a!|pepsin a!antagonists & inhibitors|pepsin a!antagonists & inhibitors*|pepsin a!metabolism|plants, medicinal!|sesquiterpenes!|sesquiterpenes!pharmacology|sesquiterpenes!pharmacology*|spectrophotometry, ultraviolet!|trichloroacetic acid!|tyrosine!|tyrosine!metabolism|"
    }

#load medcpt query tokenizer and model
# abs_12000000 = "The purpose of this randomized, double-blind parallel group study was to compare the safety, tolerability and acceptability of Easyhaler and Turbuhaler dry powder inhalers for the delivery of budesonide 800 microg day(-1) in adult asthmatic patients who had already been treated with inhaled corticosteroids for at least 6 months prior to the study Additionally the efficacy of the products was evaluated. The main objective was to evaluate the systemic safety of budesonide inhaled from Easyhaler (Giona Easyhaler, Orion Pharma, Finland) as determined by serum and urine cortisol measurements. The secondary objective was to compare the tolerability acceptability and efficacy of the two devices in the administration of budesonide. After a 2-week run-in period (baseline), patients were randomized on a 2:1 basis to receive budesonide from Easyhaler (n = 103) or from Turbuhaler (Pulmicort Turbuhaler, AstraZeneca, Sweden) (n = 58) 200 g dose(-1), two inhalations twice daily for 12 weeks. There was no statistically significant change in morning serum cortisol values from baseline to the end of treatment in either group. Urine free cortisol and urine cortisol/ creatinine ratio increased from baseline in both groups. There were no significant differences between the groups in terms of morning serum cortisol, urine cortisol, adverse events or efficacy variables, but Easyhaler was generally considered more acceptable to the patients. In conclusion, at 800 microg day(-1), Giona Easyhaler is as safe and efficacious as Pulmicort Turbuhaler in adult asthmatic patients previously treated with corticosteroids, but more acceptable to patients."


#set up index and models:
# index_path = "/home/service/BioLlama/vectorstores/pma_target/pma_target_knn.index"
index_path = "/home/service/BioLlama/vectorstores/PubMed/knn.index"
pma = faiss.read_index(index_path)
query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
# source_path = "/home/service/BioLlama/vectorstores/pma_source/"
source_path = "/home/service/BioLlama/PubMed/"
num_chunks = 38

#create lookup table
index_counter = 0
raw_index_2_pmid = {}
pubmed_chunk_list = []
for i in tqdm(range(num_chunks)):
    pubmed_chunk = json.load(open(source_path + "pubmed_chunk_" + str(i) + ".json"))
    pubmed_chunk_list.append(pubmed_chunk)
    for key in pubmed_chunk.keys():
        raw_index_2_pmid[index_counter] = key
        index_counter += 1

#helper function to convert raw index to pmid
def raw_index_2_text(raw_index_to_pmid, raw_index):
    pmid = raw_index_to_pmid[raw_index]
    return pubmed_chunk_list[1][pmid]['a']

#prepare query and search
k = 5

#embed tokenized query and search
with torch.no_grad():
    encoded = query_tokenizer(candidate_chunk_0["a"], truncation=True, padding=True, return_tensors="pt", max_length=512)
    embeds = query_model(**encoded).last_hidden_state[:, 0, :]
    distances, indices = pma.search(embeds, k)
    distances = distances.flatten()
    indices = indices.flatten()
print(indices, distances)

#convert raw index to pmid
pmids = [raw_index_2_pmid[index] for index in indices]

print(pmids)
target_final = pmids[0]
print(pubmed_chunk_list[0][target_final])