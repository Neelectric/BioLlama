from tqdm import tqdm
from Bio import Entrez
import json
from time import time
# open up database/RCT200ktrain.txt as a json file and start reading lines 
filepath = "database/RCT200ktrain.txt"
n_lines = 0
pmids = []
with open(filepath, "r") as file:
        id = ""
        dict = {}
        for line in tqdm(file, desc="Processing chunks"):
            # if n_lines < 20:
            #     n_lines += 1
                # print(line)
            if line.startswith("###"):
                id = line[3:-1]
                pmids.append(id)
# print(pmids)


# Replace with your email address
Entrez.email = "Neel.R@web.de"

# List of PMIDs
# pmid_list = ['24491034', '20497432', '19062107', '19769482', '26077436']  # Replace with your list of PMIDs
pmid_list = pmids
# pmid_list = temp
# Create an empty dictionary to track mesh_term occurrences
mesh_term_counts = {}
time_before = time()
pmid_counter = 0
for pmid in tqdm(pmid_list):
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
    record = Entrez.read(handle)
    handle.close()

    # Extract MeSH terms
    mesh_terms = []
    try:
        article = record["PubmedArticle"][0]
        medline_citation = article["MedlineCitation"]
        mesh_heading_list = medline_citation["MeshHeadingList"]
        # print(mesh_heading_list)
        # for mesh_heading in mesh_heading_list:
        #     descriptor_name_list = mesh_heading["DescriptorName"].split(", ")
        #     mesh_terms.extend(descriptor_name_list)
    except KeyError:
        print(f"No MeSH terms found for PMID: {pmid}")

    for mesh_heading in mesh_heading_list:
        qualifier_name_list = mesh_heading.get("QualifierName", [])
        mesh_terms.extend(qualifier_name_list)
    for term in mesh_terms:
        if term in mesh_term_counts:
            mesh_term_counts[term] += 1
        else:
            mesh_term_counts[term] = 1
    pmid_counter += 1
    if pmid_counter % 10000:
        # dump the dictionary to a file MeSH_term_analysis/MeSH_term_counts.json
        with open("MeSH_term_analysis/MeSH_term_counts.json", "w") as file:
            json.dump(mesh_term_counts, file)

time_after = time()
# Print the mesh_term_counts dictionary
for term, count in mesh_term_counts.items():
    print(f"{term}: {count}")
# add this to a dataframe:
import pandas as pd
df = pd.DataFrame(mesh_term_counts.items(), columns=["MeSH term", "Count"])
print(df)
print(f"Time taken per pmid: {(time_after - time_before)/len(pmid_list):.2f}s")