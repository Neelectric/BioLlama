# from ftplib import FTP
# from tqdm import tqdm
# ftp = FTP('ftp.ncbi.nlm.nih.gov')
# ftp.login()
# ftp.cwd('pub/lu/MedCPT/pubmed_embeddings/')
# ftp.dir()

# file_list = []
# ftp.retrlines('LIST', lambda x: file_list.append(x.split()))

# local_path = "../PubMed/"
# for item in tqdm(file_list):
#     file_type, filename = item[0], item[-1]
#     if not file_type.startswith('d'):  # Check if it's a file, not a directory
#         with open(local_path + filename, "wb") as local_file:
#             ftp.retrbinary("RETR " + filename, local_file.write)

# # file_name = "README.txt"
# # with open("../PubMed/" + file_name, 'wb') as file:
# #     ftp.retrbinary('RETR ' + file_name, file.write)

# ftp.quit()