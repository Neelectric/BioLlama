from ftplib import FTP
ftp = FTP('ftp.ncbi.nlm.nih.gov')
ftp.login()
ftp.cwd('pub/lu/MedCPT/pubmed_embeddings/')
ftp.dir()

file_name = "README.txt"
with open("../PubMed/" + file_name, 'wb') as file:
    ftp.retrbinary('RETR ' + file_name, file.write)

ftp.quit()