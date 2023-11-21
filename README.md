# BioLlama
Public repository to accompany the Level 4 Research Project "Retrieval enhancement of biomodels in a compute-scarce environment".

The aim of this project is to enhance models of varying sizes from the Llama2 family with retrieval methods, potentially through RAG, RETRO, FID and other methods. Using zero-shot, one-shot and few-shot prompting techniques, an initial baseline is created to see how much performance can be "tickled" out of vanilla Llama2 in its 7B, 13B and 70B variants. Retrieval then creates an extension on top of this to investigate its downstream impact. Performance is measured using standard Biomedical QA and OpenQA benchmarks such as BioASQ, MedQA and PubMedQA and MedMCQA.

## Results
The table below shows preliminary results, as reported by other papers or recorded by me
|    Model   |  Size  | BioASQ5b snippets  | PubMedQA | MedQA-USMLE | MedMCQA |
|:----------:|:------:|:------------------:|:--------:|:-----------:|:-------:|
|  BIOREADER | 229.5M |        81.88       |     -    |    42.96    |    -    |
|  Llama2 7B |   7B   |        73.75       |   54.5   |    21.22    |   30.4  |
| Llama2 13B |   13B  |        78.33       |   46.4   |    28.27    |   40.2  |
| Llama2 70B |   70B  |        85.41       |   69.8   |    34.48    |   39.1  |
|     M42    |   70B  |          -         |     -    |     61.5    |   60.9  |
|  Med-PaLM  |  540B  |          -         |   79.0   |     67.6    |   57.6  |
| Med-PaLM 2 |  540B  |          -         |   81.8   |     86.5    |   72.3  |
|  RAGLlama  |   70B  |                    |          |             |         |
|  BioLlama  |   70B  |                    |          |             |         |

## 🔎 Dissertation
The dissertation is on the following overleaf project: https://www.overleaf.com/read/pvgmnfpxtvby


## ✉ Contacts
* Neel Rajani, Neel.R@web.de