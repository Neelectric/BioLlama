# BioLlama
Public repository to accompany the Level 4 Research Project "Retrieval enhancement of biomodels in a compute-scarce environment".

The aim of this project is to enhance models of varying sizes from the Llama2 family with retrieval methods, potentially through RAG, RETRO, FID and other methods. Using zero-shot, one-shot and few-shot prompting techniques, an initial baseline is created to see how much performance can be "tickled" out of vanilla Llama2 in its 7B, 13B and 70B variants. Retrieval then creates an extension on top of this to investigate its downstream impact. Performance is measured using standard Biomedical QA and OpenQA benchmarks such as BioASQ, MedQA and PubMedQA and MedMCQA.

Currently, "RAGLlama" implies that retrieval was done with GTE-Large as an embedding model, "RiPLlama" implies it occured with MedCPT, and BioLlama is a RETRO-fitted Llama2

## Results
The table below shows preliminary results, as reported by other papers or recorded by me

 <!-- table -->
| Model                 | Size   | BioASQ5b snippets   | PubMedQA   | MedQA   | MedMCQA   |
|:----------------------|:-------|:--------------------|:-----------|:--------|:----------|
| BIOREADER             | 229.5M | 81.88               | -          | 42.96   | -         |
| Llama-2-7B-chat-GPTQ  | 7B     | 73.75               | 54.5       | 24.7    | 32.4      |
| Llama-2-13B-chat-GPTQ | 13B    | 78.33               | 46.4       | 27.7    | 38.6      |
| Llama-2-70B-chat-GPTQ | 70B    | 85.41               | 69.8       | 28.59   | 44.7      |
| M42                   | 70B    | -                   | -          | 61.5    | 60.9      |
| Med-PaLM              | 540B   | -                   | 79.0       | 67.6    | 57.6      |
| Med-PaLM 2            | 540B   | -                   | 81.8       | 86.5    | 72.3      |
| RAGLlama              | 70B    |                     |            | 34.0    | 46.3      |
| RiPLlama              | 70B    |                     |            | 36.0    | 46.9      |
| BioLlama              | 70B    |                     |            |         |           |
<!-- table -->

## 🔎 Dissertation
The dissertation is on the following overleaf project: https://www.overleaf.com/read/pvgmnfpxtvby

## ✉ Contacts
* Neel Rajani, Neel.R@web.de

## Changelog
<!-- changelog -->
 * 01:49:52, 26.12.2023 | RiPLlama | MedQA |  --> 36.0(1*brc RCT20ktrain)

 * 19:11:47, 25.12.2023 | RAGLlama | MedQA | 33.0 --> 34.0 (1*bomrc RCT200k)

 * 19:06:31, 25.12.2023 | RAGLlama | MedQA | 33.0 --> 33.0 (1*brc RCT200k)

 * 18:43:21, 25.12.2023 | RAGLlama | MedQA |  --> 33.0 (1*full RCT200k)

 * 13:05:10, 23.12.2023 | RiPLlama | MedMCQA |  --> 46.9

 * 00:15:32, 23.12.2023 | RAGLlama | MedQA | 30.0 --> 20.0

 * 00:07:05, 23.12.2023 | RAGLlama | MedQA | 30.0 --> 30.0

 * 00:03:09, 23.12.2023 | RAGLlama | MedQA | 20.0 --> 30.0

 * 00:01:26, 23.12.2023 | RAGLlama | MedQA | 30.0 --> 20.0

 * 17:13:34, 22.12.2023 | RAGLlama | MedQA |  --> 30.0

 * 2023-12-22 16:37:54 | Llama-2-70B-chat-GPTQ | MedQA | 34.48 --> 28.599999999999998

 * 2023-12-22 16:02:04 | Llama-2-13B-chat-GPTQ | MedQA | 28.27 --> 27.7

 * 2023-12-22 15:49:21 | Llama-2-7B-chat-GPTQ | MedQA | 21.22 --> 24.7

 * 2023-12-22 15:32:43 | Llama-2-7B-chat-GPTQ | MedMCQA | 30.3 --> 32.4

 * 2023-12-22 15:28:15 | Llama-2-7B-chat-GPTQ | MedMCQA |  32.2       --> 30.3

 * 2023-12-22 15:07:53 | Llama-2-7B-chat-GPTQ | MedMCQA | 30.4 --> 32.2

 * 2023-12-19 18:05:44 | Llama-2-13B-chat-GPTQ | MedMCQA | 38.6

 * 2023-12-19 15:07:21 | RAGLlama | MedMCQA | 46.304

 * 2023-12-19 10:33:54 | Llama-2-70B-chat-GPTQ | MedMCQA | 44.4

 * 2023-12-19 01:49:29 | RAGLlama | MedMCQA | 46.1

 * 2023-12-18 22:37:20 | BioLlama | PubMedQA | 00.00







<!-- changelog -->