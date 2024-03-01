# BioLlama
Public repository to accompany the Level 4 Research Project "Retrieval enhancement of biomodels in a compute-scarce environment".

The aim of this project is to enhance models of varying sizes from the Llama2 family through retrieval. Using few-shot prompting techniques, an initial baseline is created to see how much performance can be "tickled" out of vanilla Llama2 in its 7B, 13B and 70B variants. Performance is measured using standard Biomedical QA and OpenQA benchmarks such as BioASQ, MedQA, PubMedQA and MedMCQA. After setting this baseline, a RETRO-fitted version of Llama 2, dubbed "BioLLama" as inspired by the recent "BioReader" paper, is benchmarked. BioLlama uses MedCPT as a retriever to extract 32 token long snippets from RCT200k or PubMedAbstracts, augmenting Llama-2 on designated RETRO layers.

## Results
The table below shows preliminary results, as reported by other papers or recorded by me. Note that in many cases these are out of date, a more accu

 <!-- table -->
| Model                 | Size   | BioASQ5b (snippets)   | PubMedQA   | MedQA-4   | MedQA-5   | MedMCQA   |
|:----------------------|:-------|:----------------------|:-----------|:----------|:----------|:----------|
| **As reported**       |        |                       |            |           |           |           |
| M42                   | 70B    | -                     | -          | 61.5      | -         | 60.9      |
| Med-PaLM              | 540B   | -                     | 79.0       | 67.6      | -         | 57.6      |
| Med-PaLM 2            | 540B   | -                     | 81.8       | 86.5      | -         | 72.3      |
| BIOREADER             | 229.5M | 81.88                 | -          | 42.96     | -         | -         |
| **Produced By Me**    |        |                       |            |           |           |           |
| Llama-2-7B-chat-GPTQ  | 7B     | 91.91                 | 59.49      | 30.9      | 25.6      | 32.1      |
| Llama-2-13B-chat-GPTQ | 13B    | 91.70                 | 73.74      | 36.9      | 31.3      | 37.8      |
| Llama-2-70B-chat-GPTQ | 70B    | 93.4                  | 75.35      | 44.3      | 36.4      | 46.6      |
| BioLlama-7B           | 7B     | 82.34                 | 54.7       |           | 35        | 31.0      |
| BioLlama-13B          | 13B    | 87.02                 | 67.5       |           | 39        | 36.1      |
| BioLlama-70B          | 70B    | 87.45                 | 70.4       |           | 37        | 37.6      |
<!-- table -->

## 🔎 Dissertation
The dissertation is on the following overleaf project: https://www.overleaf.com/read/pvgmnfpxtvby

## ✉ Contacts
* Neel Rajani, Neel.R@web.de

## Changelog
<!-- changelog -->
 * 21:16:34, 01.03.2024 | Llama-2-70B-chat-GPTQ | MedQA-4 |  --> 44.3, 1000 questions

 * 20:48:50, 01.03.2024 | Llama-2-13B-chat-GPTQ | MedQA-4 |  --> 36.9, 1000 questions

 * 07:09:59, 28.02.2024 | BioLlama-70B | MedMCQA |  --> 37.6, 1000 questions

 * 19:58:33, 27.02.2024 | BioLlama-7B | MedMCQA | 30 --> 31.0, 1000 questions

 * 19:58:06, 27.02.2024 | BioLlama-13B | MedMCQA | 24.0 --> 36.1, 1000 questions

 * 19:52:11, 27.02.2024 | BioLlama-13B | MedMCQA |  --> 24.0, 1000 questions

 * 17:46:46, 27.02.2024 | BioLlama-7B | BioASQ5b (snippets) |  --> 82.34, 1000 questions

 * 16:43:29, 27.02.2024 | BioLlama-13B | BioASQ5b (snippets) |  --> 87.02, 1000 questions

 * 15:13:16, 27.02.2024 | BioLlama-70B | BioASQ5b (snippets) |  --> 87.45, 1000 questions (int4)

 * 04:34:59, 27.02.2024 | BioLlama-70B | PubMedQA | 61.0 --> 70.4, 1000 questions

 * 20:36:31, 26.02.2024 | BioLlama-70B | PubMedQA |  --> 61.0, 200 questions (int4)

 * 19:32:05, 26.02.2024 | BioLlama-13B | PubMedQA |  --> 67.5, 200 questions (float16)

 * 17:39:27, 26.02.2024 | BioLlama-7B | PubMedQA |  --> 54.7, 1000 questions (float32)

 * 13:02:36, 26.02.2024 | Llama-2-70B-chat-GPTQ | PubMedQA | 69.8 --> 75.35, 1000 questions

 * 12:34:34, 26.02.2024 | Llama-2-13B-chat-GPTQ | PubMedQA | 46.4 --> 73.74, 1000 questions

 * 12:25:12, 26.02.2024 | Llama-2-7B-chat-GPTQ | PubMedQA | 54.5 --> 59.49, 1000 questions

 * 02:32:15, 24.02.2024 | Llama-2-70B-chat-GPTQ | BioASQ5b (snippets) | 85.41 --> 93.4, 1000 questions

 * 01:58:29, 24.02.2024 | Llama-2-13B-chat-GPTQ | BioASQ5b (snippets) | 78.33 --> 91.70212765957447, 1000 questions

 * 01:34:42, 24.02.2024 | Llama-2-7B-chat-GPTQ | BioASQ5b (snippets) | 73.75 --> 91.91489361702128, 1000 questions

 * 16:54:24, 16.02.2024 | BioLlama-70B | MedQA | --> 37.0, 100 questions

 * 14:51:52, 13.02.2024 | BioLlama | MedQA | 31.0 --> 35, 100 questions

 * 10:34:46, 13.02.2024 | Llama-2-70B-chat-GPTQ | MedMCQA | 46.3 --> 46.6, 1000 questions

 * 00:02:58, 13.02.2024 | Llama-2-13B-chat-GPTQ | MedMCQA | 38.6 --> 37.8, 1000 questions

 * 23:56:04, 12.02.2024 | Llama-2-7B-chat-GPTQ | MedMCQA | 32.4 --> 32.1, 1000 questions

 * 22:25:53, 12.02.2024 | Llama-2-70B-chat-GPTQ | MedQA | 28.59 --> 36.4, 1000 questions

 * 19:33:23, 12.02.2024 | Llama-2-7B-chat-GPTQ | MedQA | 28 --> 25.6, 1000 questions

 * 19:25:00, 12.02.2024 | Llama-2-13B-chat-GPTQ | MedQA | 27.7 --> 31.3 (1*brc RCT200ktrain)

 * 18:33:52, 12.02.2024 | BioLlama | MedQA | 30 --> 31.0 (1*brc RCT200ktrain)

 * 18:19:49, 12.02.2024 | Llama-2-7B-chat-GPTQ | MedQA | 24.7 --> 28.000000000000004 (1*brc RCT200ktrain) only 100 questions!

 * 18:18:55, 12.02.2024 | Llama-2-7B-chat-finetune | MedQA | 25.0 --> 17.0 (1*brc RCT200ktrain) only 100 questions!

 * 10:36:24, 22.01.2024 | MedCPT-Llama | MedQA | 34.2 --> 34.8 (1*brc RCT200ktrain)

 * 09:15:40, 22.01.2024 | MedCPT-Llama | MedMCQA | 46.9 --> 46.1 (1*input_segmentation RCT200ktrain)

 * 00:57:59, 22.01.2024 | Llama-2-70B-chat-GPTQ | MedMCQA | 44.7 --> 46.300000000000004 (1*input_segmentation RCT200ktrain)

 * 18:52:10, 21.01.2024 | MedCPT-Llama | MedQA | 36.0 --> 34.2 (1*input_segmentation RCT200ktrain)

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