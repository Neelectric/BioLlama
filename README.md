# BioLlama
Public repository to accompany the Level 4 Research Project "Retrieval enhancement of biomodels in a compute-scarce environment".

The aim of this project is to enhance models of varying sizes from the Llama2 family through retrieval. Using zero-shot, one-shot and few-shot prompting techniques, an initial baseline is created to see how much performance can be "tickled" out of vanilla Llama2 in its 7B, 13B and 70B variants. Performance is measured using standard Biomedical QA and OpenQA benchmarks such as BioASQ, MedQA, PubMedQA and MedMCQA. After setting this baseline, Llama 2 is retrieval augmented through simple "prompt pre-pending", where chunks of biomedical literature are retrieved from a vectorstore and then pre-pended to the prompt. The embedding model used to create this vectorstore is either GTE-Large, or MedCPT. The performance of these two approaches is denoted as "GTE-Llama" or "MedCPT-LLama". Finally, a RETRO-fitted version of Llama 2, dubbed "BioLLama" as inspired by the recent "BioReader" paper, is benchmarked.

In the past, "RAGLlama" implied that retrieval was done with GTE-Large as an embedding model, "RiPLlama" implied it occured with MedCPT.

## Results
The table below shows preliminary results, as reported by other papers or recorded by me

 <!-- table -->
| Model                    | Size   | BioASQ5b snippets   | PubMedQA   |   MedQA | MedMCQA   |
|:-------------------------|:-------|:--------------------|:-----------|--------:|:----------|
| BIOREADER                | 229.5M | 81.88               | -          |   42.96 | -         |
| Llama-2-7B-chat-GPTQ     | 7B     | 73.75               | 54.5       |   25.6  | 32.1      |
| Llama-2-13B-chat-GPTQ    | 13B    | 78.33               | 46.4       |   31.3  | 37.8      |
| Llama-2-70B-chat-GPTQ    | 70B    | 85.41               | 69.8       |   36.4  | 46.6      |
| M42                      | 70B    | -                   | -          |   61.5  | 60.9      |
| Med-PaLM                 | 540B   | -                   | 79.0       |   67.6  | 57.6      |
| Med-PaLM 2               | 540B   | -                   | 81.8       |   86.5  | 72.3      |
| GTE-Llama                | 70B    |                     |            |   34    | 46.3      |
| MedCPT-Llama             | 70B    |                     |            |   34.8  | 46.1      |
| Llama-2-7B-chat-finetune | 7B     |                     |            |   17    |           |
| BioLlama                 | 7B     |                     |            |   35    |           |
<!-- table -->

## 🔎 Dissertation
The dissertation is on the following overleaf project: https://www.overleaf.com/read/pvgmnfpxtvby

## ✉ Contacts
* Neel Rajani, Neel.R@web.de

## Changelog
<!-- changelog -->
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