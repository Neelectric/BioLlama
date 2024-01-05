# Timelog

* Retrieval enhancement of biomodels in a compute-scarce environment
* Neel Rajani
* 2514211R
* Jake Lever

## Guidance

* This file contains the time log for your project. It will be submitted along with your final dissertation.
* **YOU MUST KEEP THIS UP TO DATE AND UNDER VERSION CONTROL.**
* This timelog should be filled out honestly, regularly (daily) and accurately. It is for *your* benefit.
* Follow the structure provided, grouping time by weeks.  Quantise time to the half hour.

## Week 1 - 14 hours total

### Tuesday - 19 Sept 2023

* *1 hour* Skimmed 2 relevant papers for idea gathering
* *1 hour* Prepared notes/timeline for first meeting
* *1 hour* Read the project guidance notes

### Wednesday - 20 Sept 2023
* *0.5 hours* First meeting with supervisor
* *2 hours* Setup of technologies, reading project advice

### Thursday - 21 Sept 2023
* *0.5 hours* Researched RETRO-fitting Llama2

### Friday - 22 Sept 2023
* *2.5 hours* Revisited RETRO paper and its implementation in depth
* *1.5 hours* Revisited BIOREADER paper, its explanation of retro-fitting Sci-Five, and studied a RETRO implementation on GitHub

### Saturday - 23 Sept 2023
* *0.5 hours* Watched video and read article on RETRO
* *0.5 hours* Watched interview with Patrick Lewis on RAG
* *1 hour* Familiarised myself with lucidrains/RETRO-pytorch implementation

### Sunday - 24 Sept 2023
* *2 hours* Fought with lucidrains/RETRO-pytorch implementation and various pip and conda errors

## Week 2 - 12.5 hours total

### Monday - 25 Sept 2023
* *1.5 hours* Cloned kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference 
* *0.5 hours* Got llama2-7b-q4-quantized CPU inference for docQA working, retrieval enhanced with FAISS-indexed random study from RCT20K dataset

### Tuesday - 26 Sept 2023
* *0.5 hours* Second meeting with supervisor
* *1 hour* Further configutation of technologies, restructuring GitHub project and local folders
* *2 hours* Browsing of possibly relevant studies, discovery of a Llama2-7B biomedical finetune & a GPT4 medQA study

### Wednesday - 27 Sept 2023
* *1.5 hours* Trying various (eventually successful) methods to connect to workstation via SSH
* *1 hour* Trialling functionality of workstation, download and parsing of BioASQ5b

### Thursday - 28 Sept 2023
* *1 hour* Breakthrough! Achieving fully functional batch-inference of Llama-2-70B-4quant, trialling answering of BioASQ5b

### Friday - 29 Sept 2023
* *1.5 hours* Fighting with FAISS library, adopting own FAISS code to avoid use of external libraries (langchain etc)

### Sunday - 01 Oct 2023
* *2 hours* Re-wrote large parts of exllama library for own use

## Week 3 - 9 hours total

### Monday - 02 Oct 2023
* *2 hours* Achieved batch-inference of 10 BioASQ5b questions on Llama-2-70B at a time
* *1.5 hours* Extended batch-inference from 10 to 100 questions of BioASQ5b
* *1 hour* Extended batch-inference from 100 to 480 questions of BioASQ5b
* *2 hours* Implemented fully automatic eval of responses with LLM-as-judge, scoring 0.4875 accuracy

### Tuesday - 03 Oct 2023
* *0.5 hours* Third meeting with supervisor
* *0.5 hours* Investigation into different few-shot prompting practices

### Sunday - 08 Oct 2023
* *0.5 hours* Drafting E-Mail to Giacomo Frisoni asking for BIOREADER resources
* *1 hour* Starting to read "Judging LLM-as-judge"

## Week 4 - 16.5 hours total

### Tuesday - 10 Oct 2023
* *0.5 hours* Started reading meta-review of recent advances in NLP
* *2 hours* Refactored prompt-building to new file

### Wednesday - 11 Oct 2023
* *1 hour* Attempts to improve judging prompt

### Thursday - 12 Oct 2023
* *1 hour* Further investigations into size of BioASQ5b_factoid
* *2 hours* Refactoring benchmark parsing, achieving first benchmarking with MedQA-USMLE

### Friday - 13 Oct 2023
* *1 hour* Benchmarking on MedQA-USMLE with 1000 questions
* *2 hours* Refactoring of architecture for greater modularity
* *1 hour* Figma graph of architecture for better overview

### Saturday - 14 Oct 2023
* *1 hour* Refactoring of inference script
* *2 hours* Benchmarking on MedQA-USMLE with 1000 questions
* *0.5 hours* Listening to an LLM Podcast
* *1 hour* Further collecting of papers, creation of first overleaf table

### Sunday - 14 Oct 2023
* *1 hour* Creating of first LaTeX table to represent results
* *0.5 hours* Further benchmarking on MedQA-USMLE with 10178 questions

## Week 5 - 14.5 hours total
### Monday - 16 Oct 2023
* *2.5 hours* Refactoring of benchmark parsing and prompting
* *1.5 hours* Further research into conception of BioASQ and Task B design
* *1 hour* Adding support for proper snippet consideration in BioASQ
* *0.5 hours* Suspicious accuracy score of 81.66% using automatic judging

### Tuesday - 17 Oct 2023
* *0.5 hours* Fourth Meeting with supervisor
* *0.5 hours* Re-read of LLM-As-Judge
* *0.5 hours* Benchmarking of Llama-2-13B on BioASQ and MedQA-USMLE

### Wednesday - 17 Oct 2023
* *0.5 hours* Watched tutorial on how to finetune Llama-2-70B on 48GB GPU RAM

### Thursday - 18 Oct 2023
* *0.5 hours* Downloaded M42 Llama-2-70b finetune for quantization

### Saturday - 21 Oct 2023
* *0.5 hours* Further amendment of main results table
* *0.5 hours* Identified and downloaded appropriate Llama-2-7B chat version (ie 4bit 128g actorder=true)
* *0.5 hours* Read MedPalm2 paper's treating of PubMedQA, and PubMedQA paper
* *0.5 hours* Fixed bug where specified llm was ignored and only L2-70B was used for inference in llm.py

### Saturday - 21 Oct 2023
* *2.5 hours* Further efforts to extend coverage of different models under varying benchmarks

### Sunday - 22 Oct 2023
* *1.5 hours* Attempts at bugfixing 7B and 13B inference on all three benchmarks
* *0.5 hours* Further benchmarking of 7B and 13B on MedQA

## Week 6 - 10 hours total
### Monday - 23 Oct 2023
* *0.5 hours* Write up of count_tokens.py utility script to count tokens in prompt
* *2.5 hours* Fixed bug where LLM was not split on GPUs properly, creating graph for initial baselines
* *2 hours* Further reading into ATLAS, RETRO, RETRO++ and FiD
* *0.5 hours* Further research into RETRO implementations from lucidrains and NVIDIA

### Tuesday - 24 Oct 2023
* *0.5 hours* Fifth meeting with supervisor
* *0.5 hours* Further research into 

### Wednesday - 25 Oct 2023
* *0.5 hours* Contact with the M42 team on benchmarking Med42 on BioASQ5b

### Thursday - 26 Oct 2023
* *1 hour* Perusing "Editing Knowledge in LLMs", "Improving QA" and "Biomedical LLMs" by P. Lewis

### Friday - 27 Oct 2023
* *1 hour* Reading Anthropic's LLM evaluation paper

### Saturday - 28 Oct 2023
 * *1 hour* Starting to implement MedMCQA for full model comparisons

## Week 6
### Monday - 30 Oct 2023
 * *1 hour* Reading "Improving qa robustness" by Sebastian Riedel and Patrick Lewis 2020 paper on BioLLM SOTA

 ### Tuesday - 31 Oct 2023
 * *0.5 hours* Sixth meeting with supervisor

 ### Wednesday - 01 Nov 2023
 * *1 hour* Reading blogposts on MoE LLMs

### Thursday - 02 Nov 2023
 * *1 hour* Fixing a massive branch divergence by adding new branch develop
 * *0.5 hours* Added 100 question samples for each benchmark

### Friday - 03 Nov 2023
 * *1 hour* Started a graph explaining a potential classifier system

### Saturday - 04 Nov 2023
 * *1 hour* Continuation of classifier graph, data visualisations on MedMCQA

## Week 7
### Monday - 06 Nov 2023
 * *0.5 hours* Minor bugfixes

 ### Tuesday - 07 Nov 2023
 * *0.5 hours* Seventh meeting with supervisor

## Week 8
### Monday - 13 Nov 2023
 * *0.5 hours* Minor bugfixes

 ### Tuesday - 14 Nov 2023
 * *0.5 hours* Eigth meeting with supervisor

## Week 9
### Monday - 20 Nov 2023
 * *0.5 hours* Minor bugfixes

 ### Tuesday - 21 Nov 2023
 * *0.5 hours* Ninth meeting with supervisor

## Week 10
### Monday - 27 Nov 2023
 * *2 hours* Small changes to automatic README changes

### Tuesday - 28 Nov 2023
 * *0.5 hours* Addition of automatic timelog to README
 * *0.5 hours* Tenth meeting with supervisor

### Monday - 04 Dec 2023
 * *2 hours* First steps towards simple retrieval system
 * *1 hour* Important changes to promptification

### Saturday - 09 Dec 2023
* *0.5 hours* Made parse_benchmark.py slightly more modular

### Sunday - 10 Dec 2023
* *1 hour* Reading "REALTIME QA: What’s the Answer Right Now?"
* *1 hour* Further work on promptification & simple retrieval

### Sunday - 17 Dec 2023
* *2 hours* Starting to look further into Llama2 paper
* *1.5 hours* Re-reading RETRO paper in full
* *1 hour* Revisiting existing RETRO implementations

### Monday - 18 Dec 2023
* *1 hour* Update on db_build.py to build initial 20k RCT datastore
* *1 hour* Changes to inference.py to make retrieval in prompt possible
* *1.5 hours* Refactor of db_build.py into db_retrieval.py to compartmentalise both building and retrieval

### Tuesday - 19 Dec 2023
* *1 hour* Succesful implementation of Retrieval in Prompt
* *0.5 hours* Eleventh meeting with supervisor
* *1.5 hours* First benchmarks of Llama2 70B without retrieval on MedMCQA, after refactor
* *1.5 hours* Benchmarking of RiP Llama2 70b on MedMCQA, slight performance increase

### Wednesday - 20 Dec 2023
* *2 hours* Finishing Llama2 paper in full

### Friday - 22 Dec 2023
* *1 hour* Further post-refactor benchmarking and RiP work

### Saturday - 23 Dec 2023
* *1 hour* Starting to add MedCPT support

### Sunday - 25 Dec 2023
* *1.5 hours* Creating vectorstores for RCT20k train on GTE-Large and MedCPT in "bomrc", "brc" and "full" settings

### Monday - 26 Dec 2023
* *1 hour* Fixing write_to_readme
* *2.5 hours* Extensive work creating vectorstores for RCT200ktrain on GTE-Large and MedCPT in "bomrc", "brc" and "full" settings

### Tuesday - 27 Dec 2023
* *1 hour* Trials in improving retrieval with reranker
* *1 hour* Adding "bc" and "input_segmentation" settings to RCT200ktrain on MedCPT

### Saturday - 30 Dec 2023
* *1 hour* Further modifications on retrieval
* *1.5 hours* Deep diving into exllama's exact model architecture

### Sunday - 31 Dec 2023
* *1 hour* Watching Karpathy's "Intro to LLMs" on YouTube
* *1.5 hours* Further deep diving into exllama
* *1.5 hours* Revisiting all materials on RETRO-fitting (RETRO, BioReader, RETRO++ and InstructRETRO)

### Monday - 01 Dec 2023
* *1 hour* Debugging exllama to get a better feeling for model setup
* *1 hour* Further perusing of RETRO papers

### Tuesday - 02 Dec 2023
* *1 hour* Setup work on finetuning Llama2 7B 

### Wednesday - 03 Dec 2023
* *2 hours* Further work finetuning Llama2 7B with frozen layers

### Thursday - 04 Dec 2023
* *1 hour* Further research on RETRO implementations, T5-base and T5-11B
* *0.5 hours* Full debug of Llama-2-7B-GPTQ 
* *1.5 hours* Deep dive into Llama-2 and RETRO architecture comparison implementations
* *1 hour* New attempt at finetuning different Llama2 variants
* *3.5 hours* Full local downloads of transformers, optimum, AutoGPTQ and PEFT libraries with import rerouting

### Friday - 05 Dec 2023
* *1 hour* Creating Figma diagram comparison of Llama2, BioLlama and RETRO decoders
* *3 hours* First work implementing new BioLlama architecture using local transformers/optimum/AutoGPTQ/PEFT stack
