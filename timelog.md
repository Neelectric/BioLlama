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

## Week 5 - 
### Monday - 16 Oct 2023
* *2.5 hours* Refactoring of benchmark parsing and prompting
* *1.5 hours* Further research into conception of BioASQ and Task B design
* *1 hour* Adding support for proper snippet consideration in BioASQ
* *0.5 hours* Suspicious accuracy score of 81.66% using automatic judging

### Tuesday - 17 Oct 2023
* *0.5 hours* Fourth Meeting with supervisor
* *0.5 hours* Re-read of LLM-As-Judge
* *0.5 hours* Benchmarking of Llama-2-13B on BioASQ and MedQA-USMLE