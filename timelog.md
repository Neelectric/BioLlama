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

## Week 1

### 19 Sept 2023

* *1 hour* Skimmed 2 relevant papers for idea gathering
* *1 hour* Prepared notes/timeline for first meeting
* *1 hour* Read the project guidance notes

### 20 Sept 2023

* *0.5 hours* First meeting with supervisor
* *2 hours* Setup of technologies, reading project advice

### 21 Sept 2023
* *0.5 hours* Researched RETRO-fitting Llama2

### 22 Sept 2023
* *2.5 hours* Revisited RETRO paper and its implementation in depth
* *1.5 hours* Revisited BIOREADER paper, its explanation of retro-fitting Sci-Five, and studied a RETRO implementation on GitHub

### 23 Sept 2023
* *0.5 hours* Watched video and read article on RETRO
* *0.5 hours* Watched interview with Patrick Lewis on RAG
* *1 hour* Familiarised myself with lucidrains/RETRO-pytorch implementation

### 24 Sept 2023
* *2 hours* Fought with lucidrains/RETRO-pytorch implementation and various pip and conda errors

## Week 2

### 25 Sept 2023
* *1.5 hours* Cloned kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference 
* *0.5 hours* Got llama2-7b-q4-quantized CPU inference for docQA working, retrieval enhanced with FAISS-indexed random study from RCT20K dataset

### 26 Sept 2023
* *0.5 hours* Second meeting with supervisor
* *1 hour* Further configutation of technologies, restructuring GitHub project and local folders
* *2 hours* Browsing of possibly relevant studies, discovery of a Llama2-7B biomedical finetune & a GPT4 medQA study

### 27 Sept 2023
* *1.5 hours* Trying various (eventually successful) methods to connect to workstation via SSH
* *1 hour* Trialling functionality of workstation, download and parsing of BioASQ5b

### 28 Sept 2023
* *1 hours* Achieving fully functional batch-inference of Llama-2-70B-4quant, trialling answering of BioASQ5b
* *3 hours* Setting up project architecture (main.py, .yml config, /src, /data etc)
* *1.5 hours* Computing FAISS index of RCT20k and computing nearest neighbors