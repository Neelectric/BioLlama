# Timelog

* Retrieval enhancement of biomodels in a compute-scarce environment
* By: Neel Rajani, 2514211R
* Supervisor: Dr. Jake Lever

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

## Week 7 - 6 hours total
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

## Week 8 - 2.5 hours total
### Monday - 06 Nov 2023
 * *0.5 hours* Minor bugfixes

 ### Tuesday - 07 Nov 2023
 * *0.5 hours* Seventh meeting with supervisor

 ### Thursday - 09 Nov 2023
 * *0.5 hours* Reading RA-DIT

 ### Friday - 10 Nov 2023
 * *1 hour* Repo refactoring work to make things cleaner

## Week 9 - 3 hours total
### Monday - 13 Nov 2023
 * *0.5 hours* Minor bugfixes

 ### Tuesday - 14 Nov 2023
 * *0.5 hours* Eigth meeting with supervisor
 * *1 hour* Searching and finding OpenAI RAG talk

 ### Saturday - 18 Nov 2023
 * *0.5 hours* Re-reading MedPaLM 1
 * *0.5 hours* Watching 30min video report on issues with MMLU

## Week 10 - 4 hours total
### Monday - 20 Nov 2023
 * *0.5 hours* Minor bugfixes

 ### Tuesday - 21 Nov 2023
 * *0.5 hours* Ninth meeting with supervisor
 * *1 hour* Reading Chain-of-Note paper for improved retrieval incorporation
 * *0.5 hours* Following-up on definition of exact_match metric, DPR paper

 ### Friday - 24 Nov 2023
 * *0.5 hours* Meeting with Dr Sean Macavaney on BioLlama retrieval component

 ### Sunday - 26 Nov 2023
 * *1 hour* Automatic README changes on new benchmark trials

## Week 11 - 6.5 hours total
### Monday - 27 Nov 2023
 * *2 hours* Small changes to automatic README changes

### Tuesday - 28 Nov 2023
 * *0.5 hours* Addition of automatic timelog to README
 * *0.5 hours* Tenth meeting with supervisor

### Wednesday - 29 Nov 2023
 * *1.5 hours* Cloning of RETRO implementations and working through to see how they work

### Friday - 31 Nov 2023
 * *2 hours* Reading flurry of papers from twitter rabbit hole on LLM prompting

## Week 12 - 6 hours total
### Monday - 04 Dec 2023
 * *2 hours* First steps towards simple retrieval system
 * *1 hour* Important changes to promptification

### Tuesday - 05 Dec 2023
* *0.5 hours* Eleventh meeting with supervisor

### Saturday - 09 Dec 2023
* *0.5 hours* Made parse_benchmark.py slightly more modular

### Sunday - 10 Dec 2023
* *1 hour* Reading "REALTIME QA: What’s the Answer Right Now?"
* *1 hour* Further work on promptification & simple retrieval

## Week 13 - 5 hours total
### Monday - 11 Dec 2023
* *1 hour* Further rework of promptify and minor LLM call changes
* *0.5 hours* Cleaning up repository and moving things into subdirectories

### Sunday - 17 Dec 2023
* *2 hours* Starting to look further into Llama2 paper
* *1.5 hours* Re-reading RETRO paper in full
* *1 hour* Revisiting existing RETRO implementations

## Week 14 - 15 hours
### Monday - 18 Dec 2023
* *1 hour* Update on db_build.py to build initial 20k RCT datastore
* *1 hour* Changes to inference.py to make retrieval in prompt possible
* *1.5 hours* Refactor of db_build.py into db_retrieval.py to compartmentalise both building and retrieval

### Tuesday - 19 Dec 2023
* *1 hour* Succesful implementation of Retrieval in Prompt
* *0.5 hours* Twelfth meeting with supervisor
* *1.5 hours* First benchmarks of Llama2 70B without retrieval on MedMCQA, after refactor
* *1.5 hours* Benchmarking of RiP Llama2 70b on MedMCQA, slight performance increase

### Wednesday - 20 Dec 2023
* *2 hours* Finishing Llama2 paper in full

### Friday - 22 Dec 2023
* *1 hour* Further post-refactor benchmarking and RiP work
* *0.5 hours* Adding comments to fileheaders for increased attribution transparency

### Saturday - 23 Dec 2023
* *1 hour* Starting to add MedCPT support

### Sunday - 25 Dec 2023
* *1.5 hours* Creating vectorstores for RCT20k train on GTE-Large and MedCPT in "bomrc", "brc" and "full" settings
* *1 hour* Further benchmarking of RiPLlama and RAGLlama

## Week 15 - 12 hours
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

## Week 16 - 20 hours
### Monday - 01 Jan 2023
* *1 hour* Debugging exllama to get a better feeling for model setup
* *1 hour* Further perusing of RETRO papers

### Tuesday - 02 Jan 2023
* *1 hour* Setup work on finetuning Llama2 7B 

### Wednesday - 03 Jan 2023
* *2 hours* Further work finetuning Llama2 7B with frozen layers

### Thursday - 04 Jan 2023
* *1 hour* Further research on RETRO implementations, T5-base and T5-11B
* *0.5 hours* Full debug of Llama-2-7B-GPTQ 
* *1.5 hours* Deep dive into Llama-2 and RETRO architecture comparison implementations
* *1 hour* New attempt at finetuning different Llama2 variants
* *3.5 hours* Full local downloads of transformers, optimum, AutoGPTQ and PEFT libraries with import rerouting

### Friday - 05 Jan 2023
* *1 hour* Creating Figma diagram comparison of Llama2, BioLlama and RETRO decoders
* *5 hours* First work implementing new BioLlama architecture using local transformers/optimum/AutoGPTQ/PEFT stack
* *1 hour* Further studying of existing RETRO implementations (lucidrains, labml.ai, megatron)
* *0.5 hours* Collating all the different medical lama1/2 finetunes

### Saturday - 06 Jan 2023
* *2.5 hours* Further sketching of CCA changes, and CCA implementation

### Sunday - 07 Jan 2023
* *2 hours* Synchronizing the length of existing input embedding, and retrieved chunk embedding
* *1 hour* Getting BioLlama7B to talk for the first time!

## Week 17 - 24 hours
### Monday - 08 Jan 2023
* *1.5 hours* Changing CCA such retrieved chunks are pruned to be no longer than the existing size of input_ids that the model is attending to

### Tuesday - 09 Jan 2023
* *3.5 hours* Preparing post-break debrief presentation
* *6 hours* Building new functionality to check from which sections chunks are mostly retrieved from
* *2.5 hours* Collecting data on section stats and creating graphs

### Wednesday - 10 Jan 2023
* *4 hours* Further additions to presentation and new benchmarks alongside graphs
* *4.5 hours* More benchmarking of retrieval, further work on presentation and changes to biollama
* *1.5 hours* Preparation of BioLlama training/finetuning

### Thursday - 11 Jan 2023
* *1 hour* Thirteenth meeting with supervisor

## Week 18 - 10.5 hours
### Monday - 15 Jan 2023
* *1 hour* Further efforts fighting with new HF transformers training code of L2 7B

### Tuesday - 16 Jan 2023
* *1 hour* Working on transformers.training_arguments error
* *1 hour* First finetune of Llama 2 with last 8 layers and LM head!

### Wednesday - 17 Jan 2023
* *0.5 hours* Fourteenth meeting with supervisor
* *1.5 hours* Further finetuning accomplishments, saving finetuned checkpoint

### Sunday - 21 Jan 2023
* *1 hour* Revisiting old retrieval and inference pipelines, fixing issues that were introduced in BioLlama
* *1.5 hours* Re-running old Llama2 7B benchmarks with and without retrieval
* *1 hour* Investigating low Llama2 70B MedQA performance
* *2 hours* Fixing low Llama2 70B MedQA performance, debugging retrieval pipeline for minor improvements

## Week 19 - 32 hours
### Monday - 22 Jan 2023
* *1 hour* Revisiting some old benchmark scores
* *1 hour* Investigation on how to translate Llama 2 7B finetuning into BioLlama finetuning
* *0.5 hours* Adding DB pre-loading to BioLlama, gaining 12x speedup from 0.02t/s -->0.3t/s
* *1.5 hours* Efforts to fix BioLlama decoding when max_tokens>32
* *1.5 hours* Further efforts finetuning Llama2 7B when quantized, using MedQA
* *1 hour* Loading MedQA as a benchmark works together with its prompting, working on finetuning too
* *1 hour* Playing around with finetuning parameters and attempts to quantized finetuning (LoRA?)

### Tuesday - 23 Jan 2023
* *1 hour* Adding parse_output.py to handle LLM output writing
* *1 hour* First workable output by finetuned LLM
* *1.5 hours* Playing with wandb
* *1 hour* Further finetuning work

### Wednesday - 24 Jan 2023
* *1 hour* Fifteenth meeting with supervisor
* *1 hour* Investigation into feasibility of PEFT and LoRA for BioLlama
* *0.5 hours* Changing BioLlama to use Llama 2 7B unquantized as a base
* *2.5 hours* Fixing errors associated with this change
* *1.5 hours* Further changes to BioLlama in effort to increase max_new_tokens
* *1.5 hours* Max_new_tokens can now be infinite, further efforts to finally train biollama

### Thursday - 25 Jan 2023
* *1 hour* Bugfixes in trying to get TRL Trainer to accept BioLlama
* *1 hour* Further attempts trying to get TRL Trainer to work with BioLlama
* *1 hour* Modifying a custom .forward() class

### Saturday - 27 Jan 2023
* *1.5 hours* Downloading same SciFive PubMed pretraining files BioReader trains on with gsutil
* *1.5 hours* Debugging of SFT-Trainer Data_collator
* *1 hour* Fighting with lack of padding token in llama tokenizer, dataset issues and training
* *2 hours* Fighting batch sizes, different dimension sizes and biollama .forward() passes not being prepared for multiple sequences
* *1.5 hours* First BioLlama training run??

### Sunday - 28 Jan 2023
* *0.5 hours* Revisiting first biollama training run
* *1 hour* Trying to get first trained biollama to talk, with limited success

## Week 20 - 20 hours
### Monday - 29 Jan 2023
* *1 hour* Further investigation into post-training biollama inference
* *2 hours* Changing BioLlama parameters, adding new params to differentiate from normal layer 15 params
* *2 hours* Hand-debugging constructor for vanilla llama2, to understand when and where weights are initialized, to understand how to add new params with random initilisation
* *1.5 hours* About to add new CCA weights, toying with SDPA attention modules
* *2 hours* Importing LlamaSdpaAttention from transformers to try and initialise it differently
* *1 hour* New SdpaAttention module now trains

### Tuesday - 30 Jan 2023
* *1 hour* Starting some new training run attempts
* *2 hours* Trying to identify point of failure, ie why NaNs are outputted by self-attention in Layer 15
* *2.5 hours* Further troubleshooting, state_dict analysis
* *0.5 hours* Some crying
* *1 hour* Problem seems to be the layernorm modules

### Wednesday - 31 Jan 2023
* *1 hour* New finetuning run of llama2 on new parameters

### Thursday - 01 Feb 2023
* *0.5 hours* Looking at results of new llama2 finetune, comparison to biollama training loss
* *1 hour* Debugging BioLlama decoding to understand layernorm issues

### Friday - 02 Feb 2023
* *1 hour* Looking at an issue in training runs where a retrieved chunk seems to have length 31??

## Week 21 - 11.5 hours
### Tuesday - 06 Feb 2023
* *1 hour* Call with Dr Qiao Jin, lead author of PubMedQA/MedCPT for advice on using his systems
* *1 hour* Comparing L2 finetune vs BioLlama finetune
* *1 hour* Fixing some L2 finetune inference errors

### Wednesday - 07 Feb 2023
* *1 hour* 1k question inference on MedQA with L2 finetune
* *2 hours* Playing with BioLlama inference, further L2 finetuning
* *1 hour* Sixteenth meeting with supervisor
* *1.5 hours* Cleaning up BioLlama.py, further benchmarking
* *1 hour* Small changes to prompting, parse benchmark, parse output, further inference trials

### Thursday - 08 Feb 2023
* *0.5 hours* Inspecting Llama2 20 epoch run, Starting new 10 epoch BioLlama run
* *0.5 hours* Inspecting results of 10 epoch BioLlama run, freeing up 500Gb disk space

### Saturday - 10 Feb 2023
* *1 hour* Starting to edit RETRO-fit approach

## Week 22 - 47.5 hours
### Monday - 12 Feb 2023
* *1 hour* Further changes to RETRO-fitting
* *1 hour* Major refactor of RETRO-fit, replaying RETROLayer/CCA classes with attributes and methods
* *0.5 hours* Adding support for MedCPT retrieval with no reranker
* *1 hours* Finishing cleaner RETRO-fit, adapting for variable batch sizes
* *2 hours* First time BioLlama gets 30% accuracy on MedQA!!!
* *2.5 hours* Starting to implement true CCA
* *1 hour* Further work implementing true CCA
* *1 hour* True CCA coming along with custom SdpaAttention implementation
* *1 hour* Starting first finetune of BioLlama with true CCA

### Tuesday - 13 Feb 2023
* *1 hour* Benchmarking of BioLlama with true CCA
* *1 hour* 100 question BioLlama with true CCA gets 0.35???
* *1.5 hours* Further work on quantization support for float16/bfloat16
* *1.5 hours* Further work on true CCA, pretty graphs
* *1.5 hours* More work on pretty diagrams
* *1 hour* Trying to optimise BioLlama true CCA
* *1.5 hours* Making batch inference faster, further benchmarking
* *1 hour* Coding in bed (low-key kind of fun, should do this more often)

### Wednesday - 14 Feb 2023
* *1 hour* Achieving 4bit quantization of biollama
* *1 hour* Further benchmarking of biollama
* *1 hour* Diss write-up work on key contributions
* *0.5 hours* Seventeenth meeting with supervisor
* *1.5 hours* Further work refining key contributions, graphs and bibtex
* *1.5 hours* Structure of diss, thinking of a new title, Background section

### Thursday - 15 Feb 2023
* *1 hour* Working on BioLlama-70B

### Friday - 16 Feb 2023
* *0.5 hours* Debugging and restarting BioLlama-70B inference
* *1 hour* Fixing BioLlama batch inference to re-add support for training
* *1 hour* 5k step BioLlama-7B-float32 training run, trying to piece things together
* *3 hours* Trying to get BioLlama-7B to train with all layers in torch.float16
* *1.5 hours* Fighting full layer finetunes, BioLlama-70B, and finally crying a little

### Saturday - 17 Feb 2023
* *0.5 hours* Reviewing some inference results
* *2.5 hours* Starting to flesh out diss introduction, some background sections

### Sunday - 18 Feb 2023
* *6.5 hours* Further work on the diss introduction, explaining transformers in background chapter
* *2.5 hours* Reworking images in the diss, formatting, referencing figures and acknowledgements

## Week 23 - 30 hours
### Monday - 19 Feb 2023
* *2 hours* Researching self-attention and cross-attention explanations
* *0.5 hours* Further inference tests

### Tuesday - 20 Feb 2023
* *2.5 hours* Trying to fix Nvidia drivers that a coworker blew up
* *1 hour* Working on improving understanding of BioLlama

### Wednesday - 21 Feb 2023
* *0.5 hours* Eighteenth meeting with supervisor
* *2.5 hours* Trying to debug and fix NVIDIA drivers as destroyed on workstation by coworker

### Thursday - 22 Feb 2023
* *2 hours* Reading multiple Med-RAG papers

### Friday - 23 Feb 2023
* *2.5 hours* Further work on background section of diss, different benchmark examples
* *1.5 hours* Fixing BioASQ with snippets, adding support for enriched datasets
* *1 hour* Readding support for LLM_as_judge

### Sunday - 25 Feb 2023
* *13.5 hours* Close to finished with introduction and background
* *0.5 hours* Some minor benchmarking of Llama-2 on bioASQ

## Week 24 - 53 hours
### Monday - 26 Feb 2023
* *8.5 hours* Further work on background, some inference tests
* *2 hours* First benchmarks of BioLlama-7B/13B/70B on PubMedQA
* *2 hours* Further work on explaining attention and cross attention
* *2 hours* Changes to CCA figure, starting to explain RETRO in Background

### Tuesday - 27 Feb 2023
* *2 hours* First benchmarks of BioLlama-70B/13B/7B on BioASQ
* *5 hours* Adding explanations of all benchmarks to Background, removing Retrieval in Prompt (RiP)
* *1 hour* Thorough proof-reading of progress so far, editing of benchmark explanations
* *0.5 hours* Refining proposed research direction
* *1.5 hours* Describing PMA and PMC
* *1.5 hours* Explaining CCA and RETRO with figure

### Wednesday - 28 Feb 2023
* *1 hour* Revisiting autofaiss and PMA index
* *1.5 hours* Further work on PMA indices
* *0.5 hours* Nineteenth meeting with supervisor

### Thursday - 29 Feb 2023
* *3 hours* Working on PMA indexing and retrieval
* *1 hours* Retrieval attempts

### Friday - 01 March 2023
* *1.5 hours* Revisiting PMA retrieval
* *0.5 hours* PMC pretraining
* *0.5 hours* Dissertation feeback
* *2 hours* Revisiting finetuning, MedQA BioLlama-7B inference
* *2 hours* Adding support for MedQA-4 option, differentiating into MedQA-4 and MedQA-5
* *1 hours* Adding proper support for MedMCQA train/dev/test splits

### Saturday - 02 March 2023
* *1 hour* Checking results of MedMCQA finetune 

### Sunday - 03 March 2023
* *1.5 hours* New finetuning runs on MedQA, new inference results of vanilla/finetuned BioLlama in varying sizes
* *0.5 hours* Finishing forgotten meeting minutes for three meetings
* *1 hour* New inference results on MedQA-4 and 5
* *3 hours* Adding support for PubMedQA finetuning
* *3.5 hours* Adding support for BioASQ finetuning, further inference results
* *2 hours* Fixing error in BioASQ finetuning 

## Week 25 - 22.5 hours
### Monday - 04 March 2023
* *0.5 hours* New inference results
* *2.5 hours* Adding support for BioLlama-13B finetuning in bfloat16, starting first finetune and inference tests

### Tuesday - 05 March 2023
* *3.5 hours* Re-visiting 13b finetuning, new results on 2 epoch training
* *2.5 hours* Further 2 epoch training, visualising BioLlama13b finetune improvements

### Wednesday - 06 March 2023
* *2.5 hours* New 2 epoch training and inference results
* *0.5 hours* Twentieth meeting with supervisor
* *2 hours* Fixing old zero-shot results

### Thursday - 07 March 2023
* *1.5 hours* Fixing introuction, re-adding support for zero-shot benchmarking
* *1 hour* New zero-shot results

### Friday - 08 March 2023
* *3 hours* Further work fixing background section

### Saturday - 09 March 2023
* *10 hours* Fixing Intro and Background, new MedMCQA and zs results

### Sunday - 10 March 2023
* *3 hours* Further work on Background, graphs

## Week 26 - hours
### Monday - 11 March 2023
* *13 hours* Starting methods section, how to prompt, mark and benchmark pipeline

### Tusday - 12 March 2023
* *9 hours* Further work on methods section, fleshing out BioLlama design and training
* *2 hours* Further 13B 2 epoch finetuning attempts

### Wednesday - 13 March 2023 
* *0.5 hours* Twenty-first meeting with supervisor

### Saturday - 16 March 2023
* *2.5 hours* Starting Results / Evaluation, re-wording research questions, starting graphs

### Sunday - 17 March 2023
* *6 hours* Writing up the first three research questions

### Monday - 18 March 2023
* *12.5 hours* Writing up the last three research questions

### Tuesday - 19 March 2023
* *2.5 hours* Going over feedback, restructuring, editing Results / Evaluation
* *7 hours* Writing up Discussion / Limitations and Conclusion

### Wednesday - 20 March
* *2 hours* General editing
* *0.5 hours* Twenty-second meeting with supervisor