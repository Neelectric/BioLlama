from utilities.parse_benchmark import parse_benchmark, parse_bioASQ_no_snippet, parse_BioASQ_with_snippet, parse_MedQA, parse_PubMedQA, parse_MedMCQA
from utilities.prompts2 import promptify, promptify_for_judging
from utilities.db_retrieval import gte_FAISS_retrieval, medcpt_FAISS_retrieval


benchmark_questions, benchmark_answers = parse_benchmark("bioASQ_no_snippet")

b_start = 110
b_end = b_start +100
# print(benchmark_questions[b_start:min(b_end, len(benchmark_questions))])
# print(benchmark_answers[b_start:min(b_end, len(benchmark_answers))])
db_name = "RCT200ktrain"
retrieval_text_mode = "input_segmentation"
chunk_length = 32
retrieved_chunks = medcpt_FAISS_retrieval(benchmark_questions[b_start:min(b_end, len(benchmark_questions))], db_name, retrieval_text_mode, chunk_length=chunk_length)