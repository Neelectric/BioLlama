from utilities.parse_benchmark import parse_benchmark, parse_bioASQ_no_snippet, parse_BioASQ_with_snippet, parse_MedQA, parse_PubMedQA, parse_MedMCQA
from utilities.prompts2 import promptify, promptify_for_judging
from utilities.db_retrieval import gte_FAISS_retrieval, medcpt_FAISS_retrieval


benchmark_questions, benchmark_answers = parse_benchmark("bioASQ_no_snippet")

b_start = 10
b_end = 11
print(benchmark_questions[b_start:min(b_end, len(benchmark_questions))])
db_name = "RCT200ktrain"
retrieval_text_mode = "bomrc"
retrieved_chunks = gte_FAISS_retrieval(benchmark_questions[b_start:min(b_end, len(benchmark_questions))], db_name, retrieval_text_mode)