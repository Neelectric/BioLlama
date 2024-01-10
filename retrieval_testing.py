from utilities.parse_benchmark import parse_benchmark
from utilities.prompts2 import promptify, promptify_for_judging
from utilities.db_retrieval import gte_FAISS_retrieval, medcpt_FAISS_retrieval, build_lookupchart_medcpt, section_distribution_stats

# benchmark = "PubMedQA"
benchmark = "bioASQ_no_snippet"
benchmark_questions, benchmark_answers = parse_benchmark(benchmark=benchmark)


b_start = 110
b_end = b_start +1
benchmark_questions = benchmark_questions[b_start:min(b_end, len(benchmark_questions))]
# benchmark_questions = ["Which is the main calcium pump of the sarcoplasmic reticulum?"]
# answer = ["Sarcoplasmic reticulum Ca(2+)-ATPase"]

db_name = "RCT200ktrain"
retrieval_text_mode = "input_segmentation"
chunk_length = 16

print(f"{benchmark} with mode {retrieval_text_mode}")
# chunks = medcpt_FAISS_retrieval(benchmark_questions,"RCT200ktrain",retrieval_text_mode=retrieval_text_mode, chunk_length=chunk_length)

print("\n-----------------------------------------------------------")
print(f"BioASQ question: {benchmark_questions[0]}")
# print(f"Retrieved chunk: {chunks[0]}")
temp = section_distribution_stats(questions=benchmark_questions, chunk_length=chunk_length)
print(temp)