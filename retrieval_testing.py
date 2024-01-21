from utilities.parse_benchmark import parse_benchmark
from utilities.prompts2 import promptify, promptify_for_judging
from utilities.db_retrieval import gte_FAISS_retrieval, medcpt_FAISS_retrieval, build_lookupchart_medcpt, section_distribution_stats

# benchmark = "PubMedQA"
benchmark = "MedQA"
benchmark_questions, benchmark_answers = parse_benchmark(benchmark=benchmark)


b_start = 21
b_end = b_start +1000
benchmark_questions = benchmark_questions[b_start:min(b_end, len(benchmark_questions))]
benchmark_answers =  benchmark_answers[b_start:min(b_end, len(benchmark_answers))]
# benchmark_questions = ["Which is the main calcium pump of the sarcoplasmic reticulum?"]
# answer = ["Sarcoplasmic reticulum Ca(2+)-ATPase"]

embedding_model = "gte-large"
db_name = "RCT200ktrain"
retrieval_text_mode = "full"
chunk_length = None

print(f"{benchmark} with mode {retrieval_text_mode}")
if chunk_length:
    print(f"Chunk length: {chunk_length}")
if embedding_model == "gte-large":
    chunks = gte_FAISS_retrieval(benchmark_questions, db_name, retrieval_text_mode)
elif embedding_model == "medcpt":
    chunks = medcpt_FAISS_retrieval(benchmark_questions,"RCT200ktrain",retrieval_text_mode=retrieval_text_mode, chunk_length=chunk_length)


#uncomment following for section distribution stats
# print(f"{benchmark} question: {benchmark_questions[0]}")
# print(f"{benchmark} answer: {benchmark_answers[0]}")
# print(f"Retrieved chunk: {chunks[0]}")
# temp = section_distribution_stats(questions=benchmark_questions, chunk_length=chunk_length)
# print(temp)