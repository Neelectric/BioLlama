from src.utilities import load_benchmark, load_knowledge_db

from sentence_transformers import SentenceTransformer


model = "Llama-2-70B-q4"
benchmark = load_benchmark("BioASQ-training5b")
knowledge_db = load_knowledge_db()
embedding_model = SentenceTransformer("thenlper/gte-large")
responses = []

for question in benchmark:
    question_embedding = embedding_model.encode(question)
    similar_chunks = knowledge_db.get_similar_chunks(question_embedding)
    prompt = question + " " + similar_chunks
    response = model(prompt)
    responses.append(response)