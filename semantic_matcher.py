from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load once (heavy model)
model = SentenceTransformer("all-MiniLM-L6-v2")

def sbert_similarity(text1: str, text2: str) -> float:
    embeddings = model.encode([text1, text2])
    score = cosine_similarity(
        [embeddings[0]], 
        [embeddings[1]]
    )[0][0]
    return score * 100
