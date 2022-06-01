from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "Jarvis, Play music.",
    "Hey Jarvis, hit some music.",
]
model_name = "bert-base-nli-mean-tokens"
model = SentenceTransformer(model_name)
sentence_vecs = model.encode(sentences)

results = cosine_similarity([sentence_vecs[0]], sentence_vecs[1:])[0][0]
print(results*100)
