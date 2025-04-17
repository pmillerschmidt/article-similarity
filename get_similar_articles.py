import math
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_file = "title_embeddings.npz"
csv_file = "medium_data.csv"

def load_data():
    if os.path.exists(embedding_file):
        data = np.load(embedding_file, allow_pickle=True)
        titles = data["titles"].tolist()
        embeddings = data["embeddings"]
        claps = data["claps"].tolist()
    else:
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=["title"])
        df = df.drop_duplicates(subset="title")
        titles = df['title'].astype(str).tolist()
        claps = df['claps'].fillna(0).astype(int).tolist()
        embeddings = model.encode(titles, convert_to_numpy=True)
        np.savez(embedding_file, embeddings=embeddings, titles=titles, claps=claps)
    return titles, embeddings, claps

# Compute similarities

def get_top_similar(titles, embeddings, query, claps=None, top_k=5, popularity_coefficient=0.1):
    # Get similarity scores
    query_vec = model.encode(query, convert_to_numpy=True)
    cosine_scores = cosine_similarity([query_vec], embeddings)[0]
    # Init results
    results = []
    for i, title in enumerate(titles):
        if title == query:
            continue
        sim_score = cosine_scores[i]
        clap_boost = math.log1p(claps[i]) if claps else 0
        adjusted_score = sim_score * (1 + popularity_coefficient * clap_boost)
        # Get results
        results.append({
            "title": title,
            "score": sim_score,
            "adjusted_score": adjusted_score,
            "claps": claps[i] if claps else 0
        })
    results = sorted(results, key=lambda x: x["adjusted_score"], reverse=True)
    return results[:top_k]

def main():
    titles, embeddings, claps = load_data()
    query = "How to grow your startup using content marketing"
    print("Top matches for:", query)
    results = get_top_similar(titles, embeddings, query, claps=claps)
    for r in results:
        print(f"- {r['title']}")
        print(f"  Claps: {r['claps']}")
        print(f"  Similarity Score: {r['score']:.2f}")
        print(f"  Adjusted Score: {r['adjusted_score']:.2f}")
        print()

if __name__ == "__main__":
    main()

