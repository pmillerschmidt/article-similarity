import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute similarities
def get_top_similar(titles, title_embeddings, query_title, top_k=5):
    # Encode and move to CPU
    query_vec = model.encode(query_title, convert_to_numpy=True)
    # Compute cosine similarity
    cosine_scores = cosine_similarity([query_vec], title_embeddings)[0]
    # Get top K results (excluding exact match if needed)
    top_indices = cosine_scores.argsort()[::-1]
    results = [(titles[i], cosine_scores[i]) for i in top_indices if titles[i] != query_title][:top_k]
    return results

def main():
    # Path to your saved embeddings
    embedding_file = "title_embeddings.npz"
    csv_file = "medium_data.csv"
    # Load from npz if it exists
    if os.path.exists(embedding_file):
        print("Loading precomputed embeddings from file...")
        data = np.load(embedding_file, allow_pickle=True)
        title_embeddings = data["embeddings"]
        titles = data["titles"].tolist()  # convert to regular Python list
    else:
        print("Loading data from CSV and computing embeddings...")
        df = pd.read_csv(csv_file)
        titles = df['title'].dropna().unique().tolist()
        print(f"Embedding {len(titles)} titles...")
        title_embeddings = model.encode(titles, convert_to_numpy=True)
        print("Saving embeddings to .npz...")
        np.savez(embedding_file, embeddings=title_embeddings, titles=titles)
    # Example
    query = "How to grow your startup using content marketing"
    print("Top matches for:", query)
    for title, score in get_top_similar(titles, title_embeddings, query):
        print(f"{title} — Score: {score:.2f}")

if __name__ == "__main__":
    main()

