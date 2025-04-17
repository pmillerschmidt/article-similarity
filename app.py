import streamlit as st
from get_similar_articles import load_data, get_top_similar, model

# --- Load data with caching ---
@st.cache_data
def get_cached_data():
    return load_data()

titles, embeddings, claps = get_cached_data()

# --- UI ---
st.title("📰 Medium Article Recommender")

query = st.text_input("Enter a Medium-style article title:")

if query:
    st.subheader("Top Similar Articles")
    results = get_top_similar(titles, embeddings, query, claps=claps)

    for article in results:
        st.markdown(f"**{article['title']}**")
        st.markdown(f"- Claps: `{article['claps']}`")
        st.markdown(f"- Similarity Score: `{article['score']:.2f}`")
        st.markdown(f"- Adjusted Score: `{article['adjusted_score']:.2f}`")
        st.markdown("---")
