import streamlit as st
import pandas as pd
from pathlib import Path

import joblib
from scipy.sparse import load_npz

from notebooks.model import recommend

PROJECT_ROOT = Path(__file__).resolve().parent
models_path = PROJECT_ROOT / "models"

vectorizer = joblib.load(
    models_path / "vectorizer.joblib"
)

vectors = load_npz(
    models_path / "movie_matrix.npz"
)

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide")

st.title("🎬 Movie Recommender System")

st.markdown("Find your next favorite movie based on the film you already love.")

data = pd.read_csv("data/processed/movies_nltk.csv")
data_descr = pd.read_csv("data/processed/movies_processed.csv")

movie_titles = sorted(data["title"])

selected_movie = st.selectbox(
    "Choose a movie:", movie_titles)

if st.button("Recommend movies"):
    st.subheader("You may also like:")
    for movie, sim in recommend(selected_movie, data, vectors):
        st.markdown(f"### {movie}")
        st.write(data_descr.loc[data_descr['title'] == movie, 'overview'].iloc[0])
        st.markdown(f"*Similarity with {selected_movie}*: {sim:.3f}")

        st.divider()