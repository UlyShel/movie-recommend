# Movie Recommender System

## Overview

A content-based movie recommendation system built with Python and NLP techniques.

The system recommends five movies based on the content similarity of movie descriptions, genres, keywords, cast members, and directors.

## Technologies

- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- CountVectorizer
- TF-IDF
- Cosine Similarity
- Streamlit

## Methodology

1. Data collection and preprocessing using the TMDb 5000 Movie Dataset.
2. Data integration and cleaning.
3. Text preprocessing and feature extraction.
4. Text vectorization using Bag of Words and TF-IDF.
5. Movie similarity calculation using cosine similarity.
6. Recommendation generation.
7. Model comparison and evaluation.

## Results

The project compares CountVectorizer and TF-IDF approaches and examines the quality of generated recommendations.

## Limitations

- The system relies on metadata and textual similarity.
- It does not account for individual user preferences.
- It does not understand semantic relationships between words.
- It may recommend films with similar metadata but different actual viewing experiences.

## Future Improvements

- Experiment with different feature weights.
- Implement semantic embeddings.
- Add collaborative filtering.
- Integrate TMDb API for movie posters.
- Deploy the application.
