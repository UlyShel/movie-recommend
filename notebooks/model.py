import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import joblib
from scipy.sparse import save_npz

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = PROJECT_ROOT / "data" / "processed"
database_path = next(data_path.rglob("movies_nltk.csv"))

df = pd.read_csv(database_path)

vectorizer = CountVectorizer(max_features=5000, stop_words='english')
vectors = vectorizer.fit_transform(df['full_description'])
# print(vectors[0].nnz) - количество ненулевых эелементов в 1 фильме -> 30

def recommend(movie, data, vectors, top_n=5):
    index = df[df['title'] == movie].index[0]

    metrics = cosine_similarity(vectors[index], vectors)
    metrics_sorted = sorted(list(enumerate(metrics[0])), key=lambda x: x[1], reverse=True)[1:6] # первые пять фильмов без исходного

    count = 0

    for idx, sim in metrics_sorted:
        if idx == index:
            continue

        yield data.iloc[idx]['title'], sim

        count += 1

        if count == top_n:
            break

test_movies = ['Dead Poets Society',
               'American Beauty',
               'The Hobbit: The Battle of the Five Armies',
               'Shrek Forever After',
               "Madagascar 3: Europe's Most Wanted"]

for movie in test_movies:
    print('\n\n')
    print('Recommended movies for lovers of', movie, 'are:\n')
    for mov, sim in recommend(movie, df, vectors):
        print(f"{mov} — similarity: {sim:.3f}")
# Имеет смысл! Хотя рекомендации моментами очевидные, но лучше скучные, чем совсем неправильные

models_path = PROJECT_ROOT / "models"

models_path.mkdir(parents=True, exist_ok=True)

joblib.dump(
    vectorizer,
    models_path / "vectorizer.joblib"
)

save_npz(
    models_path / "movie_matrix.npz",
    vectors
)