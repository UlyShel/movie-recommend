import pandas as pd
import numpy as np
from ast import literal_eval # строковое представление внастоящий объект
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path_movie = PROJECT_ROOT / 'data' / 'processed'
data_path_credits = PROJECT_ROOT / 'data' / 'raw'

database_path = next(data_path_movie.rglob("movies_first_fixed.csv"))
database_path_2 = next(data_path_credits.rglob("tmdb_5000_credits.csv"))

movies = pd.read_csv(database_path)
credits = pd.read_csv(database_path_2)

data_first = movies.merge(credits, left_on='id', right_on='movie_id', how='inner')
data = data_first[['genres', 'title_x', 'overview', 'keywords', 'cast', 'crew']]
data = data.rename(columns={'title_x': 'title'})

# обрабатываем фичи
def get_name(value): #genre, keywords
    try:
       items = literal_eval(value)

       return [item['name'] for item in items if isinstance(item, dict) and "name" in item]

    except (ValueError, SyntaxError, TypeError):
        return []

def get_top_actors(value):
    try:
        items = literal_eval(value)

        return [item['name'] for item in items[:3] if isinstance(item, dict) and 'name' in item]

    except (ValueError, SyntaxError, TypeError):
        return []

def get_director(value):
    try:
        items = literal_eval(value)
        return [item['name'] for item in items if isinstance(item, dict) and 'name' in item and item.get('job') == 'Director']

    except (ValueError, SyntaxError, TypeError):
        return []

data['genres'] = data['genres'].apply(get_name)
data['keywords'] = data['keywords'].apply(get_name)
data['cast'] = data['cast'].apply(get_top_actors)
data['crew'] = data['crew'].apply(get_director)

#  можно иначе:
# def names(text):
#     return [d['name'] for d in ast.literal_eval(text)]
# data['cast'] = data['cast'].apply(lambda x: names(x)[:3])
# data['crew'] = data['crew'].apply(job=Director) ???

data.to_csv(f"{data_path_movie}/movies_processed.csv", index=False)