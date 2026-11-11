import pandas as pd
import numpy as np
from pathlib import Path
import ast
import string

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

ROOT = Path(__file__).resolve().parent.parent
data_path = ROOT / 'data' / 'processed'
database_path = next(data_path.rglob("movies_processed.csv"))

df = pd.read_csv(database_path)

def normalize_names(names):
    return [name.replace(" ", "_").lower() for name in ast.literal_eval(names)]

def final_str(new_df):
    cast = normalize_names(new_df['cast'])
    crew = normalize_names(new_df['crew'])
    genres = normalize_names(new_df['genres'])
    overview = str(new_df['overview'].lower())
    keywords = normalize_names(new_df['keywords'])

    stroka = " ".join([overview] + cast + crew + genres + keywords)
    return stroka
    # return ''.join(c for c in stroka if c not in string.punctuation) # убрала все знаки пунктуации

df['full_description'] = df.apply(final_str, axis=1)

# Стемминг

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

def tokenize(text):
    tokens = re.findall(r"(?u)\b\w+\b", text.lower())
    res = [token if "_" else stemmer.stem(token) for token in tokens]
    return [word for word in res if word not in stop_words]

df['full_description'] = df['full_description'].apply(tokenize)

df_final = df[['title', 'full_description']]

df_final.to_csv(f"{data_path}/movies_nltk.csv", index=False)